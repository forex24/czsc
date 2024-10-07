import glob
import hashlib
import inspect
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from multiprocessing import cpu_count
from pathlib import Path
from typing import List

import czsc
import numpy as np
import pandas as pd
import streamlit as st
from czsc import CzscStrategyBase, CzscTrader, KlineChart, Freq, Operate, Position
from czsc.connectors.research import get_symbols, get_raw_bars
from czsc.utils.bar_generator import freq_end_time
from loguru import logger
from stqdm import stqdm as tqdm
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide", page_title="CTA研究", page_icon="🧭")

os.environ['base_path']='/Users/gxj/策略研究'
os.environ['signals_module_name'] = 'czsc.signals'

# ======================================================================================================================
# 信号工具相关功能
# ======================================================================================================================


@st.cache_resource()
class ShowSignal:
    def __init__(self, **kwargs) -> None:
        self.base_freq = kwargs.get("base_freq")
        self.signal = kwargs.get("signal")
        self.bar_sdt = kwargs.get("bar_sdt")
        self.sdt = kwargs.get("sdt")
        self.edt = kwargs.get("edt")
        self.symbol = kwargs.get("symbol")
        self.kwargs = kwargs
        assert isinstance(self.base_freq, str) and isinstance(self.signal, str)

        bars = get_raw_bars(self.symbol, self.base_freq, self.bar_sdt, self.edt)
        signals_config = czsc.get_signals_config([self.signal])
        sigs = czsc.generate_czsc_signals(bars, signals_config, sdt=self.sdt, df=False)  # type: ignore
        sigs = pd.DataFrame(sigs)
        if "freq" in sigs.columns:
            sigs.drop(columns=["freq"], inplace=True)
        self.cache_keys = list(sigs.iloc[0]["cache"].keys())
        cols = [x for x in sigs.columns if len(x.split("_")) == 3]
        assert len(cols) == 1, f"信号 {self.signal} 有多个匹配项，请检查"
        sigs["match"] = sigs.apply(czsc.Signal(self.signal).is_match, axis=1)
        sigs["text"] = np.where(sigs["match"], sigs[cols[0]], "")
        self.sigs = sigs

        # 在图中绘制指定需要观察的信号
        cache_key = self.kwargs.get("cache_key", None)
        if cache_key and cache_key not in self.cache_keys:
            st.warning(f"cache_key {cache_key} 不存在，请检查")

        n_rows = 4 if cache_key and cache_key in self.cache_keys else 3
        chart = czsc.KlineChart(n_rows=n_rows, height=700)
        chart.add_kline(sigs, self.base_freq)
        chart.add_sma(sigs, row=1, ma_seq=(5, 10, 20), visible=True)
        chart.add_vol(sigs, row=2)
        chart.add_macd(sigs, row=3)
        df1 = sigs[sigs["text"] != ""][["dt", "text", "close", "low"]].copy()
        chart.add_scatter_indicator(
            x=df1["dt"],
            y=df1["low"],
            row=1,
            name="信号",
            mode="markers",
            marker_size=20,
            marker_color="red",
            marker_symbol="triangle-up",
        )
        self.chart = chart
        if cache_key and cache_key in self.cache_keys:
            self.add_cache_indicator(cache_key)

    def add_cache_indicator(self, cache_key: str):
        df = self.sigs.copy()
        df[cache_key] = df["cache"].apply(lambda x: x[cache_key])
        self.chart.add_scatter_indicator(df["dt"], df[cache_key], name=cache_key, row=4, line_width=2)


def update_nxb(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """在给定的 df 上计算并添加后面 n 根 bar 的累计收益列

    收益计量单位：BP；1倍涨幅 = 10000BP

    :param df: 数据，DataFrame结构
    :param kwargs: 参数字典

        - nseq: 考察的bar的数目的列表，默认为 (1, 2, 3, 5, 8, 10, 13)
        - bp: 是否将收益转换为BP，默认为 False

    :return: pd.DataFrame
    """
    if kwargs.copy():
        df = df.copy()

    assert "dt" in df.columns, "必须包含 dt 列，标记K线结束时刻"
    assert "symbol" in df.columns, "必须包含 symbol 列，标记K线所属的品种"
    assert "price" in df.columns, "必须包含 price 列，标记K线结束时刻的可交易价格"

    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values(["dt", "symbol"]).reset_index(drop=True)

    nseq = kwargs.get("nseq", (1, 2, 3, 5, 8, 10, 13))
    for symbol, dfg in df.groupby("symbol"):

        for n in nseq:
            dfg[f"n{n}b"] = dfg["price"].shift(-n) / dfg["price"] - 1
            df.loc[dfg.index, f"n{n}b"] = dfg[f"n{n}b"].fillna(0)

            if kwargs.get("bp", False) is True:
                df[f"n{n}b"] = df[f"n{n}b"] * 10000
    return df


@czsc.disk_cache(path=czsc.home_path, ttl=3600 * 24)
def __generate_signals(signals_config):
    base_freq = czsc.get_signals_freqs(signals_config)[0]
    symbols = get_symbols("A股主要指数")
    bars = get_raw_bars(symbols[0], base_freq, "20150101", "20190101", raw_bars=True, fq="后复权")
    st.info(f"获取到 {len(bars)} 根K线数据")
    dfs = czsc.generate_czsc_signals(bars, signals_config, sdt="20170101", df=True)
    dfs["price"] = dfs["close"]
    dfs = update_nxb(dfs)
    # czsc.update_nbars(dfs, price_col="close", numbers=(1, 2, 5, 10, 20, 30))
    st.info(f"生成信号数据 {len(dfs)} 条")
    return dfs


@st.experimental_fragment
def show_markout(df, factor, **kwargs):
    """Mark Out 分析"""
    sub_title = kwargs.get("sub_title", "Markout分析")
    if sub_title:
        st.subheader(sub_title, divider="rainbow")

    if df[factor].nunique() > 20:
        st.warning("因子分布过于离散，无法进行Markout分析")
        return

    df = df.copy()
    df[factor] = df[factor].astype(str)

    c1, c2, c3 = st.columns([1, 1, 1])
    agg_method = c1.selectbox("聚合方法", ["mean", "median", "盈亏比", "胜率"], index=0, key=f"agg_method_{factor}")
    sdt = pd.to_datetime(c2.date_input("开始时间", value=df["dt"].min()))
    edt = pd.to_datetime(c3.date_input("结束时间", value=df["dt"].max()))
    df = df[(df["dt"] >= sdt) & (df["dt"] <= edt)].copy()

    st.caption(
        f"时间范围：{df['dt'].min().strftime('%Y%m%d')} ~ {df['dt'].max().strftime('%Y%m%d')}；聚合方法：{agg_method}"
    )
    nb_cols = [x for x in df.columns.to_list() if x.startswith("n") and x.endswith("b")]

    if agg_method == "盈亏比":
        agg_method = lambda x: np.mean([y for y in x if y > 0]) / abs(np.mean([y for y in x if y < 0]))

    if agg_method == "胜率":
        agg_method = lambda x: len([y for y in x if y > 0]) / len(x)

    def __markout(dfm, cols):
        if isinstance(cols, str):
            cols = [cols]
        dfy = dfm.groupby(cols).agg({x: agg_method for x in nb_cols}).reset_index()
        dfy["出现次数"] = dfm.groupby(cols).size().values
        dfy["覆盖率"] = dfm.groupby(cols).size().values / len(dfm)
        dfy = dfy[cols + ["出现次数", "覆盖率"] + nb_cols]
        dfy = dfy.style.background_gradient(cmap="RdYlGn_r", subset=nb_cols, axis=None)
        dfy = dfy.background_gradient(cmap="RdYlGn_r", subset=["出现次数"], axis=None)
        dfy = dfy.background_gradient(cmap="RdYlGn_r", subset=["覆盖率"], axis=None)
        dfy = dfy.format({x: "{:.3%}" for x in nb_cols + ["覆盖率"]}, na_rep="MISS")
        return dfy

    dfy1 = __markout(df.copy(), factor)
    st.dataframe(dfy1, use_container_width=True)

    with st.expander("查看品种详细数据", expanded=False):
        symbol = st.selectbox("选择品种", df["symbol"].unique().tolist(), index=0, key=f"markout_symbol1_{agg_method}")
        dfx = __markout(df[df["symbol"] == symbol].copy(), ["symbol", factor])
        st.dataframe(dfx, use_container_width=True)


def show_signal_validation():
    """信号验证页面"""
    conf = st.text_input("请输入信号配置：", value='{"freq": "15分钟", "name": "czsc.signals.cxt_third_bs_V230319"}')
    conf = eval(conf)

    func = czsc.import_by_name(conf["name"])
    codes = inspect.getsource(func)
    st.code(codes, language="python")

    dfs = __generate_signals([conf])
    col = [x for x in dfs.columns if len(x.split("_")) == 3][0]
    show_markout(dfs, col, sub_title="信号验证")


def show_signal_observation():
    """信号观察页面"""
    plotly_config = {
        "scrollZoom": True,
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": [
            "toggleSpikelines",
            "select2d",
            "zoomIn2d",
            "zoomOut2d",
            "lasso2d",
            "autoScale2d",
            "hoverClosestCartesian",
            "hoverCompareCartesian",
        ],
    }

    with st.expander("设置观察参数", expanded=False):
        with st.form(key="my_form_signal"):
            signal = st.text_input("请输入信号：", value="日线_D1N30M120_AMV能量V230603_看多_任意_任意_0")
            col1, col2, col3 = st.columns([1, 1, 1])
            col4, col5, col6 = st.columns([1, 1, 1])
            base_freq = col1.selectbox("选择观察周期：", czsc.utils.sorted_freqs, index=5)
            symbol = col2.selectbox("请选择股票：", get_symbols("ALL"), index=0)
            cache_key = col3.text_input("请输入缓存KEY：")
            bar_sdt = col4.date_input("行情开始日期：", value=pd.to_datetime("2017-01-01"))
            sdt = col5.date_input(
                "观察开始日期：", value=pd.to_datetime("2019-01-01"), min_value=pd.to_datetime(bar_sdt)
            )
            edt = col6.date_input(
                "观察结束日期：",
                value=pd.to_datetime("2021-01-01"),
                min_value=pd.to_datetime(sdt),
                max_value=pd.to_datetime("2022-01-01"),
            )
            submitted = st.form_submit_button(label="开始观察")

    if not submitted:
        st.warning("请设置观察参数")
        st.stop()

    ss = ShowSignal(
        base_freq=base_freq,
        symbol=symbol,
        signal=signal.strip('"'),
        bar_sdt=bar_sdt,
        sdt=sdt,
        edt=edt,
        cache_key=cache_key,
    )
    st.caption(f"cache_keys: {ss.cache_keys}")
    st.plotly_chart(ss.chart.fig, use_container_width=True, config=plotly_config)


def show_signal_functions():
    selected = option_menu(
        "",
        ["信号解析", "信号观察", "信号验证"],
        icons=["bi-film", "bi-mortarboard", "bi-mortarboard"],
        menu_icon="",
        default_index=0,
        orientation="horizontal",
    )
    parser = czsc.SignalsParser()

    if selected == "信号解析":
        st.subheader("一、信号值转配置")
        signals_seq = st.text_input("请输入信号：", value="日线_D1#SMA#5_BS3辅助V230319_三卖_均线新低_任意_0")
        res = parser.parse([signals_seq.strip('"')])[0]
        st.write("配置：", res)
        if res and res["name"].startswith("czsc.signals"):
            st.write(f"文档：https://czsc.readthedocs.io/en/latest/api/{res['name']}.html")

        st.divider()
        st.subheader("二、配置转信号KEY")
        conf_example = {
            "freq": "日线",
            "di": 1,
            "ma_type": "SMA",
            "timeperiod": "5",
            "name": "czsc.signals.cxt_third_bs_V230319",
        }
        conf = st.text_input("请输入配置：", value=f"{conf_example}")
        st.write("信号：", parser.config_to_keys([eval(conf)]))

    if selected == "信号观察":
        show_signal_observation()

    if selected == "信号验证":
        show_signal_validation()


# ======================================================================================================================
# 以下是回放的代码
# ======================================================================================================================
class JsonStreamStrategy(CzscStrategyBase):
    """读取 streamlit 传入的 json 策略，进行回测"""

    @property
    def positions(self) -> List[Position]:
        """返回当前的持仓策略"""
        json_strategies = self.kwargs.get("json_strategies")
        assert json_strategies, "请在初始化策略时，传入参数 json_strategies"
        positions = []
        for _, pos in json_strategies.items():
            pos["symbol"] = self.symbol
            positions.append(Position.load(pos))
        return positions


def show_trader(trader: CzscTrader, files, max_k_num=300):
    from czsc.utils.ta import MACD

    if not trader.freqs or not trader.kas or not trader.positions:
        st.error("当前 trader 没有回测数据")
        return

    freqs = trader.freqs
    tabs = st.tabs(freqs + ["回测记录", "策略详情"])

    i = 0
    for freq in freqs:
        c = trader.kas[freq]
        sdt = c.bars_raw[-max_k_num].dt if len(c.bars_raw) > max_k_num else c.bars_raw[0].dt
        df = pd.DataFrame(c.bars_raw)
        df["DIFF"], df["DEA"], df["MACD"] = MACD(df["close"], fastperiod=12, slowperiod=26, signalperiod=9)

        df = df[df["dt"] >= sdt].copy()
        kline = KlineChart(n_rows=3, row_heights=(0.5, 0.3, 0.2), title="", width="100%", height=600)
        kline.add_kline(df, name="")

        if len(c.bi_list) > 0:
            bi = pd.DataFrame(
                [{"dt": x.fx_a.dt, "bi": x.fx_a.fx} for x in c.bi_list]
                + [{"dt": c.bi_list[-1].fx_b.dt, "bi": c.bi_list[-1].fx_b.fx}]
            )
            fx = pd.DataFrame([{"dt": x.dt, "fx": x.fx} for x in c.fx_list])
            fx = fx[fx["dt"] >= sdt]
            bi = bi[bi["dt"] >= sdt]
            kline.add_scatter_indicator(fx["dt"], fx["fx"], name="分型", row=1, line_width=1.2, visible=True)
            kline.add_scatter_indicator(bi["dt"], bi["bi"], name="笔", row=1, line_width=1.5)

        kline.add_sma(df, ma_seq=(5, 20, 60), row=1, visible=False, line_width=1)
        kline.add_vol(df, row=2, line_width=1)
        kline.add_macd(df, row=3, line_width=1)

        for pos in trader.positions:
            bs_df = pd.DataFrame([x for x in pos.operates if x["dt"] >= sdt])
            if not bs_df.empty:
                bs_df["dt"] = bs_df["dt"].apply(lambda x: freq_end_time(x, Freq(freq), market="默认"))
                bs_df["tag"] = bs_df["op"].apply(lambda x: "triangle-up" if x == Operate.LO else "triangle-down")
                bs_df["color"] = bs_df["op"].apply(lambda x: "red" if x == Operate.LO else "silver")
                kline.add_scatter_indicator(
                    bs_df["dt"],
                    bs_df["price"],
                    name=pos.name,
                    text=bs_df["op_desc"],
                    row=1,
                    mode="markers",
                    marker_size=15,
                    marker_symbol=bs_df["tag"],
                    marker_color=bs_df["color"],
                    hover_template="价格: %{y:.2f}<br>时间: %{x}<br>操作: %{text}<extra></extra>",
                )

        with tabs[i]:
            config = {
                "scrollZoom": True,
                "displayModeBar": True,
                "displaylogo": False,
                "modeBarButtonsToRemove": [
                    "toggleSpikelines",
                    "select2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "lasso2d",
                    "autoScale2d",
                    "hoverClosestCartesian",
                    "hoverCompareCartesian",
                ],
            }
            st.plotly_chart(kline.fig, use_container_width=True, config=config)
        i += 1

    with tabs[i]:
        st.caption("回放区间所有开平交易记录如下：")
        show_cols = ["策略标记", "交易方向", "盈亏比例", "开仓时间", "平仓时间", "持仓K线数", "事件序列"]
        st.dataframe(st.session_state.pos_pairs[show_cols], use_container_width=True, hide_index=True)

    i += 1
    with tabs[i]:
        with st.expander("查看最新信号", expanded=False):
            if len(trader.s):
                s = {k: v for k, v in trader.s.items() if len(k.split("_")) == 3}
                st.write(s)
            else:
                st.warning("当前没有信号配置信息")

        for file in files:
            with st.expander(f"持仓策略配置：{file.name}", expanded=False):
                st.json(json.loads(file.getvalue().decode("utf-8")), expanded=True)


def init_trader(files, symbol, bar_sdt, sdt, edt):
    """初始化回放参数

    :param files: 策略文件
    :param symbol: 交易标的
    :param bar_sdt: 行情开始日期
    :param sdt: 回放开始日期
    :param edt: 回放结束日期
    """

    print("init_trader_symbol:", symbol)
    assert pd.to_datetime(bar_sdt) < pd.to_datetime(sdt) < pd.to_datetime(edt), "回放起止日期设置错误"

    json_strategies = {file.name: json.loads(file.getvalue().decode("utf-8")) for file in files}
    tactic: CzscStrategyBase = JsonStreamStrategy(
        symbol=symbol, signals_module_name=os.environ["signals_module_name"], json_strategies=json_strategies
    )
    bars = get_raw_bars(symbol, tactic.base_freq, sdt=bar_sdt, edt=edt)
    bg, bars_right = tactic.init_bar_generator(bars, sdt=sdt)
    trader = CzscTrader(bg=bg, positions=deepcopy(tactic.positions), signals_config=deepcopy(tactic.signals_config))

    st.session_state.trader = deepcopy(trader)
    st.session_state.bars_right = deepcopy(bars_right)
    st.session_state.bars_index = 0
    st.session_state.run = False

    # 跑一遍回测，生成持仓记录，用于回放时给人工检查策略一个参考
    for bar in bars_right:
        trader.on_bar(bar)

    assert trader.positions, "当前策略没有持仓记录"
    pairs = [pd.DataFrame(pos.pairs) for pos in trader.positions if pos.pairs]
    st.session_state.pos_pairs = pd.concat(pairs, ignore_index=True)


def replay(files):
    """CTA策略回放"""
    st.subheader("策略回放", divider="rainbow")
    with st.expander("设置回放参数", expanded=False):
        with st.form(key="my_form_replay"):
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
            symbol = col1.selectbox("选择交易标的：", get_symbols("ALL"), index=0)
            bar_sdt = col2.date_input(label="行情开始日期：", value=pd.to_datetime("2017-01-01"))
            sdt = col3.date_input(label="回放开始日期：", value=pd.to_datetime("2019-01-01"))
            edt = col4.date_input(
                label="回放结束日期：", value=pd.to_datetime("2020-01-01"), max_value=pd.to_datetime("2022-01-01")
            )
            max_k_num = int(col5.number_input(label="最大绘制K线数", value=100, min_value=50, max_value=600))
            submitted = st.form_submit_button(label="设置回放参数")

    if submitted:
        init_trader(files, symbol, bar_sdt, sdt, edt)
        st.session_state.max_k_num = max_k_num
    else:
        if not hasattr(st.session_state, "trader"):
            st.warning("请先设置回放参数")
            st.stop()
        max_k_num = st.session_state.max_k_num

    if files and hasattr(st.session_state, "trader"):
        trader = deepcopy(st.session_state.trader)
        bars_right = deepcopy(st.session_state.bars_right)
        bars_num = len(bars_right)

        c1, c2, c3, c4, c5 = st.columns([5, 5, 5, 5, 25])

        bar_edt = bars_right[st.session_state.bars_index].dt
        target_bar_edt = c5.text_input(
            "行情定位到指定时间：", placeholder=bar_edt.strftime("%Y-%m-%d %H:%M"), key="bar_edt"
        )
        if target_bar_edt:
            target_bar_edt = pd.to_datetime(target_bar_edt)
            for i, bar in enumerate(bars_right):
                if bar.dt >= target_bar_edt:
                    st.session_state.bars_index = i
                    break

        if c1.button("行情播放"):
            st.session_state.run = True
        if c2.button("行情暂停"):
            st.session_state.run = False
        if c3.button("左移一根K线"):
            st.session_state.bars_index -= 1
        if c4.button("右移一根K线"):
            st.session_state.bars_index += 1

        # 约束 bars_index 的范围在 [0, bars_num]
        st.session_state.bars_index = max(0, st.session_state.bars_index)
        st.session_state.bars_index = min(st.session_state.bars_index, bars_num)

        suffix = (
            f"共{bars_num}根K线"
            if bars_num < 1000
            else f"共{bars_num}根K线，回放数据量较大（超过1000根K线），建议缩小回放时间范围"
        )
        st.caption(f"行情播放时间范围：{bars_right[0].dt} - {bars_right[-1].dt}; 当前K线：{bar_edt}；{suffix}")

        if st.session_state.run:
            idx = st.session_state.bars_index
            bars1 = bars_right[0:idx].copy()
            while bars1:
                bar_ = bars1.pop(0)
                trader.on_bar(bar_)

            bars2 = bars_right[idx:].copy()
            with st.empty():
                while bars2:
                    bar_ = bars2.pop(0)
                    trader.on_bar(bar_)
                    show_trader(trader, files, max_k_num=max_k_num)
                    st.session_state.bars_index += 1

        else:
            bars2 = bars_right[: st.session_state.bars_index + 1].copy()
            with st.empty():
                while bars2:
                    bar_ = bars2.pop(0)
                    trader.on_bar(bar_)
                show_trader(trader, files)
    else:
        st.warning("请上传策略文件, 文件格式为 json，配置回放参数")


# ======================================================================================================================
# 以下是回测的代码
# ======================================================================================================================


@st.cache_data()
def read_holds_and_pairs(files_traders, pos_name, fee=1):
    holds, pairs = [], []
    for file in tqdm(files_traders):
        try:
            trader = czsc.dill_load(file)
            pos = trader.get_position(pos_name)
            if not pos.holds:
                logger.info(f"{trader.symbol} {pos_name} 无持仓，跳过")
                continue

            hd = pd.DataFrame(pos.holds)
            hd["symbol"] = trader.symbol
            hd = czsc.subtract_fee(hd, fee=fee)
            holds.append(hd)

            pr = pd.DataFrame(pos.pairs)
            pairs.append(pr)
        except Exception as e:
            logger.warning(f"{file} {pos_name} 读取失败: {e}")

    dfh = pd.concat(holds, ignore_index=True)
    dfp = pd.concat(pairs, ignore_index=True)
    return dfh, dfp


@st.cache_data()
def get_daily_nv(df):
    """获取每日净值"""
    res = []
    for symbol, hd in tqdm(df.groupby("symbol")):
        hd = hd.sort_values("dt", ascending=True)
        try:
            daily = hd.groupby("date").agg({"edge_pre_fee": "sum", "edge_post_fee": "sum"}).reset_index()
            daily["symbol"] = symbol
            res.append(daily)
        except Exception as e:
            logger.exception(f"{symbol} 日收益获取失败: {e}")

    dfr = pd.concat(res, ignore_index=True)
    return dfr


def show_pos_detail(file_trader, pos_name):
    """显示持仓策略详情"""
    trader = czsc.dill_load(file_trader)
    pos = trader.get_position(pos_name)
    with st.expander(f"{pos_name} 持仓策略详情", expanded=False):
        _pos = pos.dump()
        _pos.pop("symbol")
        st.json(_pos)


def show_backtest_results(file_traders, pos_name, fee=1):
    dfh, dfp = read_holds_and_pairs(file_traders, pos_name, fee=fee)
    dfr = get_daily_nv(dfh)
    show_pos_detail(file_traders[0], pos_name)

    st.subheader("一、单笔收益评价", divider="rainbow")
    pp = czsc.PairsPerformance(dfp)
    # st.write(pp.basic_info)
    df1 = pp.agg_statistics("标的代码")
    _res = pp.basic_info
    _res["标的代码"] = "全部品种"
    df1 = pd.concat([pd.DataFrame([_res]), df1], ignore_index=True)
    _cols = [
        "标的代码",
        "开始时间",
        "结束时间",
        "交易标的数量",
        "总体交易次数",
        "平均持仓K线数",
        "平均单笔收益",
        "单笔收益标准差",
        "交易胜率",
        "单笔盈亏比",
        "累计盈亏比",
        "盈亏平衡点",
        "每根K线收益",
    ]
    df1 = df1[_cols].set_index("标的代码")
    color_cols = [
        "交易标的数量",
        "总体交易次数",
        "平均持仓K线数",
        "平均单笔收益",
        "单笔收益标准差",
        "交易胜率",
        "单笔盈亏比",
        "累计盈亏比",
        "盈亏平衡点",
        "每根K线收益",
    ]
    df1 = df1.style.format("{0:,.2f}", subset=color_cols, na_rep="-").background_gradient(
        cmap="RdYlGn_r", subset=color_cols
    )

    st.dataframe(df1, use_container_width=True)

    st.divider()
    st.subheader("二、品种等权收益曲线", divider="rainbow")
    dfd = dfr.groupby("date").agg({"edge_pre_fee": "mean", "edge_post_fee": "mean"}) / 10000
    dfd.index = pd.to_datetime(dfd.index)
    dfd.rename({"edge_pre_fee": "等权费前收益", "edge_post_fee": f"双边扣费{2*fee}BP"}, axis=1, inplace=True)
    czsc.show_daily_return(dfd, stat_hold_days=False)


def symbol_backtest(strategies, symbol, bar_sdt, sdt, edt, results_path):
    """回测单个标的

    :param strategies: 策略配置
    :param symbol: 标的代码
    :param bar_sdt: 行情开始日期
    :param sdt: 回测开始日期
    :param edt: 回测结束日期
    :param results_path: 回测结果保存路径
    """
    logger.info(f"backtesting {symbol}，result in {results_path}")
    file_trader = results_path / f"{symbol}.trader"
    if file_trader.exists():
        logger.info(f"{symbol} 已回测，跳过")
        return

    try:
        tactic = JsonStreamStrategy(json_strategies=strategies, symbol=symbol)
        bars = get_raw_bars(symbol, tactic.base_freq, sdt=bar_sdt, edt=edt)
        trader = tactic.backtest(bars, sdt=sdt)
        czsc.dill_dump(trader, file_trader)
    except Exception as e:
        logger.exception(f"{symbol} 回测失败：{e}")


@st.cache_data(ttl=60 * 60 * 24)
def backtest_all(strategies, results_path):
    """回测全部标的

    :param strategies: 策略配置
    :param results_path: 回测结果保存路径
    """
    bar_sdt = st.session_state.bar_sdt
    group = st.session_state.group
    sdt = st.session_state.sdt
    edt = st.session_state.edt
    max_workers = st.session_state.max_workers
    symbols = get_symbols(group)
    print("bar_sdt:", bar_sdt, "group:", group, "sdt:", sdt, "edt:", edt, "symbols:", symbols)
    if max_workers <= 1:
        for symbol in tqdm(symbols, desc="On Bar 回测进度"):
            symbol_backtest(strategies, symbol, bar_sdt, sdt, edt, results_path)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                executor.submit(symbol_backtest, strategies, symbol, bar_sdt, sdt, edt, results_path)
                for symbol in symbols
            ]
            for future in tqdm(as_completed(tasks), desc="On Bar 回测进度", total=len(tasks)):
                future.result()


def backtest(files):
    st.subheader("策略回测", divider="rainbow")
    with st.form(key="my_form_czsc"):
        col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
        bar_sdt = col2.date_input(label="行情开始日期", value=pd.to_datetime("2018-01-01"))
        group = col1.selectbox(
            label="回测品类", options=["A股主要指数", "A股场内基金", "中证500成分股", "期货主力", "外汇"], index=3
        )
        sdt = col3.date_input(
            label="回测开始日期", value=pd.to_datetime("2019-01-01"), min_value=pd.to_datetime(bar_sdt)
        )
        edt = col4.date_input(
            label="回测结束日期",
            value=pd.to_datetime("2021-01-01"),
            min_value=pd.to_datetime(sdt),
            max_value=pd.to_datetime("2023-01-01"),
        )
        max_workers = int(
            col5.number_input(label="指定进程数量", value=cpu_count() // 4, min_value=1, max_value=cpu_count() // 2)
        )
        fee = int(col6.number_input(label="单边费率（BP）", value=2, min_value=0, max_value=100))
        submit_button = st.form_submit_button(label="开始回测")

    if submit_button:
        st.session_state.bar_sdt = bar_sdt
        st.session_state.group = group
        st.session_state.sdt = sdt
        st.session_state.edt = edt
        st.session_state.max_workers = max_workers
        st.session_state.fee = fee

    if not hasattr(st.session_state, "files") or not hasattr(st.session_state, "bar_sdt"):
        st.warning("请先设置策略回测参数")
        st.stop()

    files = st.session_state.files
    bar_sdt = st.session_state.bar_sdt
    group = st.session_state.group
    sdt = st.session_state.sdt
    edt = st.session_state.edt
    max_workers = st.session_state.max_workers
    fee = st.session_state.fee

    strategies = {file.name: json.loads(file.getvalue().decode("utf-8")) for file in files}
    hash_code = hashlib.sha256(f"{str(strategies)}".encode("utf-8")).hexdigest()[:8].upper()
    results_path = Path(os.getenv("base_path")) / "CTA策略回测" / f"{sdt}_{edt}_{hash_code}" / group
    results_path.mkdir(exist_ok=True, parents=True)

    with st.sidebar.expander("策略详情", expanded=False):
        tactic = JsonStreamStrategy(json_strategies=strategies, symbol="symbol")
        st.caption(f"K线周期列表：{tactic.freqs}")
        st.caption("独立信号列表：")
        st.json(tactic.unique_signals)
        st.caption("信号函数配置：")
        st.json(tactic.signals_config)

    backtest_all(strategies, results_path)

    file_traders = glob.glob(fr"{results_path}/*.trader")
    #logger.info(f"{file_traders} lookup result")
    if not file_traders:
        st.warning("当前回测参数下，没有任何标的回测结果；请调整回测参数后重试")
        st.stop()

    all_pos_names = [x.name for x in czsc.dill_load(file_traders[0]).positions]
    tabs = st.tabs(["全部品种", "选择特定品种组合"])

    with tabs[0]:
        pos_name = st.selectbox("选择持仓", all_pos_names, index=0, key="pos_name")
        show_backtest_results(file_traders, pos_name, fee=fee)

    with tabs[1]:
        candidates = [Path(x).stem for x in file_traders]
        sel_symbols = []
        with st.form(key="my_form_czsc_2"):
            col1, col2 = st.columns([1, 3])
            pos_name_a = col1.selectbox("选择持仓", all_pos_names, index=0, key="pos_name_a")
            sel_symbols = col2.multiselect("选择品种", candidates, default=candidates[:3])
            submit_button = st.form_submit_button(label="分析特定品种组合")

        if not sel_symbols:
            st.warning("请先选择品种组合")
            st.stop()

        sel_files = [x for x in file_traders if Path(x).stem in sel_symbols]
        show_backtest_results(sel_files, pos_name_a, fee=fee)


# ======================================================================================================================
# 以下是主函数
# ======================================================================================================================


def main():
    logger.add("cta.log", format="{time} {level} {message}", level="INFO", rotation="10 MB")
    with st.sidebar:
        # https://icons.getbootstrap.com/
        selected = option_menu(
            "CTA研究",
            ["信号函数", "策略回放", "策略回测"],
            icons=["bi-film", "bi-mortarboard"],
            menu_icon="bi-filter",
            default_index=0,
        )

    if selected == "信号函数":
        show_signal_functions()
        st.stop()

    with st.sidebar:
        st.divider()
        files = st.file_uploader(label="上传策略文件：", type="json", accept_multiple_files=True)

    if not files and not hasattr(st.session_state, "files"):
        st.warning("请先在侧边栏上传策略文件")
        st.stop()

    st.session_state.files = files or st.session_state.files

    if selected == "策略回放":
        replay(files)

    if selected == "策略回测":
        backtest(files)


if __name__ == "__main__":
    main()
