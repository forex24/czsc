from typing import List
from czsc import CzscStrategyBase, Position, Event


def create_intraday_long(symbol, freq1="1分钟", freq2="5分钟", freq3="30分钟", **kwargs):
    """创建日内多头策略

    三个级别进行交易，小级别作为触发级别，本级别作为交易级别，大级别作为辅助过滤。

    ## 小级别找入场点


    ## 本级别开仓条件


    ## 大级别过滤条件


    :param symbol: 交易标的
    :param freq1: 小级别
    :param freq2: 本级别
    :param freq3: 大级别
    """
    pos_name = f"{freq1}#{freq2}#{freq3}#日内多头模板"
    opens = [
        {
            "operate": "开多",
            "signals_not": [],
            "signals_all": [],
            "factors": [
                {
                    "name": f"{freq2}底背驰",
                    "signals_all": [
                        f"{freq3}_D1W20_分类V240428_加速上涨_任意_任意_0",
                    ],
                    "signals_not": [
                        # 日内交易，不留过夜仓，下午2点半、晚上10点半之后不开仓
                        f"{freq1}_T1430#1500_时间区间_是_任意_任意_0",
                        f"{freq1}_T2230#2300_时间区间_是_任意_任意_0",
                    ],
                    "signals_any": [
                        f"{freq2}_D1N30柱子背驰_BS辅助V240307_底背驰_第1次_任意_0",
                        f"{freq2}_D1N30柱子背驰_BS辅助V240307_底背驰_第2次_任意_0",
                    ],
                }
            ],
        },
    ]
    exits = [
        {
            "operate": "平多",
            "signals_all": [],
            "signals_not": [],
            "factors": [
                {
                    "name": "临收盘",
                    "signals_all": [
                        f"{freq1}_T1455#1500_时间区间_任意_任意_任意_0",
                    ],
                    "signals_any": [
                        # 日内交易，不留过夜仓，下午2点55分、晚上10点55分之后平掉所有仓位
                        f"{freq1}_T1455#1500_时间区间_是_任意_任意_0",
                        f"{freq1}_T2255#2300_时间区间_是_任意_任意_0",
                    ],
                },
                {
                    "name": "5跳止损单",
                    "signals_all": [
                        f"{pos_name}_{freq1}T5N5_止损V240428_多头止损_任意_任意_0",
                    ],
                },
                {
                    "name": "放量止盈",
                    "signals_all": [
                        f"{pos_name}_{freq1}T1N5_止盈V240428_多头止盈_任意_任意_0",
                    ],
                },
                {
                    "name": "保本单",
                    "signals_all": [
                        f"{pos_name}_{freq1}H50T20N5_保本V240428_多头保本_任意_任意_0",
                    ],
                },
            ],
        }
    ]

    pos = Position(
        name=pos_name,
        symbol=symbol,
        opens=[Event.load(x) for x in opens],
        exits=[Event.load(x) for x in exits],
        # interval 两次开仓之间的最小间隔时间
        interval=kwargs.get("interval", 3600),
        # timeout 最后一次开仓信号出现后的最大持仓时间
        timeout=kwargs.get("timeout", 60),
        # stop_loss 止损点
        stop_loss=kwargs.get("stop_loss", 300),
        T0=True,
    )
    return pos


def create_intraday_short(symbol, freq1="1分钟", freq2="5分钟", freq3="30分钟", **kwargs):
    """创建日内空头策略

    三个级别进行交易，小级别作为触发级别，本级别作为交易级别，大级别作为辅助过滤。

    ## 小级别找入场点


    ## 本级别开仓条件


    ## 大级别过滤条件


    :param symbol: 交易标的
    :param freq1: 小级别
    :param freq2: 本级别
    :param freq3: 大级别
    """
    pos_name = f"{freq1}#{freq2}#{freq3}#日内空头模板"
    opens = [
        {
            "operate": "开空",
            "signals_not": [],
            "signals_all": [],
            "factors": [
                {
                    "name": f"{freq2}顶背驰",
                    "signals_all": [
                        f"{freq3}_D1W20_分类V240428_加速下跌_任意_任意_0",
                    ],
                    "signals_not": [
                        # 日内交易，不留过夜仓，下午2点半、晚上10点半之后不开仓
                        f"{freq1}_T1430#1500_时间区间_是_任意_任意_0",
                        f"{freq1}_T2230#2300_时间区间_是_任意_任意_0",
                    ],
                    "signals_any": [
                        f"{freq2}_D1N30柱子背驰_BS辅助V240307_顶背驰_第1次_任意_0",
                        f"{freq2}_D1N30柱子背驰_BS辅助V240307_顶背驰_第2次_任意_0",
                    ],
                }
            ],
        },
    ]
    exits = [
        {
            "operate": "平空",
            "signals_all": [],
            "signals_not": [],
            "factors": [
                {
                    "name": "临收盘",
                    "signals_all": [
                        f"{freq1}_T1455#1500_时间区间_任意_任意_任意_0",
                    ],
                    "signals_any": [
                        # 日内交易，不留过夜仓，下午2点55分、晚上10点55分之后平掉所有仓位
                        f"{freq1}_T1455#1500_时间区间_是_任意_任意_0",
                        f"{freq1}_T2255#2300_时间区间_是_任意_任意_0",
                    ],
                },
                {
                    "name": "5跳止损单",
                    "signals_all": [
                        f"{pos_name}_{freq1}T5N5_止损V240428_空头止损_任意_任意_0",
                    ],
                },
                {
                    "name": "放量止盈",
                    "signals_all": [
                        f"{pos_name}_{freq1}T1N5_止盈V240428_空头止盈_任意_任意_0",
                    ],
                },
                {
                    "name": "保本单",
                    "signals_all": [
                        f"{pos_name}_{freq1}H50T20N5_保本V240428_空头保本_任意_任意_0",
                    ],
                },
            ],
        }
    ]

    pos = Position(
        name=pos_name,
        symbol=symbol,
        opens=[Event.load(x) for x in opens],
        exits=[Event.load(x) for x in exits],
        # interval 两次开仓之间的最小间隔时间
        interval=kwargs.get("interval", 3600),
        # timeout 最后一次开仓信号出现后的最大持仓时间
        timeout=kwargs.get("timeout", 60),
        # stop_loss 止损点
        stop_loss=kwargs.get("stop_loss", 300),
        T0=True,
    )
    return pos


class Strategy(CzscStrategyBase):
    @property
    def positions(self) -> List[Position]:
        _pos = [
            create_intraday_long(symbol=self.symbol, freq1="1分钟", freq2="5分钟", freq3="30分钟"),
            create_intraday_short(symbol=self.symbol, freq1="1分钟", freq2="5分钟", freq3="30分钟"),
        ]
        return _pos


if __name__ == "__main__":
    from czsc.connectors.research import get_raw_bars

    tactic = Strategy(symbol="EURUSD")
    tactic.save_positions(r"/Users/gxj/日内策略模板")

    bars = get_raw_bars("EURUSD", freq="1分钟", sdt="2015-01-01", edt="2022-07-01")
    tactic.replay(bars, res_path=r"/Users/gxj/日内策略模板/回放结果", refresh=True, sdt="2021-01-01")
