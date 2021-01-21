#from envs.sp500_trading import Sp500TradingEnv
#from envs.fund_selection_v0 import FundSelectionEnv  # MultiDiscrete Env. n to n reward calculation
#from envs.fund_selection_v1 import FundSelectionEnvCov  # FundSelectionEnv_Cov

import header.index_forecasting.RUNHEADER as RUNHEADER
if RUNHEADER.market_timing:
    if RUNHEADER.release:
        from libs.envs.market_timing_v0 import MarketTimingEnv
    else:
        from envs.market_timing_v0 import MarketTimingEnv
else:
    if RUNHEADER.release:
        from libs.envs.index_forecasting_v0 import IndexForecastingEnv  # IndexForecastingEnv
    else:
        from envs.index_forecasting_v0 import IndexForecastingEnv  # IndexForecastingEnv
