import MonteCarlo_Projections_tda as mcp_tda
import MonteCarlo_Projections_yfinance as mcp_yf


def run_simulation_MC(symbol, length, steps, centered):
    mc_norm = mcp_tda.Normal_MC(symbol, 'daily', length=length, daysBack=0)
    # mc_norm = mcp_yf.Normal_MC(symbol, '1d', length=length, daysBack=0)  # for mcp_yf
    mc_norm.simulate(trials=200000, steps=steps, sampling="Sobol", distribution="Normal", centered=centered)
    mc_norm.summary("mean_stddev")
    # mc_norm.plot()  # unnecessary


def run_simulation_Analytical(symbol, length, steps, centered):
    mc_norm = mcp_tda.Closed_form_forecast(symbol, 'daily', length=length, daysBack=0)
    # mc_norm = mcp_yf.Closed_form_forecast(symbol, '1d', length=length, daysBack=0)  # for mcp_yf
    mc_norm.fit(steps=steps, centered=centered)
    mc_norm.summary()


if __name__ == "__main__":
    symbols = ['$SPX.X', '$NDX.X', '$VIX.X']
    # symbols = ['^SPX', '^NDX', '^VIX']  # for mcp_yf
    configurations = [(252, True), (30, True), (252, False), (30, False)]
    steps = 4

    for symbol in symbols:
        for length, centered in configurations:
            # run_simulation_MC(symbol, length, steps, centered)
            run_simulation_Analytical(symbol, length, steps, centered)
