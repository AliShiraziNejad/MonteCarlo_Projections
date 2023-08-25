import MonteCarlo_Projections_tda as mcp_tda
import MonteCarlo_Projections_yfinance as mcp_yf

if __name__ == "__main__":
    """
    TDA API
    """
    mc_norm = mcp_tda.Normal_MC('$SPX.X', 'daily', length=30, daysBack=0)
    mc_norm.simulate(trials=200000, steps=5, sampling="Sobol", distribution="Normal", centered=False)
    mc_norm.summary("mean_stddev")
    # mc_norm.plot()

    """
    mc_GMM = mcp_tda.GMM_MC('$SPX.X', 'daily', length=30, daysBack=0)
    mc_GMM.fit(n_components=5, plot=False)
    mc_GMM.simulate(trials=200000, steps=5)
    mc_GMM.summary()
    #mc_GMM.plot()

    ""
    Yfinance library
    ""
    mc_norm = mcp_yf.Normal_MC('^SPX', '1d', length=30, daysBack=0)
    mc_norm.simulate(trials=200000, steps=5, sampling="Sobol", distribution="Normal", centered=False)
    mc_norm.summary("mean_stddev")
    #mc_norm.plot()

    mc_GMM = mcp_yf.GMM_MC('^SPX', '1d', length=30, daysBack=0)
    mc_GMM.fit(n_components=5, plot=False)
    mc_GMM.simulate(trials=200000, steps=5)
    mc_GMM.summary()
    #mc_GMM.plot()
    """
