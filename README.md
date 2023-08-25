
# MonteCarlo Projections

This project provides tools for numerically calculating the probability distribution function (PDF) of the closing price for `n` days ahead for a selected asset. The methods are based on Monte Carlo simulations.

## Warning:

The _tda code will not run without TD Ameritrade's API key. 

## Features:

1. **Normal Monte Carlo (`Normal_MC` class):** Uses a Gaussian (normal) or Cauchy distribution to sample from in order to generate the next step in the simulation.
2. **Gaussian Mixture Model Monte Carlo (`GMM_MC` class):** Uses a Gaussian Mixture Model (GMM) to sample from. When sampling from a GMM, the model first randomly selects one of its Gaussian distributions according to their weights, and then it generates a sample from the selected Gaussian distribution.

## Dependencies:

- requests
- pandas
- numpy
- scipy
- sklearn
- matplotlib
- yfinance

## Usage:

### Using the TDA API:

To use the Normal Monte Carlo method:
```python
import MonteCarlo_Projections as mcp_tda

mc_norm = mcp_tda.Normal_MC(asset='$SPX.X', frequencyType='daily', length=30, daysBack=0) # pulls data from API
mc_norm.simulate(trials=200000, steps=5, sampling="Sobol", distribution="Normal", centered=True) # runs the MC simulation
mc_norm.summary("mean_stddev") # prints preferred statistics
mc_norm.plot() # plots the MC simulation and the histogram of the final values of the MC
```

To use the Gaussian Mixture Model Monte Carlo method:
```python
import MonteCarlo_Projections as mcp_tda

mc_GMM = mcp_tda.GMM_MC(asset='$SPX.X', frequencyType='daily', length=30, daysback=0) # pulls data from API
mc_GMM.fit(n_components=5, plot=False) # fit a GMM to the log returns
mc_GMM.simulate(trials=30000, steps=5) # runs the MC simulation
mc_GMM.summary() # prints statistics
mc_GMM.plot() # plots the MC simulation and the histogram of the final values of the MC
```

### Using Yfinance:

To use the Normal Monte Carlo method:
```python
import MonteCarlo_Projections as mcp_yf

mc_norm = mcp_yf.Normal_MC('^SPX', '1d', length=30, daysBack=0) # pulls data from yfinance
mc_norm.simulate(trials=200000, steps=5, sampling="Sobol", distribution="Normal", centered=True) # runs the MC simulation
mc_norm.summary("mean_stddev") # prints preferred statistics
mc_norm.plot() # plots the MC simulation and the histogram of the final values of the MC
```

To use the Gaussian Mixture Model Monte Carlo method:
```python
import MonteCarlo_Projections as mcp_yf

mc_GMM = mcp_yf.GMM_MC('^SPX', '1d', length=30, daysBack=0) # pulls data from yfinance
mc_GMM.fit(n_components=5, plot=False) # fit a GMM to the log returns
mc_GMM.simulate(trials=30000, steps=5) # runs the MC simulation
mc_GMM.summary() # prints statistics
mc_GMM.plot() # plots the MC simulation and the histogram of the final values of the MC
```

## Data Source:

API data is provided by TD Ameritrade's API. This will soon be replaced by Charles Schwab's developer API.

## Author:

Ali Shirazi-Nejad
