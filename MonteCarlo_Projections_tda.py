import requests
import pandas as pd
import numpy as np
from scipy.stats import qmc, norm, cauchy, median_abs_deviation
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from Code.API_CREDENTIALS import TDA_API_CREDENTIALS

"""
These classes are used to numerically calculate the probability distribution function (PDF) of the
closing price for n days ahead for a selected asset.

Normal_MC uses a Gaussian (normal) or Cauchy distribution to sample from in order to generate the next step.

GMM_MC uses a :
Gaussian Mixture Model (GMM) to sample from in order to generate the next step. HOWEVER, when sampling from a GMM,
the model first randomly selects one of its Gaussian distributions according to their weights, and then it generates
a sample from the selected Gaussian distribution. If this process is repeated many times, the samples will
approximate the overall distribution of the GMM, which is a combination of all its Gaussian distributions.

Data is provided by TD Ameritrade's API... soon to be replaced by Charles Schwab's developer API.

Last Updated: 08-01-2023 1:30AM CST
"""


class Normal_MC:
    def __init__(self, asset, frequencyType, length, daysBack):
        """
        Constructor for the Normal_MC class.

        Parameters:
        asset (str): Ticker symbol of the asset ie "$SPX.X".
        frequencyType (str): Type of frequency for the data ie "Daily" intervals.
        length (int): Number of periods to consider for the Monte Carlo simulation.
        daysback (int): Number of days back to consider for the Monte Carlo simulation.
        """

        self.asset = asset
        self.distribution = None
        self.mc_result = None
        self.sampling_method = None
        self.statistics_method = None

        self.LOC = None
        self.SCALE = None

        self.SCALE_UPPER_1 = None
        self.SCALE_LOWER_1 = None
        self.SCALE_UPPER_2 = None
        self.SCALE_LOWER_2 = None

        self.STATISTICS_METHODS = {
            'mean_stddev': self.calculate_mean_stddev,
            'median_mad': self.calculate_median_mad,
            'median_percentiles': self.calculate_median_percentiles,
            'mean_mad': self.calculate_mean_mad,
        }
        self.SAMPLERS = {
            'Pseudorandom': None,
            'Halton': qmc.Halton,
            'Sobol': qmc.Sobol,
            'Lhc': qmc.LatinHypercube,
        }
        self.DISTRIBUTIONS = {
            'Normal': norm,
            'Gaussian': norm,
            'Cauchy': cauchy,
        }
        apiurl = r"https://api.tdameritrade.com/v1/marketdata/{}/pricehistory".format(asset)
        apiparam = {'apikey': TDA_API_CREDENTIALS,
                    'period': '20',
                    'periodType': 'year',
                    'frequency': '1',
                    'frequencyType': '{}'.format(frequencyType),
                    'needExtendedHoursData': 'true'}

        try:
            geturl = requests.get(url=apiurl, params=apiparam)
            geturl.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("HTTP Error:", errh)
            return
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:", errc)
            return
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:", errt)
            return
        except requests.exceptions.RequestException as err:
            print("Something went wrong with the request:", err)
            return

        try:
            data = geturl.json()
            candle_data = pd.DataFrame(data['candles'])

            if daysBack == 0:
                close_data = candle_data['close'].iloc[-1 - length:]
            else:
                close_data = candle_data['close'].iloc[-1 - length - daysBack: -daysBack]

            logReturns = np.log(close_data) - np.log(close_data.shift(1))

            self.dfLogReturns = pd.Series(logReturns).iloc[1:]
            self.closePrice = close_data.iloc[-1]
        except KeyError as ke:
            print("Data fetching failed due to missing key in response:", ke)
        except Exception as e:
            print("Unexpected error occurred while processing data:", e)

    def simulate(self, trials, steps, sampling, distribution, centered):
        """
        Method that performs the Monte Carlo simulation.

        Parameters:
        trials (int): Number of trials to run in the Monte Carlo simulation.
        steps (int): Number of steps in each trial.
        sampling (str): Sampling method to use (one of: 'Pseudorandom', 'Halton', 'Sobol', 'Lhc').
        centered (bool): Whether the simulation should use a mean of zero or the calculated mean from the log returns.
        """
        if sampling not in self.SAMPLERS:
            raise ValueError("Not valid sampling method! Available methods: Pseudorandom, Halton, Sobol, Lhc")
        self.sampling_method = self.SAMPLERS[sampling]

        if distribution not in self.DISTRIBUTIONS:
            raise ValueError("Not valid distribution! Available distributions: Normal, Cauchy")
        dist = self.DISTRIBUTIONS[distribution]

        loc = 0 if centered else np.mean(self.dfLogReturns)
        stdDev = np.std(self.dfLogReturns)
        sim_series = np.full(shape=(steps, trials), fill_value=0.0)

        if sampling == "Pseudorandom":
            for i in range(trials):
                norm_sample = dist.rvs(loc=loc, scale=stdDev, size=(steps, 1))
                sim_series[:, i] = np.cumprod(norm_sample + 1)
        else:
            sampler = self.sampling_method(d=steps, scramble=True)
            for i in range(trials):
                sample = sampler.random(n=1)
                dist_sample = dist.ppf(sample, loc=loc, scale=stdDev)
                sim_series[:, i] = np.cumprod(dist_sample + 1)

        self.mc_result = sim_series * self.closePrice
        self.sampling_method = sampling
        self.distribution = distribution

    def summary(self, statistics_method):
        """
        Method that prints a summary of the Monte Carlo simulation.
        Shows the sampling method, asset, close price, and the method of statistical analysis.

        Parameters:
        statistics_method (str): statistical method to use (one of: 'mean_stddev', 'mean_mad', 'median_mad', 'median_percentiles')
        """
        self.statistics_method = statistics_method

        print("##################################################")
        print(f"{self.distribution} Distribution\n---------------")
        print(f"Asset: {self.asset}")
        print(f"Sampling Method: {self.sampling_method}")
        print(f"Statistics Method: {statistics_method}")
        print("-----")
        print("Close price: ", self.closePrice)
        print("-----")
        self.data = np.array(self.mc_result[-1])

        if self.statistics_method not in self.STATISTICS_METHODS:
            raise ValueError(
                "Not valid statistics method! Available methods: mean_stddev, mean_mad, median_mad, median_percentiles")

        self.STATISTICS_METHODS[self.statistics_method]()

    def calculate_mean_stddev(self):
        """
        Method that calculates the mean and ±2 standard deviations of the simulated prices.
        """
        if self.mc_result is None:
            raise ValueError("Please run the simulation before calculating statistics.")

        self.LOC = np.mean(self.data)
        self.SCALE = np.std(self.data)

        self.SCALE_UPPER_1 = self.LOC + np.std(self.data)
        self.SCALE_LOWER_1 = self.LOC - np.std(self.data)

        self.SCALE_UPPER_2 = self.LOC + ((np.std(self.data)) * 2)
        self.SCALE_LOWER_2 = self.LOC - ((np.std(self.data)) * 2)

        print(f'Mean = {self.LOC:.2f}')
        print(f'Standard Deviation = {self.SCALE:.2f}')
        print("---------------")

        print(f'+1 STD = {self.SCALE_UPPER_1:.2f}')
        print(f'-1 STD = {self.SCALE_LOWER_1:.2f}')
        print("-----")
        print(f'+2 STD = {self.SCALE_UPPER_2:.2f}')
        print(f'-2 STD = {self.SCALE_LOWER_2:.2f}')
        print("##################################################")

    def calculate_mean_mad(self):
        """
        Method that calculates the median and ±2 mean absolute deviation of the simulated prices.
        """
        if self.mc_result is None:
            raise ValueError("Please run the simulation before calculating statistics.")

        self.LOC = np.mean(self.data)
        self.SCALE = np.mean(np.abs(self.data - self.LOC))

        self.SCALE_UPPER_1 = self.LOC + np.mean(np.abs(self.data - self.LOC))
        self.SCALE_LOWER_1 = self.LOC - np.mean(np.abs(self.data - self.LOC))

        self.SCALE_UPPER_2 = self.LOC + (np.mean(np.abs(self.data - self.LOC)) * 2)
        self.SCALE_LOWER_2 = self.LOC - (np.mean(np.abs(self.data - self.LOC)) * 2)

        print(f'Mean = {self.LOC:.2f}')
        print(f'Mean Absolute Deviation (MAD) = {self.SCALE:.2f}')
        print("---------------")

        print(f'+1 MAD = {self.SCALE_UPPER_1:.2f}')
        print(f'-1 MAD = {self.SCALE_LOWER_1:.2f}')
        print("-----")
        print(f'+2 MAD = {self.SCALE_UPPER_2:.2f}')
        print(f'-2 MAD = {self.SCALE_LOWER_2:.2f}')
        print("##################################################")

    def calculate_median_mad(self):
        """
        Method that calculates the median and ±2 median absolute deviation of the simulated prices.
        """
        if self.mc_result is None:
            raise ValueError("Please run the simulation before calculating statistics.")

        self.LOC = np.median(self.data)
        self.SCALE = median_abs_deviation(self.data)

        self.SCALE_UPPER_1 = self.LOC + median_abs_deviation(self.data)
        self.SCALE_LOWER_1 = self.LOC - median_abs_deviation(self.data)

        self.SCALE_UPPER_2 = self.LOC + ((median_abs_deviation(self.data)) * 2)
        self.SCALE_LOWER_2 = self.LOC - ((median_abs_deviation(self.data)) * 2)

        print(f'Median = {self.LOC:.2f}')
        print(f'Median Absolute Deviation (MEAN) = {self.SCALE:.2f}')
        print("---------------")

        print(f'+1 MAD = {self.SCALE_UPPER_1:.2f}')
        print(f'-1 MAD = {self.SCALE_LOWER_1:.2f}')
        print("-----")
        print(f'+2 MAD = {self.SCALE_UPPER_2:.2f}')
        print(f'-2 MAD = {self.SCALE_LOWER_2:.2f}')
        print("##################################################")

    def calculate_median_percentiles(self):
        """
        Method that calculates the 2.5th and 97.5th percentiles of the simulated prices.
        """
        if self.mc_result is None:
            raise ValueError("Please run the simulation before calculating statistics.")

        self.LOC = np.median(self.data)

        self.SCALE_UPPER_1 = np.percentile(self.data, 84)
        self.SCALE_LOWER_1 = np.percentile(self.data, 16)

        self.SCALE_UPPER_2 = np.percentile(self.data, 90)
        self.SCALE_LOWER_2 = np.percentile(self.data, 10)

        print(f'Median = {self.LOC:.2f}')
        print(f'1 Standard Deviation Percentile Equivalent= {(self.LOC - self.SCALE_LOWER_1):.2f}')
        print("---------------")

        print(f'84% Percentile = {self.SCALE_UPPER_1:.2f}')
        print(f'16% Percentile = {self.SCALE_LOWER_1:.2f}')
        print("-----")
        print(f'90% Percentile = {self.SCALE_UPPER_2:.2f}')
        print(f'10% Percentile = {self.SCALE_LOWER_2:.2f}')
        print("##################################################")

    def plot(self):
        """
        Method that plots the Monte Carlo simulation results.
        It first plots the simulated prices over the number of
        steps, and then plots a histogram of the final prices from all trials.
        """
        plt.plot(self.mc_result)
        plt.ylim([np.percentile(self.data, 2.5), np.percentile(self.data, 97.5)])
        plt.ylabel("Ticker's Close Price ($)")
        plt.xlabel('Steps')
        plt.title(f"{self.mc_result.shape[0]}-Step {self.asset} Closing Price Monte Carlo Simulations\nSampled From a "
                  f"{self.distribution} Distribution")
        plt.show()

        fdBins = np.histogram_bin_edges(self.data, bins='auto', range=None, weights=None)
        plt.hist(self.data, bins=fdBins, density=True)
        plt.xlim([np.percentile(self.data, 2.5), np.percentile(self.data, 97.5)])
        plt.axvline(x=self.LOC, color='white')
        plt.axvline(x=self.SCALE_UPPER_1, color='yellow')
        plt.axvline(x=self.SCALE_LOWER_1, color='yellow')
        plt.axvline(x=self.SCALE_UPPER_2, color='green')
        plt.axvline(x=self.SCALE_LOWER_2, color='green')
        plt.title(
            f"{self.asset} Price Projections Over {self.mc_result.shape[0]} Days\n(White: LOC, Yellow: ±1 SCALE, "
            f"Green: ±2 SCALE)\nSampled From a {self.distribution} Distribution")
        plt.xlabel(f"{self.asset}'s 5-Day Closing Price")
        plt.ylabel("Density of Simulations")
        plt.show()


class GMM_MC:
    def __init__(self, asset, frequencyType, length, daysBack):
        """
        Constructor for the GMM_MC class.

        Parameters:
        asset (str): Ticker symbol of the asset ie "$SPX.X".
        frequencyType (str): Type of frequency for the data ie "Daily" intervals.
        length (int): Number of periods to consider for the Monte Carlo simulation.
        daysback (int): Number of days back to consider for the Monte Carlo simulation.
        """

        self.gmm = None
        self.mc_result = None
        self.data = None
        self.asset = asset

        self.LOC = None
        self.SCALE = None

        self.SCALE_UPPER_1 = None
        self.SCALE_LOWER_1 = None
        self.SCALE_UPPER_2 = None
        self.SCALE_LOWER_2 = None
        
        apiurl = r"https://api.tdameritrade.com/v1/marketdata/{}/pricehistory".format(asset)
        apiparam = {'apikey': TDA_API_CREDENTIALS,
                    'period': '20',
                    'periodType': 'year',
                    'frequency': '1',
                    'frequencyType': '{}'.format(frequencyType),
                    'needExtendedHoursData': 'true'}

        try:
            geturl = requests.get(url=apiurl, params=apiparam)
            geturl.raise_for_status()
        except requests.exceptions.HTTPError as errh:
            print("HTTP Error:", errh)
            return
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:", errc)
            return
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:", errt)
            return
        except requests.exceptions.RequestException as err:
            print("Something went wrong with the request:", err)
            return

        try:
            data = geturl.json()
            candle_data = pd.DataFrame(data['candles'])

            if daysBack == 0:
                close_data = candle_data['close'].iloc[-1 - length:]
            else:
                close_data = candle_data['close'].iloc[-1 - length - daysBack:-daysBack]

            logReturns = np.log(close_data) - np.log(close_data.shift(1))

            self.dfLogReturns = pd.Series(logReturns).iloc[1:]
            self.closePrice = close_data.iloc[-1]
        except KeyError as ke:
            print("Data fetching failed due to missing key in response:", ke)
        except Exception as e:
            print("Unexpected error occurred while processing data:", e)

    def fit(self, n_components, plot):
        """
        Method that fits a Gaussian Mixture Model to the log returns and plots the fit.

        Parameters:
        n_components (int) : Number of mixture components.
        plot (bool) : optional, plots a histogram of the fitted distribution
        """
        self.gmm = GaussianMixture(n_components=n_components, max_iter=1000, n_init=100, tol=1e-5)
        self.gmm.fit(X=np.expand_dims(self.dfLogReturns, 1))

        if plot:
            x = np.linspace(min(self.dfLogReturns) * 1.5, max(self.dfLogReturns) * 1.5, 1000000).reshape(-1, 1)
            y = np.exp(self.gmm.score_samples(x))

            fdBins = np.histogram_bin_edges(self.dfLogReturns, bins='auto', range=None, weights=None)

            fig, ax = plt.subplots()
            ax.plot(x, y, color='green', lw=3, label="Gaussian Mixture Model")
            ax.hist(self.dfLogReturns, fdBins, density=True)
            plt.legend()
            plt.show()

    def simulate(self, trials, steps):
        """
        Method that performs the Monte Carlo simulation.

        Parameters:
        trials (int): Number of trials to run in the Monte Carlo simulation.
        steps (int): Number of steps in each trial.
        """
        sim_series = np.full(shape=(steps, trials), fill_value=0.0)

        for i in range(trials):
            gmm_samples = self.gmm.sample(steps)[0]
            sim_series[:, i] = np.cumprod(gmm_samples + 1)

        self.mc_result = sim_series * self.closePrice

    def summary(self):
        """
        Method that prints a summary of the Monte Carlo simulation.

        Shows the sampling method, asset, close price, mean and standard deviation of the simulated prices.
        """
        print("##################################################")
        print(f"Gaussian Mixture Model\n---------------")
        print(f"Asset: {self.asset}")
        print(f"Sampling Method: pseudorandom")
        print(f"Statistics Method: mean_stddev")
        print("-----")
        print("Close price: ", self.closePrice)
        print("-----")

        self.data = np.array(self.mc_result[-1])
        self.LOC = np.mean(self.data)
        self.SCALE = np.std(self.data)

        self.SCALE_UPPER_1 = self.LOC + np.std(self.data)
        self.SCALE_LOWER_1 = self.LOC - np.std(self.data)

        self.SCALE_UPPER_2 = self.LOC + ((np.std(self.data)) * 2)
        self.SCALE_LOWER_2 = self.LOC - ((np.std(self.data)) * 2)

        print(f'Mean = {self.LOC:.2f}')
        print(f'Standard Deviation = {self.SCALE:.2f}')
        print("---------------")

        print(f'+1 STD = {self.SCALE_UPPER_1:.2f}')
        print(f'-1 STD = {self.SCALE_LOWER_1:.2f}')
        print("-----")
        print(f'+2 STD = {self.SCALE_UPPER_2:.2f}')
        print(f'-2 STD = {self.SCALE_LOWER_2:.2f}')
        print("##################################################")

    def plot(self):
        """
        Method that plots the Monte Carlo simulation results.

        It first plots the simulated prices over the number of
        steps, and then plots a histogram of the final prices from all trials.
        """
        plt.plot(self.mc_result)
        plt.ylabel("Ticker's Close Price ($)")
        plt.xlabel('Steps')
        plt.title(f"{self.mc_result.shape[0]}-Step {self.asset} Closing Price Monte Carlo Simulations")
        plt.show()

        fdBins = np.histogram_bin_edges(self.data, bins='auto', range=None, weights=None)
        plt.hist(x=self.data, bins=fdBins, density=True)
        plt.axvline(x=self.LOC, color='white')
        plt.axvline(x=(self.LOC + self.SCALE), color='yellow')
        plt.axvline(x=(self.LOC - self.SCALE), color='yellow')
        plt.axvline(x=(self.LOC + 2 * self.SCALE), color='green')
        plt.axvline(x=(self.LOC - 2 * self.SCALE), color='green')
        plt.title(
            f"{self.asset} Price Projections Over {self.mc_result.shape[0]} Days\n(White: Mean, Yellow: ±1SD, Green: ±2SD)")
        plt.xlabel(f"{self.asset}'s 5-Day Closing Price")
        plt.ylabel("Density of Simulations")
        plt.show()
