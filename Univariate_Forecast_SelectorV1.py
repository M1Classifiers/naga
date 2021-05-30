import numpy as np
import pandas as pd
from statistics import mode

from warnings import simplefilter

# Data split and CV modules
from sktime.forecasting.model_selection import (
    temporal_train_test_split,
    ForecastingGridSearchCV,
    SlidingWindowSplitter,
    ExpandingWindowSplitter,
)
from sklearn.model_selection import GridSearchCV

# Create or combine Forecasters
from sktime.forecasting.compose import (
    make_reduction,
    EnsembleForecaster,
    MultiplexForecaster,
    TransformedTargetForecaster,
)

# Performance metrics
from sktime.performance_metrics.forecasting import (
    mean_absolute_percentage_error,
    mean_absolute_scaled_error,
    mean_squared_scaled_error,
    median_absolute_scaled_error,
    median_squared_scaled_error
)
from sklearn.metrics import r2_score

# Preprocessing and Plotting
from sktime.transformations.series.detrend import Deseasonalizer, Detrender
from sktime.utils.plotting import plot_series

# Standard sktime forecasters
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.arima import ARIMA, AutoARIMA
from sktime.forecasting.bats import BATS
# from sktime.forecasting.fbprophet import Prophet

# Regressors to be used as Forecasters
from sklearn.linear_model import Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# Save forecaster as file
import pickle

# Not used
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.base import ForecastingHorizon

simplefilter("ignore", FutureWarning)


def split_data(Series, test_size):
    y_train, y_test = temporal_train_test_split(Series, test_size = test_size)
    fh = np.arange(len(y_test)) + 1
    return y_train, y_test, fh


def pred_future(forecaster, y_test, predict_timestep):
    fh = np.arange(len(y_test), len(y_test) + predict_timestep) + 1
    y_future = forecaster.predict(fh)

    return y_future


def save_model(forecaster, filename):
    with open(filename, 'wb') as f:
        pickle.dump(forecaster, f)


def sktime_forecasters(Series, test_size):
    mean = NaiveForecaster(strategy='mean', sp=11, window_length=12)
    last = NaiveForecaster(strategy='last', sp=1)
    drift = NaiveForecaster(strategy='drift', window_length=14)

    Naive_mix = EnsembleForecaster(
        [
            ("mean", mean),
            ("last", last),
            ("drift", drift),
        ])

    forecaster_init = [Naive_mix,
                       ExponentialSmoothing(trend='add', seasonal='additive', sp=12),
                       AutoETS(auto=True, sp=12, n_jobs=-1),
                       AutoARIMA(sp=12, suppress_warnings=True),
                       BATS(sp=12, use_trend=True, use_box_cox=False)
                       ]

    y_train, y_test, fh = split_data(Series, test_size)
    forecaster_list = []

    for forecaster in forecaster_init:
        forecaster.fit(y_train)
        forecaster_list.append(forecaster)

    return forecaster_list, y_train, y_test, fh


# MAIN FUNCTION. Outputs csv predictions and forecaster model file in current directory, and the model.
def generate_forecast(Series, predict_timestep):

    test_size = 0.2

    MAPE = []
    MASE = []
    RMSSE = []
    MedASE = []
    RMedSSE = []
    R2 = []

    forecaster_list, y_train, y_test, fh = sktime_forecasters(Series, test_size)

    for forecaster in forecaster_list:
        y_pred = forecaster.predict(fh)
        MAPE.append(mean_absolute_percentage_error(y_pred, y_test))
        MASE.append(mean_absolute_scaled_error(y_pred, y_test, y_train))
        RMSSE.append(mean_squared_scaled_error(y_pred, y_test, y_train, square_root=False))
        MedASE.append(median_absolute_scaled_error(y_pred, y_test, y_train))
        RMedSSE.append(median_squared_scaled_error(y_pred, y_test, y_train, square_root=False))
        R2.append(r2_score(y_pred, y_test))

    best_score_index = [metric.index(min(metric)) for metric in [MAPE, MASE, RMSSE, MedASE, RMedSSE]]
    best_score_index.append(R2.index(max(R2)))

    best_model_index = mode(best_score_index)

    forecaster_names = ['Naive_mix', 'Exp_smooth', 'AutoETS', 'AutoARIMA', 'TBATS']

    best_forecaster = forecaster_list[best_model_index]

    y_future = pred_future(best_forecaster, y_test, predict_timestep)
    y_future = pd.DataFrame({'Time': y_future.index, 'Value': y_future.values})

    y_future.to_csv(forecaster_names[best_model_index] + ' Prediction.csv')
    save_model(forecaster, forecaster_names[best_model_index] + ' Model')

    return best_forecaster