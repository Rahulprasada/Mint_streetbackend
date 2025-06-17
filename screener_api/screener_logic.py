# backend/screener_api/screener_logic.py

import pandas as pd
import numpy as np
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import RobustScaler
import yaml # Kept yaml import as the default config might start from yaml, but we'll use json
import psutil
# import aiohttp # Removed async fetching functions
# import asyncio # Removed async fetching functions
from scipy.stats.mstats import winsorize
import plotly.graph_objs as go # Keep for potential figure generation internally, but typically data is sent to frontend for plotting
import logging
from typing import List, Dict, Optional, Tuple, Any
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import multiprocessing
import json
import joblib
import os
import math

warnings.filterwarnings("ignore", category=UserWarning) # Filter specific warnings

# Set up logging - This will log to screener.log and console based on settings.py config
# Ensure this logger matches the one configured in settings.py
logger = logging.getLogger(__name__)
# Check if handlers are already configured by Django settings before adding defaults
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("screener.log"), # Path relative to manage.py
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__) # Re-get logger after basicConfig

logger.info("screener_logic.py loaded.")

# Predefined stock indices.
INDICES = {
    "NIFTY50": ([
        'KOTAKBANK', 'NTPC', 'TECHM', 'POWERGRID', 'SBIN', 'AXISBANK', 'BAJFINANCE',
        'WIPRO', 'EICHERMOT', 'HCLTECH', 'BEL', 'GRASIM', 'SBILIFE', 'RELIANCE', 'LT',
        'TCS', 'TATAMOTORS', 'MARUTI', 'SHRIRAMFIN', 'ADANIPORTS', 'ITC', 'ICICIBANK',
        'HINDALCO', 'TATASTEEL', 'BAJAJ-AUTO', 'TATACONSUM', 'ONGC', 'ULTRACEMCO',
        'ADANIENSOL', 'BPCL', 'ASIANPAINT', 'JSWSTEEL', 'APOLLOHOSP', 'HDFCLIFE', 'DRREDDY',
        'COALINDIA', 'HEROMOTOCO', 'HINDUNILVR', 'NESTLEIND', 'CIPLA', 'BRITANNIA',
        'SUNPHARMA', 'M&M', 'TRENT', 'TITAN', 'INDUSINDBK'
    ], ".NS"),
    "SENSEX": ([
        'ASIANPAINT', 'AXISBANK', 'BAJAJ-AUTO', 'BAJFINANCE', 'BAJAJFINSV', 'BHARTIARTL',
        'HCLTECH', 'HDFCBANK', 'HINDUNILVR', 'ICICIBANK', 'INDUSINDBK', 'INFY', 'ITC',
        'JSWSTEEL', 'KOTAKBANK', 'LT', 'M&M', 'MARUTI', 'NESTLEIND', 'NTPC', 'POWERGRID',
        'RELIANCE', 'SBIN', 'SUNPHARMA', 'TCS', 'TATAMOTORS', 'TATASTEEL', 'TECHM', 'TITAN',
        'ULTRACEMCO'
    ], ".NS")
}

# Default result structure.
DEFAULT_RESULT = {
    'Stock': 'Unknown',
    'Latest Regime': 'N/A',
    'Mean Annualized Return (%)': 0.0,
    'Recommendation': 'ERROR',
    'Converged': False,
    'Data': None, # Will store DataFrame converted to dict
    'Error': None,
    'Initial Portfolio Value': 0.0,
    'Final Portfolio Value': 0.0,
    'Backtest Return (%)': 0.0,
    'Annualized Backtest Return (%)': 0.0,
    'Max Drawdown (%)': 0.0,
    'Sharpe Ratio': 0.0,
    'Sortino Ratio': 0.0,
    'Calmar Ratio': 0.0,
    'Treynor Ratio': 0.0,
    'Number of Trades': 0,
    'Win/Loss Ratio (%)': "No Trades",
    'Avg Holding Period (days)': "No Trades",
    'Benchmark Return (%)': 0.0,
    'Trades Executed': False,
    'Feature_Weights': {}, # Store weights if ensemble is used, or just info if single model
    'State_Characteristics': {} # Info about each state
}

# Load configuration from JSON if available; otherwise, use defaults.
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    'sma_period': 50,
    'volume_ma_period': 22,
    # Define the features used by default and available ones
    'available_features': ['Returns', 'Momentum', 'ShortMomentum', 'VolumeSpike', 'Volatility', 'ATR', 'RSI', 'RelVolume', 'Breakout', 'Anomaly'],
    'features': ['Returns', 'Momentum', 'Volatility'], # Default features to use
    'max_states': 4,
    'train_window': 252,
    'period': '2y', # Default data fetching params
    'interval': '1d',
    'max_workers': max(1, psutil.cpu_count() // 2), # Default based on system
    'use_rolling_window': False,
    'slippage': 0.001, # Default slippage
}

CONFIG = DEFAULT_CONFIG.copy()
try:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            loaded_config = json.load(f)
            CONFIG.update(loaded_config) # Update defaults with loaded values
        logger.info(f"Loaded configuration from {CONFIG_FILE}")
    else:
        logger.info(f"Configuration file {CONFIG_FILE} not found, using defaults.")
except Exception as e:
    logger.error(f"Error loading configuration from {CONFIG_FILE}: {e}", exc_info=True)
    logger.info("Using default configuration due to load error.")


# --------------------------
# Helper Functions with File Caching
# --------------------------
DATA_CACHE_DIR = "data_cache"
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# Add this helper function inside backend/screener_api/screener_logic.py
def convert_numpy_keys_to_int(data):
    """
    Recursively converts numpy integer keys in dictionaries to standard Python int keys.
    Leaves non-dictionary/list structures and non-numpy keys unchanged.
    Also converts numpy numeric *values* to standard Python types if needed (safer).
    """
    if isinstance(data, dict):
        converted_dict = {}
        for key, value in data.items():
            # Convert numpy integer keys
            if isinstance(key, (np.integer, np.int64, np.int32)): # Check against numpy integer types
                new_key = int(key)
            else:
                new_key = key

            # Recursively convert values
            converted_dict[new_key] = convert_numpy_keys_to_int(value)
        return converted_dict
    elif isinstance(data, list):
        return [convert_numpy_keys_to_int(item) for item in data]
    elif isinstance(data, (np.integer, np.floating, np.bool_)):
         # Convert numpy numeric or boolean *values* to standard Python types
         return data.item() if hasattr(data, 'item') else data # .item() extracts Python scalar
    elif isinstance(data, np.ndarray):
         # Convert numpy arrays to lists (especially if they are values)
         return data.tolist()
    else:
        return data


def fetch_stock_data(ticker: str, period: str, interval: str, min_data_points: int = 100, max_retries: int = 3) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Fetches stock data with file caching."""
    cache_file = os.path.join(DATA_CACHE_DIR, f"{ticker.replace('.', '_')}_{period}_{interval}.pkl")

    # Attempt to load from cache
    if os.path.exists(cache_file):
        try:
            df = joblib.load(cache_file)
            # Basic validation of cached data
            if isinstance(df, pd.DataFrame) and not df.empty and len(df) >= min_data_points:
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if all(col in df.columns for col in required_cols) and not df[required_cols].isna().all().any():
                     logger.debug(f"Cache hit for {ticker}")
                     return df, None
                else:
                     logger.warning(f"Cached data for {ticker} is invalid, fetching fresh.")
            else:
                logger.warning(f"Cached data for {ticker} is empty or insufficient, fetching fresh.")
        except Exception as e:
            logger.warning(f"Cache load failed for {ticker}: {e}, fetching fresh data.")
        # If cache fails validation or loading, delete it
        if os.path.exists(cache_file):
             try:
                 os.remove(cache_file)
                 logger.debug(f"Removed invalid cache file: {cache_file}")
             except Exception as e:
                 logger.warning(f"Failed to remove invalid cache file {cache_file}: {e}")

    # Fetch fresh data if cache fails or is not present
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    suffixes = ['.NS', '.BO'] 
    used_ticker = None

    for suffix in suffixes:
        current_ticker = ticker if ticker.endswith(suffix) else ticker + suffix
        for attempt in range(max_retries):
            try:
                df = yf.download(current_ticker, period=period, interval=interval, progress=False, auto_adjust=True)

                if not isinstance(df, pd.DataFrame) or df.empty:
                     logger.debug(f"Fetch attempt {attempt + 1} for {current_ticker} returned empty data.")
                     time.sleep(1) # Wait before retry
                     continue

                if df.index.duplicated().any():
                    df = df[~df.index.duplicated(keep='last')]
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                # Validate required columns and data quality
                if not all(col in df.columns for col in required_cols):
                    logger.warning(f"Missing required columns for {current_ticker}: {df.columns}")
                    time.sleep(1) # Wait before retry
                    continue

                df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce')

                # Check for complete NaN or zero columns in required data
                if any(df[col].isna().all() or (df[col] == 0).all() for col in required_cols):
                     logger.warning(f"Invalid data (all NaN or all zero) in required columns for {current_ticker}.")
                     time.sleep(1) # Wait before retry
                     continue

                # Drop rows with NaNs in required columns and check length again
                original_len = len(df)
                df_cleaned = df[required_cols].dropna()

                if len(df_cleaned) < min_data_points:
                    logger.warning(f"Insufficient valid data for {current_ticker}: {len(df_cleaned)} < {min_data_points}")
                    time.sleep(1) # Wait before retry
                    continue

                if len(df_cleaned) < original_len:
                    logger.debug(f"Dropped {original_len - len(df_cleaned)} rows with NaNs for {current_ticker}")

                # Optional: check overall NaN ratio after dropna
                if df_cleaned.isna().mean().mean() > 0.01: # A very small threshold after dropping NaNs
                     logger.warning(f"Too many remaining NaNs in {current_ticker} data after cleaning.")
                     time.sleep(1) # Wait before retry
                     continue

                # Data seems valid and sufficient, save to cache and return
                df = df_cleaned # Use the cleaned DataFrame
                joblib.dump(df, cache_file)
                logger.info(f"Fetched and cached data for {current_ticker}, shape={df.shape}")
                return df, None

            except Exception as e:
                logger.error(f"Fetch attempt {attempt + 1} failed for {current_ticker}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2) # Longer wait on error

        # If the first suffix didn't work, try the next one
        if suffix != suffixes[-1]:
            logger.warning(f"Failed to fetch {ticker} with suffix {suffix}. Trying next suffix.")
            continue
        else:
            # Failed all suffixes and all retries
            break

    return None, f"Failed to fetch data for {ticker} after {max_retries} attempts with suffixes {suffixes}. Last error: {e if 'e' in locals() else 'Unknown error during fetch loop.'}"

# Remove async data fetching functions as the view will call the sync one
# async def fetch_stock_data_async(...) # Remove
# async def fetch_batch_data(...) # Remove


def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """Calculates Simple Moving Average."""
    if data.empty or window <= 0:
        return pd.Series(index=data.index, dtype=float)
    return data.rolling(window=window, min_periods=1).mean()

def calculate_bollinger_bands(df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
    """Calculates Bollinger Bands."""
    if df.empty:
         return pd.Series(dtype=float), pd.Series(dtype=float)
    sma = calculate_sma(df['Close'], window)
    rolling_std = df['Close'].rolling(window=window, min_periods=1).std()
    upper_band = sma + 2 * rolling_std
    lower_band = sma - 2 * rolling_std
    return upper_band, lower_band

def calculate_atr(df: pd.DataFrame, window: int) -> pd.Series:
    """Calculates Average True Range."""
    if df.empty:
        return pd.Series(dtype=float)
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean() # ATR usually uses min_periods=window

def detect_anomalies(df: pd.DataFrame) -> pd.Series:
    """Detects price anomalies using Z-scores on returns."""
    if df.empty or 'Returns' not in df.columns or df['Returns'].std() == 0:
        return pd.Series(0, index=df.index)
    returns = df['Returns'].dropna()
    if returns.empty:
         return pd.Series(0, index=df.index)
    z_scores = (returns - returns.mean()) / returns.std()
    anomaly_series = pd.Series((np.abs(z_scores) > 3).astype(int), index=returns.index)
    return anomaly_series.reindex(df.index, fill_value=0)


def calculate_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Calculates Relative Strength Index."""
    if series.empty or window <= 1:
        return pd.Series(index=series.index, dtype=float)

    delta = series.diff(1)
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    # Using EWM for accuracy in RSI calculation over rolling mean
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / (avg_loss + 1e-10) # Add small epsilon to avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi

def preprocess_data(df: pd.DataFrame, features: List[str], window: int = 3) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Preprocesses the stock data by adding technical indicators as features.

    Args:
        df: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume'.
        features: List of feature names to calculate.
        window: Rolling window size for some feature calculations.

    Returns:
        Tuple of processed DataFrame and an optional error message.
    """
    df = df.copy()
    # Use features requested from the UI/API, but validate against available_features
    requested_features = [f for f in features if f in CONFIG.get('available_features', [])]
    if not requested_features:
         logger.warning("No valid features requested for preprocessing, defaulting to Returns.")
         requested_features = ['Returns'] # Fallback to Returns

    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    if not isinstance(df, pd.DataFrame) or df.empty or not all(col in df.columns for col in required_cols):
        error_msg = f"Invalid or empty DataFrame for preprocessing: shape={df.shape}, columns={df.columns}"
        logger.error(error_msg)
        return pd.DataFrame(), error_msg

    # Winsorize extreme values before calculations
    try:
        for col in required_cols:
            if df[col].isna().all() or (df[col] == 0).all():
                error_msg = f"Column {col} is invalid (all NaN or all zero) before winsorizing."
                logger.error(error_msg)
                return pd.DataFrame(), error_msg
            df[col] = pd.Series(winsorize(df[col].values, limits=[0.01, 0.01]), index=df.index) # Winsorize 1% tails
            # After winsorizing, check again in case winsorize failed somehow
            if df[col].isna().all() or (df[col] == 0).all():
                 error_msg = f"Column {col} is invalid (all NaN or all zero) after winsorizing."
                 logger.error(error_msg)
                 return pd.DataFrame(), error_msg
    except Exception as e:
        error_msg = f"Error during winsorizing: {e}"
        logger.error(error_msg, exc_info=True)
        return pd.DataFrame(), error_msg

    # Ensure price data is strictly positive for log returns / ratios
    price_cols = ['Open', 'High', 'Low', 'Close']
    for col in price_cols:
        if (df[col] <= 0).any():
             # Replace non-positive values or adjust
             df[col] = df[col].replace(0, np.nan).ffill().bfill()
             if (df[col] <= 0).any():
                 error_msg = f"Non-positive price data found in column {col} after handling zeros."
                 logger.error(error_msg)
                 return pd.DataFrame(), error_msg

    # Calculate core MAs needed for some features or backtest
    df['PriceMA'] = calculate_sma(df['Close'], window=CONFIG.get('sma_period', 50))
    df['VolumeMA'] = calculate_sma(df['Volume'], window=CONFIG.get('volume_ma_period', 22))

    feature_functions = {
        'Returns': lambda df: df['Close'].pct_change().replace([np.inf, -np.inf], np.nan),
        # Rolling mean on momentum to smooth it out slightly
        'Momentum': lambda df: (df['Close'] / df['Close'].shift(20) - 1).rolling(window=max(window, 2), min_periods=1).mean().replace([np.inf, -np.inf], np.nan),
        'ShortMomentum': lambda df: (df['Close'] / df['Close'].shift(5) - 1).replace([np.inf, -np.inf], np.nan),
        'VolumeSpike': lambda df: (df['Volume'] / df['Volume'].rolling(window=5, min_periods=1).mean() > 1.5).astype(int).replace([np.inf, -np.inf], np.nan),
        'Volatility': lambda df: df['Close'].pct_change().rolling(window=20, min_periods=1).std().replace([np.inf, -np.inf], np.nan),
        'RelVolume': lambda df: (df['Volume'] / df['VolumeMA']).replace([np.inf, -np.inf], np.nan),
        'Breakout': lambda df: ((df['Close'] > calculate_bollinger_bands(df)[0]).astype(int) - (df['Close'] < calculate_bollinger_bands(df)[1]).astype(int)).rolling(window=max(window, 2), min_periods=1).mean().replace([np.inf, -np.inf], np.nan),
        'ATR': lambda df: calculate_atr(df, window=14).replace([np.inf, -np.inf], np.nan),
        'Anomaly': lambda df: detect_anomalies(df).replace([np.inf, -np.inf], np.nan),
        'RSI': lambda df: calculate_rsi(df['Close']).replace([np.inf, -np.inf], np.nan)
        # PriceMA and VolumeMA are calculated but not typically used *as HMM features* themselves,
        # but can be included if requested or used in backtest rules. Let's add them to available.
    }

    calculated_features = []
    temp_df = df.copy() # Use a temp df to avoid modifying the original until successful

    # Always calculate Returns as it's needed for backtesting and default fallback
    if 'Returns' not in temp_df.columns:
         try:
             temp_df['Returns'] = feature_functions['Returns'](temp_df)
             # Validate Returns column immediately
             if temp_df['Returns'].isna().all() or temp_df['Returns'].var() < 1e-12:
                 error_msg = "Calculated Returns column is invalid or has zero variance."
                 logger.error(error_msg)
                 return pd.DataFrame(), error_msg
             calculated_features.append('Returns')
         except Exception as e:
             error_msg = f"Failed to calculate essential Returns feature: {e}"
             logger.error(error_msg, exc_info=True)
             return pd.DataFrame(), error_msg

    for feature in requested_features:
        if feature == 'Returns': continue # Already handled
        if feature not in feature_functions:
            logger.warning(f"Feature {feature} not supported, skipping.")
            continue
        try:
            temp_df[feature] = feature_functions[feature](temp_df)
            # Check validity of the calculated feature
            if temp_df[feature].isna().all() or temp_df[feature].var() < 1e-12:
                 logger.warning(f"Calculated feature {feature} has invalid data or near-zero variance ({temp_df[feature].var():.2e}), dropping.")
                 temp_df.drop(columns=[feature], inplace=True)
            else:
                calculated_features.append(feature)
        except Exception as e:
            logger.error(f"Failed to calculate feature {feature}: {e}", exc_info=True)
            temp_df.drop(columns=[feature], errors='ignore', inplace=True) # Ensure column is dropped if calculation failed

    # Ensure at least 'Returns' is available and valid
    if 'Returns' not in calculated_features:
         error_msg = "Essential 'Returns' feature is missing or invalid."
         logger.error(error_msg)
         return pd.DataFrame(), error_msg

    # Check for high multicollinearity *among the calculated_features*
    if len(calculated_features) > 1:
        try:
            corr_matrix = temp_df[calculated_features].corr().abs()
            # Check for highly correlated pairs, exclude self-correlation
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            if to_drop:
                logger.warning(f"High multicollinearity (>0.95) detected among {calculated_features}. Dropping highly correlated features: {to_drop}")
                temp_df.drop(columns=to_drop, inplace=True)
                calculated_features = [f for f in calculated_features if f not in to_drop]
        except Exception as e:
            logger.warning(f"Multicollinearity check failed: {e}, retaining all features.", exc_info=True)

    # Final check for NaNs and length after calculating all features
    final_features_for_hmm = [f for f in calculated_features if f != 'Anomaly'] # Anomaly is a binary signal, not typically fed to HMM Gaussian distribution
    final_features_for_hmm = [f for f in final_features_for_hmm if f in temp_df.columns] # Ensure column exists

    if not final_features_for_hmm:
         # Should not happen if Returns is guaranteed, but safety check
         error_msg = "No suitable features remaining after calculation and filtering for HMM."
         logger.error(error_msg)
         return pd.DataFrame(), error_msg

    # Drop rows with NaNs in the final features used for HMM + essential cols
    subset_cols_for_dropna = final_features_for_hmm + required_cols
    subset_cols_for_dropna = [col for col in subset_cols_for_dropna if col in temp_df.columns]

    initial_rows = len(temp_df)
    temp_df.dropna(subset=subset_cols_for_dropna, inplace=True)
    rows_dropped = initial_rows - len(temp_df)
    if rows_dropped > 0:
        logger.debug(f"Dropped {rows_dropped} rows with NaNs in final features/required columns.")

    if temp_df.empty or len(temp_df) < 60: # HMM needs sufficient data
        error_msg = f"Insufficient valid data after preprocessing (<60 days): {len(temp_df)} rows remaining."
        logger.error(error_msg)
        return pd.DataFrame(), error_msg

    # Re-check variance for the features actually used for HMM after dropping NaNs
    variances = temp_df[final_features_for_hmm].var()
    low_variance_features = variances[variances < 1e-8].index.tolist()
    if low_variance_features:
        # Should ideally not happen after initial check, but adding robustness
        error_msg = f"Features still have too low variance after dropping NaNs: {low_variance_features}"
        logger.error(error_msg)
        return pd.DataFrame(), error_msg

    # Add Anomaly back if it was calculated and exists
    if 'Anomaly' in df.columns and 'Anomaly' not in temp_df.columns:
         # Anomaly calculation was done on original df, now add it to temp_df and align index
         temp_df['Anomaly'] = df['Anomaly'] # This might introduce NaNs if original df had NaNs

    # Final ffill/bfill for any remaining NaNs introduced by shifts/rolling *in non-HMM features*
    # Or just drop rows if any feature has NaNs? Dropping is safer for HMM.
    # We already dropped NaNs in `final_features_for_hmm`. Other columns might still have NaNs (like PriceMA, VolumeMA if window > data start).
    # Let's just ensure the final features used by HMM are clean.
    df = temp_df # Assign the processed temporary dataframe back
    df = df.copy() # Avoid SettingWithCopyWarning
    df['Features_Used'] = [final_features_for_hmm] * len(df) # Store features actually used for HMM

    logger.info(f"Preprocessing complete, shape={df.shape}, features calculated={calculated_features}, features for HMM={final_features_for_hmm}")
    return df, None


def calculate_aic(log_likelihood: float, n_params: int) -> float:
    """Calculates Akaike Information Criterion."""
    return -2 * log_likelihood + 2 * n_params

def calculate_bic(log_likelihood: float, n_params: int) -> float:
    """Calculates Bayesian Information Criterion."""
    # N = number of samples (data points)
    # k = number of parameters (n_params)
    # L = log likelihood (log_likelihood)
    # BIC = -2 * L + k * log(N)
    # HMM monitor_ doesn't directly expose N, get it from X_scaled shape
    # This function doesn't have access to X_scaled, so BIC calculation should be moved
    # into select_best_hmm or fit_hmm_ensemble where N is known.
    # For now, keep AIC as it was used before.
    raise NotImplementedError("BIC calculation requires sample size (N)")


def select_best_hmm(X_scaled: np.ndarray, features: List[str], max_states: int = 4, max_retries: int = 3) -> Tuple[Optional[GaussianHMM], Optional[float], Optional[str], Optional[float]]:
    """
    Selects the best HMM model based on AIC.

    Args:
        X_scaled: Scaled feature data (numpy array).
        features: List of feature names corresponding to X_scaled.
        max_states: Maximum number of states to test.
        max_retries: Number of retries for HMM fitting.

    Returns:
        Tuple of (best_model, best_aic, error_message, best_log_likelihood).
    """
    best_score = np.inf # Use AIC for comparison
    best_model = None
    best_log_likelihood = -np.inf
    errors = []
    logger.debug(f"Starting HMM selection for {len(features)} features, max_states={max_states}...")

    if X_scaled.shape[0] < 60 or X_scaled.shape[1] == 0:
         return None, None, "Insufficient data points (<60) or no features (0).", None

    if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
        return None, None, "Invalid values (NaN/Inf) in scaled data.", None

    variances = np.var(X_scaled, axis=0)
    if any(v < 1e-8 for v in variances):
        zero_var_features = [features[i] for i, v in enumerate(variances) if v < 1e-8]
        return None, None, f"Feature variance too low for HMM: {zero_var_features}", None

    # min_covar should be small but positive, related to feature scales
    # A heuristic: minimum of variances / 100, or a small absolute value
    min_covar = max(1e-5, np.min(variances) * 0.01)
    logger.debug(f"Using min_covar={min_covar:.2e}")

    # Try state numbers from 2 up to max_states
    # Add a small buffer (+1 or +2) just in case, but stick closer to max_states requested
    states_to_test = range(2, max_states + 1)
    if not states_to_test:
         states_to_test = [2] # Ensure at least 2 states are tested if max_states is < 2

    for n_states in states_to_test:
        for retry in range(max_retries):
            # Use random seeds to encourage finding different local optima on retries
            seed = int((time.time() * 1000) % 1000000) + retry * 100

            for cov_type in ['diag', 'full']: # diag is faster/simpler, full is more complex
                # Full covariance requires n_components <= n_features for full matrix
                if cov_type == 'full' and n_states > X_scaled.shape[1]:
                     logger.debug(f"Skipping full covar for {n_states} states with {X_scaled.shape[1]} features.")
                     continue

                model = GaussianHMM(n_components=n_states, covariance_type=cov_type, n_iter=200, tol=1e-4, # Increased iter/tightened tol slightly
                                    random_state=seed, init_params='kmeans', # Using kmeans init usually helps
                                    params='stmc', # s: means, t: transition matrix, m: emission covars
                                    min_covar=min_covar)
                try:
                    model.fit(X_scaled)

                    if hasattr(model.monitor_, 'converged') and model.monitor_.converged:
                         log_likelihood = model.score(X_scaled)
                         # Calculate number of parameters for AIC/BIC
                         # n_states initial probabilities (n_states-1 free)
                         # n_states * n_states transition probabilities (n_states * (n_states-1) free)
                         # n_states * n_features means
                         # n_states * (n_features * (n_features + 1) / 2) covars for full
                         # n_states * n_features covars for diag
                         n_features = X_scaled.shape[1]
                         n_params = (n_states - 1) + n_states * (n_states - 1) + n_states * n_features
                         if cov_type == 'full':
                             n_params += n_states * (n_features * (n_features + 1) // 2)
                         elif cov_type == 'diag':
                             n_params += n_states * n_features

                         aic = calculate_aic(log_likelihood, n_params)
                         # bic = -2 * log_likelihood + n_params * np.log(X_scaled.shape[0]) # Calculate BIC here

                         logger.debug(f"HMM fit converged (states={n_states}, cov={cov_type}, retry={retry}): logL={log_likelihood:.2f}, AIC={aic:.2f}")

                         if aic < best_score:
                             logger.debug(f"New best model found: states={n_states}, cov={cov_type}, AIC={aic:.2f} (improvement: {best_score - aic:.2f})")
                             best_score, best_model, best_log_likelihood = aic, model, log_likelihood

                         # If converged, break the retry loop for this state/cov_type
                         break
                    else:
                        logger.debug(f"HMM fit did not converge (states={n_states}, cov={cov_type}, retry={retry})")

                except ValueError as ve:
                    # Catch specific ValueError from hmmlearn when fit fails (e.g., insufficient data for state)
                    errors.append(f"HMM ValueError for {n_states} states, {cov_type}, retry {retry}: {ve}")
                    logger.warning(f"HMM ValueError for {n_states} states, {cov_type}, retry {retry}: {ve}")
                except Exception as e:
                    errors.append(f"HMM fitting failed for {n_states} states, {cov_type}, retry {retry}: {e}")
                    logger.error(f"HMM error for {n_states} states, {cov_type}, retry {retry}: {e}", exc_info=True)

    if best_model is None:
        error_msg = f"No HMM model fit successfully for feature set. Errors: {'; '.join(errors[-5:])}" # Report last few errors
        logger.error(error_msg)
        return None, None, error_msg, None

    logger.info(f"Best HMM model selected with AIC={best_score:.2f}")
    return best_model, best_score, None, best_log_likelihood


def fit_hmm_ensemble(df: pd.DataFrame, feature_sets: List[List[str]], max_states: int = 4,
                     train_window: int = 252, use_rolling_window: bool = False) -> Tuple[pd.DataFrame, bool, Optional[str], Optional[Dict[str, Any]]]:
    """
    Fits multiple HMMs using different feature sets and combines their results (ensemble).

    Args:
        df: Preprocessed DataFrame.
        feature_sets: List of lists, where each inner list is a set of feature names.
        max_states: Maximum number of states for HMMs.
        train_window: Number of recent days to use for training if use_rolling_window is True.
        use_rolling_window: Whether to use only recent data for training.

    Returns:
        Tuple of (DataFrame with Regime/State, convergence status, error message, ensemble info).
        Ensemble info includes weighted probs and state characteristics.
    """
    df = df.copy()
    all_available_features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'PriceMA', 'VolumeMA', 'Features_Used']]
    required_cols_for_hmm_data = ['Close', 'Returns'] # Need these even if not in features

    fitted_models = [] # Store (model, aic, logL, features, scaler) tuples

    logger.debug(f"Attempting to fit HMMs for {len(feature_sets)} feature sets.")

    for features in feature_sets:
        valid_features = [f for f in features if f in all_available_features]
        if not valid_features:
            logger.warning(f"Feature set {features} has no valid features in DataFrame, skipping.")
            continue

        # Ensure required columns like Returns are in the features if they aren't already, but don't add if not in data
        final_features_for_set = list(set(valid_features)) # Use set to ensure uniqueness
        if not final_features_for_set:
            logger.warning(f"Feature set {features} resulted in no valid features for HMM.")
            continue

        logger.info(f"Processing feature set for HMM: {final_features_for_set}")

        # Select training data
        train_df = df.copy()
        if use_rolling_window and len(df) > train_window:
            train_df = df.iloc[-train_window:]

        custom_min_window = max(60, 30 * max_states) # HMM needs sufficient data points
        if len(train_df) < custom_min_window:
            logger.warning(f"Training window too short ({len(train_df)} < {custom_min_window} days) for features {final_features_for_set}. Skipping this set.")
            continue # Skip this feature set

        try:
            X_train = train_df[final_features_for_set].values
            X_full = df[final_features_for_set].values

            if X_train.shape[0] == 0 or X_full.shape[0] == 0:
                 logger.warning(f"Empty data for feature set {final_features_for_set}, skipping.")
                 continue
            if X_train.shape[1] != len(final_features_for_set) or X_full.shape[1] != len(final_features_for_set):
                 logger.error(f"Feature mismatch: X_train={X_train.shape}, X_full={X_full.shape}, features={len(final_features_for_set)}. Skipping.")
                 continue

            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_full_scaled = scaler.transform(X_full)

            if np.isnan(X_train_scaled).any() or np.isinf(X_train_scaled).any():
                 logger.warning(f"Invalid values in scaled training data for {final_features_for_set}, skipping.")
                 continue

            model, aic, error, log_likelihood = select_best_hmm(X_train_scaled, final_features_for_set, max_states)

            if model is not None and model.monitor_.converged:
                fitted_models.append((model, aic, log_likelihood, final_features_for_set, scaler))
                logger.info(f"Successfully fitted HMM for features {final_features_for_set} with AIC={aic:.2f}")
            else:
                logger.warning(f"Failed to fit/converge HMM for features {final_features_for_set}. Error: {error}")

        except Exception as e:
            logger.error(f"Error processing feature set {final_features_for_set}: {e}", exc_info=True)

    if not fitted_models:
        error_msg = "No HMM models converged across all provided feature sets."
        logger.error(error_msg)
        return df, False, error_msg, None

    # Ensemble the results from fitted models
    ensemble_probs = []
    ensemble_aics = []
    ensemble_feature_sets_names = []

    for model, aic, logL, features, scaler in fitted_models:
        try:
            X_full_scaled = scaler.transform(df[features].values)
            probs = model.predict_proba(X_full_scaled)

            # Ensure probs match df length after any NaN removal in preprocess
            if len(probs) != len(df):
                logger.warning(f"Probabilities length mismatch for features {features}: {len(probs)} vs {len(df)}. Skipping this model in ensemble.")
                continue

            ensemble_probs.append(probs)
            ensemble_aics.append(aic)
            ensemble_feature_sets_names.append(str(features)) # Store string representation

        except Exception as e:
            logger.error(f"Error getting probabilities for ensemble for features {features}: {e}", exc_info=True)
            # Remove the model info from lists if processing failed
            idx = ensemble_feature_sets_names.index(str(features))
            ensemble_probs.pop(idx)
            ensemble_aics.pop(idx)
            ensemble_feature_sets_names.pop(idx)


    if not ensemble_probs:
        error_msg = "No models produced valid probabilities for ensemble."
        logger.error(error_msg)
        return df, False, error_msg, None

    # Calculate weights based on inverse AIC
    # exp(-0.5 * AIC) is proportional to the likelihood given the model
    # Summing these proportional likelihoods gives a normalizing constant
    # weights = exp(-0.5 * AIC_i) / sum(exp(-0.5 * AIC_j))
    # To avoid numerical issues with very large/small exp values,
    # use log-sum-exp trick or work with log-weights
    # log_weights = -0.5 * AIC_i
    # weights = exp(log_weights - logsumexp(log_weights))
    try:
        # Handle case where all AICs are identical (e.g., only one model or identical models)
        if len(set(ensemble_aics)) <= 1:
             weights = np.ones(len(ensemble_aics)) / len(ensemble_aics)
             logger.debug("Using uniform weights due to identical AICs.")
        else:
            log_weights = -0.5 * np.array(ensemble_aics)
            # More numerically stable way to calculate softmax (normalized exp)
            # weights = np.exp(log_weights - np.max(log_weights)) # Shift for stability
            # weights = weights / np.sum(weights)

            # Even more stable:
            max_log_weight = np.max(log_weights)
            shifted_log_weights = log_weights - max_log_weight
            exp_shifted_log_weights = np.exp(shifted_log_weights)
            weights = exp_shifted_log_weights / np.sum(exp_shifted_log_weights)


        # Ensure weights sum to 1 (handle potential floating point issues)
        weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(weights)) / len(weights)


        logger.info(f"Ensemble weights: {dict(zip(ensemble_feature_sets_names, weights.round(4)))}")
    except Exception as e:
        logger.error(f"Error calculating ensemble weights: {e}, falling back to uniform weights.", exc_info=True)
        weights = np.ones(len(ensemble_aics)) / len(ensemble_aics)


    # Calculate weighted average probabilities
    weighted_avg_probs = np.average(ensemble_probs, axis=0, weights=weights)

    # Assign final state based on the maximum weighted average probability
    df['State'] = np.argmax(weighted_avg_probs, axis=1)
    df['State_Confidence'] = np.max(weighted_avg_probs, axis=1)

    # Determine Regime mapping based on state characteristics (mean return, momentum, vol)
    state_stats = {}
    for state in sorted(df['State'].unique()): # Sort states numerically
        state_df = df[df['State'] == state]
        mean_return = state_df['Returns'].mean() if 'Returns' in state_df.columns else 0.0
        # Use median momentum if available as it's less sensitive to outliers
        mean_momentum = state_df['Momentum'].median() if 'Momentum' in state_df.columns else 0.0
        volatility = state_df['Returns'].std() if 'Returns' in state_df.columns else 0.0 # Use returns std for state volatility
        state_stats[state] = {
            'mean_return': mean_return,
            'mean_momentum': mean_momentum,
            'volatility': volatility,
            'count': len(state_df)
        }
    # Sort states to assign labels (Bullish, Bearish, Sideways)
    # Primary key: mean return, Secondary key: mean momentum
    sorted_states = sorted(state_stats, key=lambda s: (state_stats[s]['mean_return'], state_stats[s]['mean_momentum']), reverse=True)

    # Add descriptive labels based on state characteristics
    regime_labels = {}
    # Calculate average volatility across all states for comparison
    all_vols = [s['volatility'] for s in state_stats.values() if s['count'] > 0]
    avg_overall_vol = np.mean(all_vols) if all_vols else 0

    for i, state in enumerate(sorted_states):
        stats = state_stats[state]
        base_label = ""
        if i == 0:
            base_label = "Bullish"
        elif i == len(sorted_states) - 1:
            base_label = "Bearish"
        else:
            base_label = f"Sideways {i}" # Differentiate intermediate states

        # Add descriptors
        descriptors = []
        if stats['volatility'] > avg_overall_vol * 1.2: # Define high volatility threshold (e.g., 20% above average)
            descriptors.append("High Vol")
        elif avg_overall_vol > 0 and stats['volatility'] < avg_overall_vol * 0.8: # Define low volatility threshold
             descriptors.append("Low Vol")

        # Add momentum descriptor if Momentum feature was used
        if 'Momentum' in df.columns:
             all_moms = [s['mean_momentum'] for s in state_stats.values() if s['count'] > 0]
             avg_overall_mom = np.mean(all_moms) if all_moms else 0
             if stats['mean_momentum'] > avg_overall_mom * 1.5: # Define high momentum threshold
                 descriptors.append("High Mom")
             elif avg_overall_mom > 0 and stats['mean_momentum'] < avg_overall_mom * 0.5: # Define low momentum threshold
                 descriptors.append("Low Mom")


        if descriptors:
             regime_labels[state] = f"{base_label} ({', '.join(descriptors)})"
        else:
             regime_labels[state] = base_label

    df['Regime'] = df['State'].map(regime_labels).fillna('Unknown') # Handle potential missing states

    # Calculate Regime Change Signal
    try:
        # Probability difference between consecutive days
        prob_diff = np.abs(weighted_avg_probs[:-1] - weighted_avg_probs[1:]).max(axis=1)
        df['Regime_Change_Signal'] = 0
        # Signal if the maximum probability difference across all states is above a threshold
        change_threshold = 0.3 # Threshold for significant probability shift
        df.iloc[1:, df.columns.get_loc('Regime_Change_Signal')] = (prob_diff > change_threshold).astype(int)
        logger.debug(f"Calculated Regime_Change_Signal. Max prob diff: {prob_diff.max():.2f}")
    except Exception as e:
        logger.warning(f"Failed to compute Regime_Change_Signal: {e}", exc_info=True)
        df['Regime_Change_Signal'] = 0 # Default to no signal on error

    ensemble_info = {
        'weights': dict(zip(ensemble_feature_sets_names, weights.tolist())), # Store weights used
        'state_characteristics': state_stats # Store characteristics of each state
    }

    logger.info(f"HMM ensemble complete. Final Data shape={df.shape}.")
    return df, True, None, ensemble_info

def get_recommendation(regime: str, latest_returns: pd.Series, latest_momentum: pd.Series, df_full: pd.DataFrame) -> str:
    """
    Generates a trading recommendation based on the current regime and recent data.

    Args:
        regime: The latest identified regime string.
        latest_returns: Recent return value(s).
        latest_momentum: Recent momentum value(s).
        df_full: The full processed DataFrame to calculate percentiles.

    Returns:
        Recommendation string ('STRONG BUY', 'BUY', 'HOLD', 'SELL', 'WAIT', 'INCONCLUSIVE').
    """
    if df_full.empty or 'Returns' not in df_full.columns:
        return 'INCONCLUSIVE' # Cannot calculate quantiles

    # Use quantiles of historical features from the full dataset for thresholds
    try:
        returns_threshold_buy = df_full['Returns'].quantile(0.65) if not df_full['Returns'].empty else 0.0
        returns_threshold_sell = df_full['Returns'].quantile(0.35) if not df_full['Returns'].empty else 0.0
        momentum_threshold_buy = df_full['Momentum'].quantile(0.65) if 'Momentum' in df_full.columns and not df_full['Momentum'].empty else 0.0
        # momentum_threshold_sell = df_full['Momentum'].quantile(0.35) if 'Momentum' in df_full.columns else 0.0
    except Exception as e:
        logger.warning(f"Could not calculate recommendation quantiles: {e}. Using default thresholds.", exc_info=True)
        returns_threshold_buy = 0.001 # Small positive return
        returns_threshold_sell = -0.001 # Small negative return
        momentum_threshold_buy = 0.0 # Positive momentum

    # Get the latest values
    latest_return = latest_returns.iloc[-1] if not latest_returns.empty else 0.0
    latest_mom = latest_momentum.iloc[-1] if not latest_momentum.empty else 0.0

    logger.debug(f"Regime: {regime}, Latest Return: {latest_return:.4f}, Latest Momentum: {latest_mom:.4f}")
    logger.debug(f"Thresholds: Ret Buy={returns_threshold_buy:.4f}, Ret Sell={returns_threshold_sell:.4f}, Mom Buy={momentum_threshold_buy:.4f}")

    if "Bullish" in regime:
        if latest_return > returns_threshold_buy or latest_mom > momentum_threshold_buy:
            return 'STRONG BUY'
        elif latest_return > 0:
            return 'BUY'
        return 'HOLD'
    elif "Sideways" in regime:
        # In sideways, look for positive recent performance indicators
        if latest_return >= 0 and latest_mom > momentum_threshold_buy:
             return 'BUY' # Bullish tilt in sideways
        elif latest_return >= 0:
             return 'HOLD' # Neutral/slightly positive sideways
        elif latest_return > returns_threshold_sell:
             return 'WAIT' # Slightly negative but not strongly bearish
        return 'SELL' # Negative tilt in sideways
    elif "Bearish" in regime:
        # In bearish, primarily recommend selling or holding cash
        return 'SELL'
    return 'INCONCLUSIVE' # Fallback

def calculate_risk_metrics(df: pd.DataFrame, portfolio_returns: pd.Series, benchmark_returns: pd.Series, period_days: int) -> Dict:
    """
    Calculates standard financial risk metrics.

    Args:
        df: Original or processed DataFrame (needed for 'Returns' if portfolio_returns is empty).
        portfolio_returns: Series of daily portfolio returns.
        benchmark_returns: Series of daily benchmark returns.
        period_days: Number of trading days in the period.

    Returns:
        Dictionary of risk metrics.
    """
    metrics = {
        'Max Drawdown (%)': 0.0,
        'Sharpe Ratio': 0.0,
        'Sortino Ratio': 0.0,
        'Calmar Ratio': 0.0,
        'Treynor Ratio': 0.0,
        'Beta': 0.0 # Add Beta for context
    }

    if portfolio_returns.empty:
        logger.warning("Portfolio returns are empty, cannot calculate risk metrics.")
        return metrics

    # Ensure returns are aligned by date if necessary, or assume they are
    # For simplicity, assume portfolio_returns index is the same as df index slice used
    # and benchmark_returns index is also aligned.
    # Let's assume they are pandas Series with matching indices.

    cum_returns = (1 + portfolio_returns).cumprod()
    max_drawdown_series = (cum_returns / cum_returns.cummax() - 1)
    max_drawdown = max_drawdown_series.min() * 100 if not max_drawdown_series.empty else 0.0

    # Ensure non-zero period_days for annualization
    if period_days <= 0:
        logger.warning("Period days is zero or negative, cannot annualize metrics.")
        # Return daily metrics or zeros
        mean_return_daily = portfolio_returns.mean()
        std_dev_daily = portfolio_returns.std()
        downside_returns_daily = portfolio_returns[portfolio_returns < 0]
        downside_dev_daily = downside_returns_daily.std() if not downside_returns_daily.empty else 0.0
        # Cannot calculate Treynor/Calmar/Annualized Return without period_days
        return {
             'Max Drawdown (%)': round(max_drawdown, 2),
             'Sharpe Ratio': round(mean_return_daily / std_dev_daily if std_dev_daily > 0 else 0.0, 2),
             'Sortino Ratio': round(mean_return_daily / downside_dev_daily if downside_dev_daily > 0 else 0.0, 2),
             'Calmar Ratio': 0.0, # Cannot annualize
             'Treynor Ratio': 0.0, # Cannot annualize
             'Beta': 0.0
        }


    # Annualized metrics assume daily data * 252 trading days/year
    annualization_factor = np.sqrt(252) # For volatility
    annualized_mean_return = portfolio_returns.mean() * 252 # For mean return

    std_dev_annualized = portfolio_returns.std() * annualization_factor

    sharpe = annualized_mean_return / std_dev_annualized if std_dev_annualized > 0 else 0.0

    downside_returns = portfolio_returns[portfolio_returns < 0]
    downside_dev_annualized = downside_returns.std() * annualization_factor if not downside_returns.empty else 0.0
    sortino = annualized_mean_return / downside_dev_annualized if downside_dev_annualized > 0 else 0.0

    # Calmar Ratio requires annualized return over the backtest period
    # Annualized return is based on the total return over the backtest period
    total_portfolio_return = cum_returns.iloc[-1] - 1
    if total_portfolio_return is None or math.isnan(total_portfolio_return):
         total_portfolio_return = 0.0

    # Number of years in the backtest period
    years_in_backtest = period_days / 252.0 # Assuming 252 trading days per year

    if years_in_backtest > 0:
        annualized_total_return = ((1 + total_portfolio_return) ** (1 / years_in_backtest)) - 1 if total_portfolio_return >= 0 else -((1 - abs(total_portfolio_return)) ** (1 / years_in_backtest)) + 1 # Handle negative returns
        calmar = annualized_total_return / abs(max_drawdown / 100) if max_drawdown != 0 else 0.0
    else:
        annualized_total_return = 0.0
        calmar = 0.0


    beta = 0.0
    treynor = 0.0
    if not benchmark_returns.empty and benchmark_returns.var() > 0:
        # Align benchmark returns to portfolio returns index
        aligned_benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).dropna()
        aligned_portfolio_returns = portfolio_returns.reindex(aligned_benchmark_returns.index).dropna()

        if not aligned_portfolio_returns.empty and not aligned_benchmark_returns.empty:
            # Ensure consistent length and non-zero variance for calculation
            if len(aligned_portfolio_returns) == len(aligned_benchmark_returns) and aligned_benchmark_returns.var() > 1e-9:
                 cov = aligned_portfolio_returns.cov(aligned_benchmark_returns)
                 benchmark_var = aligned_benchmark_returns.var()
                 beta = cov / benchmark_var if benchmark_var > 0 else 0.0
                 # Risk-free rate is assumed 0 for Treynor calculation here
                 treynor = annualized_mean_return / beta if beta != 0 else 0.0
            else:
                 logger.warning("Cannot calculate Beta/Treynor: data length mismatch or benchmark variance is zero.")
        else:
            logger.warning("Cannot calculate Beta/Treynor: aligned returns are empty.")


    return {
        'Max Drawdown (%)': round(max_drawdown, 2),
        'Sharpe Ratio': round(sharpe, 2),
        'Sortino Ratio': round(sortino, 2),
        'Calmar Ratio': round(calmar, 2),
        'Treynor Ratio': round(treynor, 2),
        'Beta': round(beta, 4)
    }


def backtest_strategy(df: pd.DataFrame, initial_cash: float = 100000, slippage: float = 0.001) -> Dict:
    """
    Backtests the HMM regime strategy with a dynamic trailing stop.

    Args:
        df: Processed DataFrame including 'Close', 'Regime', 'Returns', 'Momentum', 'ATR'.
        initial_cash: Starting cash for the backtest.
        slippage: Transaction cost and slippage per trade (percentage).

    Returns:
        Dictionary of backtest results.
    """
    df = df.copy()
    required_cols = ['Close', 'Regime']
    # Ensure essential columns exist after preprocessing and HMM fitting
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"Missing required columns for backtest: {missing}")
        res = DEFAULT_RESULT.copy()
        res['Error'] = f"Missing required columns for backtest: {missing}"
        return res

    # Ensure other needed columns are present or handled
    if 'Returns' not in df.columns:
        df['Returns'] = df['Close'].pct_change()
        logger.warning("Returns column missing, calculated during backtest.")
    if 'Momentum' not in df.columns:
         df['Momentum'] = df['Returns'].rolling(window=20).mean() # Fallback momentum
         logger.warning("Momentum column missing, using Returns rolling mean as fallback.")
    if 'ATR' not in df.columns or df['ATR'].isna().all():
        # Fallback ATR using rolling std if calculated ATR is missing/invalid
        df['ATR_fallback'] = df['Close'].pct_change().rolling(window=14).std() * df['Close'].rolling(window=14).mean()
        if df['ATR_fallback'].isna().all():
             df['ATR'] = df['Close'].diff().abs().rolling(window=14).mean().fillna(df['Close'].diff().abs().mean()) # Simple diff mean fallback
             logger.warning("ATR calculation failed, using simple diff mean as fallback.")
        else:
             df['ATR'] = df['ATR_fallback']
             logger.warning("ATR calculation failed, using Returns rolling std based fallback.")

    # Drop rows with NaNs in critical columns before backtest simulation
    initial_len = len(df)
    df.dropna(subset=['Close', 'Regime', 'Returns', 'Momentum', 'ATR'], inplace=True)
    if len(df) < initial_len:
        logger.warning(f"Dropped {initial_len - len(df)} rows with NaNs before backtest.")

    if df.empty:
         res = DEFAULT_RESULT.copy()
         res['Error'] = "Empty DataFrame after dropping NaNs before backtest."
         return res

    cash = initial_cash
    shares = 0
    portfolio_values = []
    trades = []
    position_entry_price = None
    position_entry_date = None
    highest_price_in_position = None
    atr_multiplier = 2.5 # Adjusted ATR multiplier for trailing stop
    transaction_cost_pct = slippage

    # Backtest starts from the first available date after NaNs are dropped
    start_date = df.index[0]
    initial_close_price = df['Close'].iloc[0]

    # Calculate benchmark portfolio value (buy and hold from start_date)
    benchmark_shares = initial_cash / initial_close_price if initial_close_price > 0 else 0
    benchmark_portfolio_values = (benchmark_shares * df['Close']).tolist() # Convert to list for json

    # Initialize portfolio value list with initial cash (day before start)
    portfolio_values.append(initial_cash)

    # Loop through days starting from the second day (index 1) as strategy logic depends on day 'i' data
    for i in range(1, len(df)):
        current_date = df.index[i]
        price = df['Close'].iloc[i]
        current_atr = df['ATR'].iloc[i]
        regime = df['Regime'].iloc[i]
        current_return = df['Returns'].iloc[i]
        current_momentum = df['Momentum'].iloc[i]

        # --- Check for Sell Signal (Trailing Stop or Bearish Regime) ---
        # Check if currently holding shares
        if shares > 0 and position_entry_price is not None and highest_price_in_position is not None:
            highest_price_in_position = max(highest_price_in_position, price)
            # Dynamic Trailing Stop: highest price achieved minus ATR multiple
            trailing_stop_price = highest_price_in_position - atr_multiplier * current_atr

            # Check if stop loss is hit OR regime changes to Bearish
            sell_signal = False
            sell_reason = None
            if price <= trailing_stop_price:
                sell_signal = True
                sell_reason = 'Trailing Stop Hit'
                logger.debug(f"{current_date.strftime('%Y-%m-%d')}: Trailing Stop Hit at {price:.2f}. Stop: {trailing_stop_price:.2f}. Highest: {highest_price_in_position:.2f}")
            elif "Bearish" in regime:
                sell_signal = True
                sell_reason = 'Bearish Regime Detected'
                logger.debug(f"{current_date.strftime('%Y-%m-%d')}: Bearish Regime detected ({regime}). Selling shares.")
            # Optional: Add other exit conditions like Sideways regime
            elif "Sideways" in regime:
                 sell_signal = True # Exit on any sideways? Or only specific sideways?
                 sell_reason = 'Sideways Regime Detected' # Or specific sideways criteria
                 logger.debug(f"{current_date.strftime('%Y-%m-%d')}: Sideways Regime detected ({regime}). Selling shares.")


            if sell_signal:
                trade_value = shares * price
                fee = trade_value * transaction_cost_pct
                cash += trade_value - fee
                # Record the sell trade
                trades.append({
                    'type': 'sell',
                    'date': current_date.strftime('%Y-%m-%d'),
                    'price': price,
                    'shares': shares,
                    'value': trade_value,
                    'fee': fee,
                    'profit': (price - position_entry_price) * shares - fee,
                    'holding_period_days': (current_date - position_entry_date).days,
                    'reason': sell_reason
                })
                logger.info(f"{current_date.strftime('%Y-%m-%d')}: SOLD {shares} @ {price:.2f}. Profit: {trades[-1]['profit']:.2f}")
                shares = 0
                position_entry_price = None # Reset position tracking
                position_entry_date = None
                highest_price_in_position = None

        # --- Check for Buy Signal ---
        # Only consider buying if not currently holding shares and cash is available
        if shares == 0:
            recommendation = get_recommendation(regime, df['Returns'].iloc[i-5:i+1], df['Momentum'].iloc[i-5:i+1] if 'Momentum' in df.columns else pd.Series(), df) # Use last few days data for rec
            buy_signal = recommendation in ['STRONG BUY', 'BUY']

            if buy_signal and cash > price:
                # Determine max shares to buy (e.g., invest a percentage of cash)
                investment_percentage = 0.95 # Invest up to 95% of available cash
                max_shares_possible = (cash * investment_percentage) // price

                if max_shares_possible > 0:
                    buy_shares = max_shares_possible
                    trade_value = buy_shares * price
                    fee = trade_value * transaction_cost_pct

                    # Ensure enough cash after fee
                    if cash >= trade_value + fee:
                        shares = buy_shares
                        cash -= trade_value + fee
                        position_entry_price = price # Record entry details
                        position_entry_date = current_date
                        highest_price_in_position = price # Initialize highest price
                        # Record the buy trade
                        trades.append({
                            'type': 'buy',
                            'date': current_date.strftime('%Y-%m-%d'),
                            'price': price,
                            'shares': shares,
                            'value': trade_value,
                            'fee': fee,
                            'profit': 0, # Profit is calculated on sale
                            'holding_period_days': 0,
                            'reason': recommendation # Store the recommendation that triggered the buy
                        })
                        logger.info(f"{current_date.strftime('%Y-%m-%d')}: BOUGHT {shares} @ {price:.2f} (Signal: {recommendation})")
                    # else: # Not enough cash including fees
                         # logger.debug(f"{current_date.strftime('%Y-%m-%d')}: Insufficient cash to buy {buy_shares} shares at {price:.2f} (Need {trade_value + fee:.2f}, Have {cash:.2f}).")
            # else: # No buy signal
                 # logger.debug(f"{current_date.strftime('%Y-%m-%d')}: No buy signal ({recommendation}) or insufficient cash.")

        # Calculate portfolio value at the end of the day
        current_portfolio_value = cash + shares * price
        portfolio_values.append(current_portfolio_value) # Append value for day 'i'

    # Handle any remaining open position at the end of the backtest period
    if shares > 0:
         final_price = df['Close'].iloc[-1]
         trade_value = shares * final_price
         # Note: No fee calculation for the final 'virtual' close
         trades.append({
             'type': 'hold_end', # Indicate it was an open position at the end
             'date': df.index[-1].strftime('%Y-%m-%d'),
             'price': final_price,
             'shares': shares,
             'value': trade_value,
             'fee': 0,
             'profit': (final_price - position_entry_price) * shares if position_entry_price is not None else 0,
             'holding_period_days': (df.index[-1] - position_entry_date).days if position_entry_date is not None else 0,
             'reason': 'Position open at end of backtest'
         })
         logger.info(f"Backtest ended with open position: {shares} shares.")

    # Calculate performance metrics
    final_value = portfolio_values[-1] if portfolio_values else initial_cash
    backtest_return = (final_value - initial_cash) / initial_cash * 100 if initial_cash != 0 else 0.0

    # Calculate the total number of days in the backtest period
    if len(df) > 1:
        period_days = (df.index[-1] - df.index[0]).days # Total calendar days
        # Attempt to estimate trading days, roughly len(df)
        estimated_trading_days = len(df) - 1 # Subtract initial state/day 0
    else:
        period_days = 0
        estimated_trading_days = 0


    # Annualized return calculation
    # Annualized return = (1 + Total Return)^(252 / Trading Days) - 1
    annualized_return = 0.0
    if initial_cash > 0 and estimated_trading_days > 0:
        total_return_factor = final_value / initial_cash
        if total_return_factor > 0: # Only annualize positive growth factors
             # Use 252 trading days per year as a standard
             annualized_return = (total_return_factor ** (252 / estimated_trading_days) - 1) * 100
        elif total_return_factor == 0:
             annualized_return = -100.0 # Total loss
        else: # Handle cases where value goes below zero (not possible with this strategy)
            # Should not happen with this strategy (cash + stock value >= 0)
             annualized_return = (total_return_factor ** (252 / estimated_trading_days) - 1) * 100 # Will be negative


    # Benchmark return calculation (buy and hold)
    benchmark_final_value = benchmark_portfolio_values[-1] if benchmark_portfolio_values else initial_cash
    benchmark_return = (benchmark_final_value - initial_cash) / initial_cash * 100 if initial_cash != 0 else 0.0

    # Calculate risk metrics using the daily portfolio return series
    # Create a pandas Series for daily portfolio returns
    # The portfolio_values list contains initial_cash + daily_values
    portfolio_series = pd.Series(portfolio_values)
    portfolio_daily_returns = portfolio_series.pct_change().dropna()

    # Create a pandas Series for daily benchmark returns
    benchmark_series = pd.Series(benchmark_portfolio_values, index=df.index) # Index aligns with df
    benchmark_daily_returns = benchmark_series.pct_change().dropna()

    risk_metrics = calculate_risk_metrics(df, portfolio_daily_returns, benchmark_daily_returns, estimated_trading_days)

    # Trade statistics
    buy_trades = [t for t in trades if t['type'] == 'buy']
    sell_trades = [t for t in trades if t['type'] == 'sell']
    num_trades = len(buy_trades) # Number of buy trades equals number of positions initiated
    profitable_sell_trades = [t for t in sell_trades if t['profit'] > 0]
    win_loss_ratio = (len(profitable_sell_trades) / len(sell_trades) * 100) if len(sell_trades) > 0 else (100 if num_trades > 0 and len(sell_trades) == 0 else "No Trades") # If bought but never sold, win/loss N/A

    holding_periods = [t['holding_period_days'] for t in sell_trades]
    avg_holding_period = np.mean(holding_periods) if holding_periods else "No Trades"


    backtest_summary = {
        'Initial Value': initial_cash,
        'Final Value': round(final_value, 2),
        'Return (%)': round(backtest_return, 2),
        'Annualized Return (%)': round(annualized_return, 2),
        'Portfolio': portfolio_values, # Include the series of values for plotting
        'Trades': trades, # Include trade details for debugging/analysis if needed
        'Number of Trades': num_trades,
        'Win/Loss Ratio (%)': round(win_loss_ratio, 2) if isinstance(win_loss_ratio, (int, float)) else win_loss_ratio,
        'Avg Holding Period (days)': round(avg_holding_period, 2) if isinstance(avg_holding_period, (int, float)) else avg_holding_period,
        'Max Drawdown (%)': risk_metrics['Max Drawdown (%)'],
        'Benchmark Return (%)': round(benchmark_return, 2),
        'Trades Executed': num_trades > 0,
        'Sharpe Ratio': risk_metrics['Sharpe Ratio'],
        'Sortino Ratio': risk_metrics['Sortino Ratio'],
        'Calmar Ratio': risk_metrics['Calmar Ratio'],
        'Treynor Ratio': risk_metrics['Treynor Ratio'],
        'Beta': risk_metrics['Beta']
    }

    logger.info(f"Backtest complete. Final Value: {final_value:.2f}, Return: {backtest_return:.2f}%, Annualized: {annualized_return:.2f}%. Trades: {num_trades}")

    return backtest_summary


# backend/screener_api/screener_logic.py

# ... (Your imports and helper functions like convert_numpy_keys_to_int) ...

# backend/screener_api/screener_logic.py

# ... (Your imports and helper functions including convert_numpy_keys_to_int) ...

def screen_stock(symbol: str, period: str, interval: str, feature_sets: List[List[str]], window: int, max_states: int,
                 train_window: int, exchange_suffix: str = ".NS", use_rolling_window: bool = False, slippage: float = 0.001) -> Dict:

    # ... (Initial setup of result_dict, symbol_stripped, ticker, initial variable assignments like data=None, data_hmm=pd.DataFrame(), etc. - This part is correct from the previous version) ...
    result_dict = DEFAULT_RESULT.copy() # Use a temporary dict

    # Clean symbol and add suffix if not present
    symbol_stripped = symbol.strip().upper()
    ticker = symbol_stripped if symbol_stripped.endswith(exchange_suffix) else symbol_stripped + exchange_suffix
    result_dict['Stock'] = symbol_stripped # Store the original symbol without suffix
    logger.info(f"Screening ticker: {ticker}")

    data = None
    data_hmm = pd.DataFrame()
    fetch_error = preprocess_error = hmm_error = None
    converged = False
    ensemble_info = {}
    backtest_result = {}


    try:
        # --- 1. Fetch Data ---
        data, fetch_error = fetch_stock_data(ticker, period, interval)
        # ... (rest of the fetch_data section and subsequent checks for fetch_error, preprocess_error, hmm_error, and successful converged block - This part is correct from the previous version) ...

        if fetch_error is None: # Only proceed if fetch was successful
            # 2. Preprocess Data
            requested_features = [f for sublist in feature_sets for f in sublist]
            data_processed, preprocess_error = preprocess_data(data, requested_features, window)
            # ... (rest of preprocess section) ...

            if preprocess_error is None:
                 if 'Features_Used' not in data_processed.columns or not data_processed['Features_Used'].iloc[0]:
                      preprocess_error = "Preprocessing failed: No valid features available for HMM."
                      logger.error(f"Preprocessing failed for {ticker}: No valid features for HMM.")

            if preprocess_error is None: # Only proceed if features are available
                valid_feature_sets_for_hmm = []
                processed_features_available = data_processed['Features_Used'].iloc[0]
                for f_set in feature_sets:
                    if all(f in processed_features_available for f in f_set):
                         valid_feature_sets_for_hmm.append(f_set)
                    else:
                         logger.warning(f"Skipping feature set {f_set} for HMM as some features were not available/valid after preprocessing.")

                if not valid_feature_sets_for_hmm:
                     hmm_error = "No valid feature sets remaining after preprocessing for HMM."
                     logger.error(f"No valid feature sets remaining for {ticker} after preprocessing.")

            if hmm_error is None: # Only proceed if valid feature sets exist
                 # 3. Fit HMM Ensemble
                 data_hmm, converged, hmm_error_fit, ensemble_info = fit_hmm_ensemble( # Use specific error name
                     df=data_processed,
                     feature_sets=valid_feature_sets_for_hmm,
                     max_states=max_states,
                     train_window=train_window,
                     use_rolling_window=use_rolling_window
                 )

                 # If initial fit failed, retry with minimum states (2)
                 if not converged and hmm_error_fit: # Only retry if it failed and has an error message (indicating a fit issue)
                      logger.info(f"Initial HMM fit failed for {ticker}, retrying with max_states=2.")
                      data_hmm, converged, hmm_error_retry, ensemble_info_retry = fit_hmm_ensemble(
                          df=data_processed,
                          feature_sets=valid_feature_sets_for_hmm,
                          max_states=2,
                          train_window=train_window,
                          use_rolling_window=use_rolling_window
                      )
                      if not converged:
                          hmm_error = hmm_error_retry # If retry also failed, keep the retry error message
                          logger.error(f"HMM fitting failed for {ticker} even after retry: {hmm_error}")
                      else:
                           # If retry succeeded, update ensemble_info and clear hmm_error for this path
                           hmm_error = None # Clear the error since retry succeeded
                           ensemble_info = ensemble_info_retry # Use info from the successful retry


        # --- After potential failures or successful HMM ---
        # Only run backtest and get recommendation if HMM converged and data_hmm is valid
        if converged and not data_hmm.empty:
            latest_regime = data_hmm['Regime'].iloc[-1]

            # Get stats for the *latest* regime from the *full* dataset assigned to that regime
            latest_regime_df = data_hmm[data_hmm['Regime'] == latest_regime]
            mean_daily_return_in_regime = latest_regime_df['Returns'].mean() if 'Returns' in latest_regime_df.columns and not latest_regime_df.empty else 0.0
            # Annualize mean daily return (approximate)
            mean_annualized_return_in_regime = mean_daily_return_in_regime * 252 * 100 if not np.isnan(mean_daily_return_in_regime) else 0.0

            # Use median momentum for the latest regime as it's more robust
            momentum_in_regime = latest_regime_df['Momentum'].median() if 'Momentum' in latest_regime_df.columns and not latest_regime_df['Momentum'].empty else 0.0


            # 4. Get Recommendation (based on latest regime and recent data characteristics)
            recent_data_window = 5 # Look at the last 5 days of data
            recent_returns = data_hmm['Returns'].iloc[-min(recent_data_window, len(data_hmm)):].dropna() if 'Returns' in data_hmm.columns else pd.Series()
            recent_momentum = data_hmm['Momentum'].iloc[-min(recent_data_window, len(data_hmm)):].dropna() if 'Momentum' in data_hmm.columns else pd.Series()

            recommendation = get_recommendation(latest_regime, recent_returns, recent_momentum, data_hmm) # Pass full df for quantiles


            # 5. Run Backtest
            # Check if backtest_strategy might return a DEFAULT_RESULT with error
            backtest_result = backtest_strategy(data_hmm, slippage=slippage)


            # --- Populate result_dict for success ---
            data_columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume', 'Regime', 'PriceMA', 'VolumeMA',
                                    'State', 'State_Confidence', 'Regime_Change_Signal', 'Returns']
            # Add calculated features if they exist in the final DataFrame
            all_possible_features = DEFAULT_CONFIG.get('available_features', [])
            calculated_features_in_df = [col for col in data_hmm.columns if col in all_possible_features and col not in data_columns_to_keep] # Add only calculated features not already listed
            data_columns_to_keep.extend(calculated_features_in_df)
            data_columns_to_keep = list(set([col for col in data_columns_to_keep if col in data_hmm.columns])) # Ensure uniqueness and existence


            result_dict.update({
                # Stock is already set at the beginning
                'Latest Regime': latest_regime,
                'Mean Annualized Return (%)': round(mean_annualized_return_in_regime, 2),
                'Recommendation': recommendation,
                'Converged': True, # Successfully converged after potential retry
                'Error': None, # Clear any potential previous error message
                # Convert DataFrame to JSON-friendly dictionary format
                'Data': data_hmm[data_columns_to_keep].to_dict(orient='split') if not data_hmm.empty and data_columns_to_keep else None,
                # Use .get with defaults for safety in case backtest_strategy returned incomplete results on error
                'Initial Portfolio Value': backtest_result.get('Initial Value', 0.0),
                'Final Portfolio Value': backtest_result.get('Final Value', 0.0),
                'Backtest Return (%)': backtest_result.get('Return (%)', 0.0),
                'Annualized Backtest Return (%)': backtest_result.get('Annualized Return (%)', 0.0),
                'Max Drawdown (%)': backtest_result.get('Max Drawdown (%)', 0.0),
                'Sharpe Ratio': backtest_result.get('Sharpe Ratio', 0.0),
                'Sortino Ratio': backtest_result.get('Sortino Ratio', 0.0),
                'Calmar Ratio': backtest_result.get('Calmar Ratio', 0.0),
                'Treynor Ratio': backtest_result.get('Treynor Ratio', 0.0),
                'Beta': backtest_result.get('Beta', 0.0),
                'Number of Trades': backtest_result.get('Number of Trades', 0),
                'Win/Loss Ratio (%)': backtest_result.get('Win/Loss Ratio (%)', "No Trades"),
                'Avg Holding Period (days)': backtest_result.get('Avg Holding Period (days)', "No Trades"),
                'Benchmark Return (%)': backtest_result.get('Benchmark Return (%)', 0.0),
                'Trades Executed': backtest_result.get('Trades Executed', False),
                'Feature_Weights': ensemble_info.get('weights', {}),
                'State_Characteristics': ensemble_info.get('state_characteristics', {}),
                 # Include portfolio values list
                'Portfolio_Values': backtest_result.get('Portfolio', [])
            })
            logger.info(f"Screening completed successfully for {ticker}. Regime: {result_dict['Latest Regime']}, Rec: {result_dict['Recommendation']}")

        else: # HMM did not converge or data_hmm is empty after all attempts
            result_dict['Converged'] = False
            result_dict['Recommendation'] = 'INCONCLUSIVE' # Default to inconclusive
            # Set error message based on which step failed first
            if fetch_error:
                result_dict['Error'] = f"Data fetch failed: {fetch_error}"
                result_dict['Recommendation'] = 'ERROR' # More severe error
            elif preprocess_error:
                result_dict['Error'] = f"Preprocessing failed: {preprocess_error}"
            elif hmm_error: # This is the error message from fit_hmm_ensemble if it failed
                 result_dict['Error'] = f"HMM fitting failed: {hmm_error}"
            else:
                 result_dict['Error'] = "HMM did not converge or data_hmm is empty for an unknown reason." # Fallback error

            logger.warning(f"Screening inconclusive for {ticker}. Converged={converged}, Error='{result_dict['Error']}'")

            # Try to include partial data if preprocessing was at least successful and data_hmm has some columns
            if not data_hmm.empty and not data_hmm.columns.empty:
                 try:
                     minimal_cols_for_partial = ['Open', 'High', 'Low', 'Close', 'Volume', 'Regime', 'State', 'State_Confidence', 'Regime_Change_Signal', 'Returns']
                     # Add any calculated features that might exist in the partial df
                     all_possible_features = DEFAULT_CONFIG.get('available_features', [])
                     calculated_cols_partial = [col for col in data_hmm.columns if col in all_possible_features and col not in minimal_cols_for_partial]
                     cols_to_include_partial = list(set([c for c in minimal_cols_for_partial + calculated_cols_partial if c in data_hmm.columns]))

                     if cols_to_include_partial:
                          # !!! Use result_dict['Data'] = ... here
                          result_dict['Data'] = data_hmm[cols_to_include_partial].to_dict(orient='split')
                     else:
                         result_dict['Data'] = None
                 except Exception as e_partial:
                     logger.error(f"Failed to convert partial data_hmm to dict for {ticker} in inconclusive path: {e_partial}", exc_info=True)
                     result_dict['Data'] = None # Ensure Data is None on error

            else:
                 result_dict['Data'] = None # No partial data to include

            # Also set backtest/metrics to default values if HMM didn't converge
            result_dict.update({
                'Initial Portfolio Value': 0.0, 'Final Portfolio Value': 0.0,
                'Backtest Return (%)': 0.0, 'Annualized Backtest Return (%)': 0.0,
                'Max Drawdown (%)': 0.0, 'Sharpe Ratio': 0.0, 'Sortino Ratio': 0.0,
                'Calmar Ratio': 0.0, 'Treynor Ratio': 0.0, 'Beta': 0.0,
                'Number of Trades': 0, 'Win/Loss Ratio (%)': "No Trades",
                'Avg Holding Period (days)': "No Trades", 'Benchmark Return (%)': 0.0,
                'Trades Executed': False, 'Feature_Weights': {}, 'State_Characteristics': {},
                'Portfolio_Values': []
            })


    except Exception as e:
        # This catch block handles *any* unexpected errors that occur during the screening process for a single stock
        error_msg = f"An unexpected error occurred during screening for {ticker}: {e}"
        logger.error(error_msg, exc_info=True)

        # Populate result_dict for unexpected errors
        # Use update to ensure Stock is still included
        result_dict.update({
            'Error': error_msg,
            'Recommendation': 'ERROR_UNEXPECTED', # Differentiate from known errors
            'Converged': False,
            'Latest Regime': 'N/A', # Reset defaults
            'Mean Annualized Return (%)': 0.0,
            # Set all backtest/metric fields to default values on unexpected error
            'Initial Portfolio Value': 0.0, 'Final Portfolio Value': 0.0,
            'Backtest Return (%)': 0.0, 'Annualized Backtest Return (%)': 0.0,
            'Max Drawdown (%)': 0.0, 'Sharpe Ratio': 0.0, 'Sortino Ratio': 0.0,
            'Calmar Ratio': 0.0, 'Treynor Ratio': 0.0, 'Beta': 0.0,
            'Number of Trades': 0, 'Win/Loss Ratio (%)': "No Trades",
            'Avg Holding Period (days)': "No Trades", 'Benchmark Return (%)': 0.0,
            'Trades Executed': False, 'Feature_Weights': {}, 'State_Characteristics': {},
            'Portfolio_Values': []
        })

        # Try to include any partial data_hmm if it was created before the error
        # Use the data_hmm variable if it was assigned before the error occurred
        if isinstance(data_hmm, pd.DataFrame) and not data_hmm.empty and not data_hmm.columns.empty:
             try:
                # Select a minimal set of columns for inclusion in the error result
                minimal_cols_for_unexpected_error = ['Open', 'High', 'Low', 'Close', 'Volume', 'Regime', 'State']
                cols_to_include_unexpected = list(set([c for c in minimal_cols_for_unexpected_error if c in data_hmm.columns]))
                if cols_to_include_unexpected:
                     # !!! Use result_dict['Data'] = ... here
                     result_dict['Data'] = data_hmm[cols_to_include_unexpected].to_dict(orient='split')
             except Exception as inner_e:
                 logger.error(f"Failed to convert partial data_hmm to dict for {ticker} in unexpected error path: {inner_e}", exc_info=True)
                 result_dict['Data'] = None # Ensure Data is None on inner error
        else:
             result_dict['Data'] = None # No partial data to include


    # --- FINAL STEP: Clean the dictionary before returning ---
    # This ensures that even if errors occurred and partial data was included,
    # any numpy types used as keys or values are converted.
    final_cleaned_result = convert_numpy_keys_to_int(result_dict)

    # logger.debug(f"Final result structure for {ticker} before return: {final_cleaned_result}") # Debugging line
    return final_cleaned_result

# ... (Your screen_stocks function remains the same) ...

# ... (Your screen_stocks function remains the same) ...


def screen_stocks(symbols: List[str], period: str, interval: str, feature_sets: List[List[str]], window: int, max_states: int,
                  train_window: int, exchange_suffix: str, max_workers: int, use_rolling_window: bool, slippage: float) -> List[Dict]:
    """
    Screens a batch of stocks in parallel using ThreadPoolExecutor.

    Args:
        symbols: List of stock symbols (without suffix).
        period: Data period.
        interval: Data interval.
        feature_sets: List of feature sets for HMM ensemble.
        window: Rolling window size.
        max_states: Max HMM states.
        train_window: Days for HMM training.
        exchange_suffix: Exchange suffix.
        max_workers: Number of parallel workers.
        use_rolling_window: Use rolling window for training.
        slippage: Slippage for backtesting.

    Returns:
        List of result dictionaries for each stock.
    """
    results = []
    total = len(symbols)
    if total == 0:
        return []

    logger.info(f"Starting screening for {total} stocks with {max_workers} workers.")

    # Adjusted timeout - give each stock a minimum time slice, but cap the total.
    # A base time per stock + overhead.
    base_timeout_per_stock = 10 # seconds per stock
    max_total_timeout = 300 # seconds (5 minutes) - adjust based on expected run time
    timeout = max(base_timeout_per_stock, min(max_total_timeout, base_timeout_per_stock * total / max_workers + 30)) # Add buffer


    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_symbol = {
            executor.submit(
                screen_stock, symbol, period, interval, feature_sets, window, max_states, train_window,
                exchange_suffix, use_rolling_window, slippage
            ): symbol for symbol in symbols
        }

        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                # Retrieve the result from the future with a timeout
                result = future.result(timeout=timeout)
                results.append(result)
                logger.info(f"Finished processing {symbol}")
            except Exception as e:
                # Handle exceptions that occur during processing or due to timeout
                logger.error(f"Error processing {symbol} (Timeout or Exception): {e}")
                result = DEFAULT_RESULT.copy()
                result.update({'Stock': symbol, 'Error': str(e), 'Recommendation': 'ERROR'}) # Mark as ERROR
                results.append(result)

    logger.info("Batch screening completed.")
    return results

def validate_symbols(symbols: List[str], exchange_suffix: str, period: str, interval: str) -> Tuple[List[str], Dict[str, str]]:
    """
    Validates a list of stock symbols by attempting to fetch minimal data.

    Args:
        symbols: List of stock symbols (without suffix).
        exchange_suffix: Suffix for the exchange.
        period: Data period to attempt fetching.
        interval: Data interval to attempt fetching.

    Returns:
        Tuple of (list of valid symbols, dictionary of invalid symbol: error message).
    """
    valid_symbols = []
    invalid_symbols = {}
    min_validation_points = 10 # Just need a few points to check validity

    logger.info(f"Validating {len(symbols)} symbols...")

    # Use ThreadPoolExecutor for parallel validation fetches
    max_validation_workers = min(20, len(symbols), multiprocessing.cpu_count() * 2) # More workers for validation as it's faster
    validation_timeout_per_stock = 10 # seconds

    with ThreadPoolExecutor(max_workers=max_validation_workers) as executor:
        future_to_symbol = {}
        for symbol in symbols:
            symbol_stripped = symbol.strip().upper()
            if not symbol_stripped: continue
            ticker_with_suffix = symbol_stripped if symbol_stripped.endswith(exchange_suffix) else symbol_stripped + exchange_suffix
            # Use the cached fetch_stock_data, but with minimal requirements
            future = executor.submit(fetch_stock_data, ticker_with_suffix.split('.')[0], period, interval, min_validation_points, max_retries=1) # Short retry
            future_to_symbol[future] = symbol_stripped

        for i, future in enumerate(as_completed(future_to_symbol)):
             symbol_stripped = future_to_symbol[future]
             try:
                  # Use a timeout for each validation fetch
                  df, error = future.result(timeout=validation_timeout_per_stock)
                  if df is not None:
                      valid_symbols.append(symbol_stripped)
                      logger.debug(f"Validation successful for {symbol_stripped}")
                  else:
                      invalid_symbols[symbol_stripped] = error or "Fetch failed."
                      logger.warning(f"Validation failed for {symbol_stripped}: {error}")
             except Exception as e:
                  invalid_symbols[symbol_stripped] = str(e) # Capture exception message
                  logger.error(f"Validation failed for {symbol_stripped} (Exception): {e}")

    logger.info(f"Validation complete. Valid: {len(valid_symbols)}, Invalid: {len(invalid_symbols)}")
    return valid_symbols, invalid_symbols

def get_indices_data() -> Dict[str, Tuple[List[str], str]]:
    """Returns the predefined index data."""
    return INDICES

def get_config() -> Dict[str, Any]:
    """Returns the current loaded configuration."""
    return CONFIG

def save_config(new_config: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Saves the current configuration to config.json."""
    try:
        # Only save values that are in the default config to avoid saving arbitrary data
        config_to_save = {k: new_config.get(k, DEFAULT_CONFIG[k]) for k in DEFAULT_CONFIG if k in new_config}
        # Ensure features list contains only strings
        if 'features' in config_to_save and isinstance(config_to_save['features'], list):
            config_to_save['features'] = [str(f) for f in config_to_save['features']]
        # Ensure available_features list contains only strings
        if 'available_features' in config_to_save and isinstance(config_to_save['available_features'], list):
             config_to_save['available_features'] = [str(f) for f in config_to_save['available_features']]


        with open(CONFIG_FILE, "w") as f:
            json.dump(config_to_save, f, indent=4)
        # Update the in-memory CONFIG
        CONFIG.update(config_to_save)
        logger.info(f"Configuration saved to {CONFIG_FILE}")
        return True, None
    except Exception as e:
        error_msg = f"Error saving configuration to {CONFIG_FILE}: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg

def get_logs(tail_lines: int = 500) -> str:
    """Reads the last lines of the log file."""
    try:
        if not os.path.exists("screener.log"):
            return "Log file not found."
        with open("screener.log", "r", encoding='utf-8') as f:
            # Read all lines and get the last `tail_lines`
            lines = f.readlines()
            return "".join(lines[-tail_lines:])
    except Exception as e:
        return f"Error reading log file: {e}"

def clear_data_cache() -> Tuple[bool, Optional[str]]:
    """Removes all cached data files."""
    try:
        if os.path.exists(DATA_CACHE_DIR):
            for filename in os.listdir(DATA_CACHE_DIR):
                file_path = os.path.join(DATA_CACHE_DIR, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            os.rmdir(DATA_CACHE_DIR) # Attempt to remove the directory if empty
        os.makedirs(DATA_CACHE_DIR, exist_ok=True) # Recreate the directory
        logger.info("Data cache cleared.")
        return True, None
    except Exception as e:
        error_msg = f"Error clearing data cache: {e}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg

if __name__ == '__main__':
    # Example usage if you want to run this script directly for testing
    # Note: This won't use the Django environment or full config setup
    print("Running screener_logic.py as main (for testing)...")

    # Example parameters
    test_symbols = ['RELIANCE', 'TCS', 'SBIN', 'INFY'] # Example Nifty stocks
    test_period = '1y'
    test_interval = '1d'
    test_feature_sets = [['Returns', 'Momentum'], ['Returns', 'Volatility'], ['Returns', 'RSI', 'ATR']]
    test_window = 3
    test_max_states = 3
    test_train_window = 126 # Shorter for testing
    test_exchange = '.NS'
    test_workers = 2
    test_rolling_window = False
    test_slippage = 0.001

    print(f"Testing with symbols: {test_symbols}")

    # Ensure DATA_CACHE_DIR exists relative to this script if run directly
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Run validation first
    print("\nValidating symbols...")
    valid, invalid = validate_symbols(test_symbols, test_exchange, test_period, test_interval)
    print(f"Valid symbols: {valid}")
    print(f"Invalid symbols: {invalid}")

    if valid:
        # Run screening
        print("\nRunning screening...")
        screening_results = screen_stocks(
            valid, test_period, test_interval, test_feature_sets, test_window,
            test_max_states, test_train_window, test_exchange, test_workers,
            test_rolling_window, test_slippage
        )

        print("\n--- Screening Results ---")
        for res in screening_results:
            print(f"Stock: {res['Stock']}")
            print(f"  Regime: {res['Latest Regime']}")
            print(f"  Recommendation: {res['Recommendation']}")
            print(f"  Converged: {res['Converged']}")
            print(f"  Error: {res['Error']}")
            if res.get('Trades Executed', False):
                 print(f"  Backtest Return (%): {res['Backtest Return (%)']}")
                 print(f"  Annualized Return (%): {res['Annualized Backtest Return (%)']}")
                 print(f"  Max Drawdown (%): {res['Max Drawdown (%)']}")
                 print(f"  Sharpe Ratio: {res['Sharpe Ratio']}")
                 print(f"  Calmar Ratio: {res['Calmar Ratio']}")
                 print(f"  Number of Trades: {res['Number of Trades']}")
                 print(f"  Avg Holding Period: {res['Avg Holding Period (days)']}")
            print("-" * 20)

        # Example of accessing Data (DataFrame converted to dict)
        first_result_data = next((res['Data'] for res in screening_results if res['Data'] is not None), None)
        if first_result_data:
             print("\nExample of Data structure (converted DataFrame):")
             # To convert back to DataFrame: pd.DataFrame(**first_result_data)
             print(list(first_result_data.keys())) # e.g., ['index', 'columns', 'data']
             print("Index example:", first_result_data['index'][:5])
             print("Columns example:", first_result_data['columns'])
             print("Data shape example:", np.array(first_result_data['data']).shape)

    print("\nTesting logs retrieval...")
    logs_content = get_logs(tail_lines=50)
    print("Last 50 lines of logs:")
    print(logs_content)

    print("\nTesting cache clearing...")
    success, msg = clear_data_cache()
    print(f"Cache cleared: {success}, Message: {msg}")