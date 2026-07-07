#!/usr/bin/env python3
"""
Falcon AI Ultimate - Professional Trading Signal Bot
====================================================
Ensemble ML-powered trading signal generator with Telegram integration.
Supports Forex, Crypto, and Gold markets with multi-timeframe analysis.

Author: Falcon AI System
Version: 1.0.0
License: Proprietary
"""

import os
import sys
import time
import logging
import sqlite3
import warnings
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from matplotlib.patches import FancyBboxPatch

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import joblib

import telebot
from telebot import types

# Suppress warnings for cleaner logs
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Central configuration for the trading bot."""
    
    # Telegram settings
    TELEGRAM_TOKEN: str = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
    TELEGRAM_CHAT_ID: str = '7553333305'
    
    # Trading parameters
    TRADE_DURATION_MINUTES: int = 10
    TIMEFRAMES: List[str] = field(default_factory=lambda: ['5m', '15m'])
    
    # Symbols to monitor
    SYMBOLS: Dict[str, List[str]] = field(default_factory=lambda: {
        'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X'],
        'crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'BNB-USD'],
        'metals': ['GC=F', 'XAUUSD=X', 'SI=F']
    })
    
    # ML settings
    CONFIDENCE_THRESHOLD: float = 0.65
    RETRAINING_INTERVAL_HOURS: int = 24
    MIN_TRAINING_SAMPLES: int = 500
    TEST_SIZE: float = 0.2
    VALIDATION_SIZE: float = 0.15
    
    # Database
    DB_PATH: str = 'falcon_trading.db'
    MODELS_DIR: str = 'models'
    
    # Chart settings
    CHART_CANDLES: int = 50
    CHART_DPI: int = 150
    CHART_FIGSIZE: Tuple[int, int] = (14, 8)
    
    # Performance
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5
    SCAN_INTERVAL_MINUTES: int = 5
    
    # Paths
    LOG_FILE: str = 'falcon_bot.log'
    SIGNAL_COOLDOWN_MINUTES: int = 10  # Prevent duplicate signals

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Configure professional logging system."""
    logger = logging.getLogger('FalconAI')
    logger.setLevel(logging.INFO)
    
    # File handler with rotation
    fh = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """Professional SQLite database manager for trading records."""
    
    def __init__(self, db_path: str, logger: logging.Logger):
        """
        Initialize database connection and create tables.
        
        Args:
            db_path: Path to SQLite database file
            logger: Logger instance
        """
        self.db_path = db_path
        self.logger = logger
        self._init_database()
    
    def _init_database(self) -> None:
        """Create database tables if they don't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Signals table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        signal_type TEXT NOT NULL,
                        direction TEXT NOT NULL,
                        entry_price REAL NOT NULL,
                        confidence REAL NOT NULL,
                        m5_analysis TEXT,
                        m15_analysis TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        expiry_time DATETIME,
                        result TEXT DEFAULT 'PENDING'
                    )
                ''')
                
                # Performance metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        total_signals INTEGER DEFAULT 0,
                        wins INTEGER DEFAULT 0,
                        losses INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0.0,
                        avg_confidence REAL DEFAULT 0.0,
                        best_symbol TEXT,
                        worst_symbol TEXT,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Model training log
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS training_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        accuracy REAL,
                        precision REAL,
                        recall REAL,
                        f1_score REAL,
                        training_time REAL,
                        samples_count INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Initialize performance row if empty
                cursor.execute('SELECT COUNT(*) FROM performance')
                if cursor.fetchone()[0] == 0:
                    cursor.execute('''
                        INSERT INTO performance (total_signals, wins, losses, win_rate, avg_confidence)
                        VALUES (0, 0, 0, 0.0, 0.0)
                    ''')
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}", exc_info=True)
            raise
    
    def save_signal(self, signal_data: Dict[str, Any]) -> int:
        """
        Save trading signal to database.
        
        Args:
            signal_data: Dictionary containing signal information
            
        Returns:
            Signal ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO signals 
                    (symbol, signal_type, direction, entry_price, confidence, 
                     m5_analysis, m15_analysis, expiry_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['symbol'],
                    signal_data['signal_type'],
                    signal_data['direction'],
                    signal_data['entry_price'],
                    signal_data['confidence'],
                    signal_data['m5_analysis'],
                    signal_data['m15_analysis'],
                    signal_data['expiry_time']
                ))
                signal_id = cursor.lastrowid
                conn.commit()
                
                # Update total signals count
                cursor.execute('''
                    UPDATE performance SET 
                    total_signals = total_signals + 1,
                    updated_at = CURRENT_TIMESTAMP
                ''')
                conn.commit()
                
                self.logger.info(f"Signal saved to DB | ID: {signal_id} | {signal_data['symbol']} {signal_data['direction']}")
                return signal_id
                
        except Exception as e:
            self.logger.error(f"Failed to save signal: {e}", exc_info=True)
            return -1
    
    def update_signal_result(self, signal_id: int, result: str, exit_price: float) -> None:
        """
        Update signal result after trade completion.
        
        Args:
            signal_id: Signal ID to update
            result: Trade result ('WIN' or 'LOSS')
            exit_price: Price at trade closure
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE signals 
                    SET result = ?, exit_price = ?
                    WHERE id = ?
                ''', (result, exit_price, signal_id))
                
                # Update performance counters
                if result == 'WIN':
                    cursor.execute('''
                        UPDATE performance SET 
                        wins = wins + 1,
                        updated_at = CURRENT_TIMESTAMP
                    ''')
                elif result == 'LOSS':
                    cursor.execute('''
                        UPDATE performance SET 
                        losses = losses + 1,
                        updated_at = CURRENT_TIMESTAMP
                    ''')
                
                conn.commit()
                self.logger.info(f"Signal {signal_id} result updated: {result} at {exit_price}")
                
        except Exception as e:
            self.logger.error(f"Failed to update signal result: {e}", exc_info=True)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Calculate and return performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT * FROM performance ORDER BY id DESC LIMIT 1')
                row = cursor.fetchone()
                
                if row:
                    stats = {
                        'total_signals': row[1],
                        'wins': row[2],
                        'losses': row[3],
                        'win_rate': row[4],
                        'avg_confidence': row[5]
                    }
                    
                    # Get best/worst symbols
                    cursor.execute('''
                        SELECT symbol, 
                               SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins_count,
                               COUNT(*) as total_count
                        FROM signals 
                        WHERE result != 'PENDING'
                        GROUP BY symbol
                        HAVING total_count >= 3
                    ''')
                    
                    symbol_stats = cursor.fetchall()
                    if symbol_stats:
                        best = max(symbol_stats, key=lambda x: x[1]/x[2] if x[2] > 0 else 0)
                        worst = min(symbol_stats, key=lambda x: x[1]/x[2] if x[2] > 0 else 0)
                        stats['best_symbol'] = f"{best[0]} ({best[1]}/{best[2]})"
                        stats['worst_symbol'] = f"{worst[0]} ({worst[1]}/{worst[2]})"
                    
                    return stats
                return {}
                
        except Exception as e:
            self.logger.error(f"Failed to get performance stats: {e}", exc_info=True)
            return {}
    
    def check_signal_exists(self, symbol: str, direction: str, minutes: int) -> bool:
        """
        Check if similar signal was sent recently.
        
        Args:
            symbol: Trading symbol
            direction: Signal direction
            minutes: Cooldown period in minutes
            
        Returns:
            True if duplicate signal exists
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cutoff_time = datetime.now() - timedelta(minutes=minutes)
                
                cursor.execute('''
                    SELECT COUNT(*) FROM signals
                    WHERE symbol = ? 
                    AND direction = ?
                    AND timestamp > ?
                    AND result = 'PENDING'
                ''', (symbol, direction, cutoff_time))
                
                count = cursor.fetchone()[0]
                return count > 0
                
        except Exception as e:
            self.logger.error(f"Failed to check signal existence: {e}", exc_info=True)
            return False
    
    def get_pending_signals(self) -> List[Dict[str, Any]]:
        """
        Get all pending signals that need result checking.
        
        Returns:
            List of pending signals
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM signals 
                    WHERE result = 'PENDING' 
                    AND expiry_time <= datetime('now', 'localtime')
                ''')
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            self.logger.error(f"Failed to get pending signals: {e}", exc_info=True)
            return []
    
    def save_training_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Save model training metrics.
        
        Args:
            metrics: Dictionary with training metrics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO training_log 
                    (model_name, accuracy, precision, recall, f1_score, training_time, samples_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics['model_name'],
                    metrics['accuracy'],
                    metrics['precision'],
                    metrics['recall'],
                    metrics['f1_score'],
                    metrics['training_time'],
                    metrics['samples_count']
                ))
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to save training metrics: {e}", exc_info=True)

# ============================================================================
# TECHNICAL INDICATORS ENGINE
# ============================================================================

class TechnicalIndicators:
    """Comprehensive technical indicators calculator."""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Price series
            period: RSI period
            
        Returns:
            RSI values series
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """
        Calculate MACD indicator.
        
        Returns:
            Dictionary with MACD line, signal line, and histogram
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        Args:
            df: DataFrame with High, Low, Close columns
            period: ATR period
            
        Returns:
            ATR values series
        """
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """
        Calculate Average Directional Index (ADX).
        
        Returns:
            Dictionary with ADX, +DI, -DI
        """
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = -minus_dm
        
        tr = TechnicalIndicators.calculate_atr(df, period) * period
        
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / tr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / tr)
        
        dx = 100 * np.abs((plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Returns:
            Dictionary with upper, middle, lower bands, and %B
        """
        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        percent_b = (prices - lower) / (upper - lower)
        bandwidth = (upper - lower) / middle
        
        return {
            'upper': upper,
            'middle': middle,
            'lower': lower,
            'percent_b': percent_b,
            'bandwidth': bandwidth
        }
    
    @staticmethod
    def calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Returns:
            Dictionary with %K and %D
        """
        low_min = df['Low'].rolling(window=k_period).min()
        high_max = df['High'].rolling(window=k_period).max()
        
        k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
        d = k.rolling(window=d_period).mean()
        
        return {'k': k, 'd': d}
    
    @staticmethod
    def calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            df: DataFrame with High, Low, Close columns
            period: CCI period
            
        Returns:
            CCI values series
        """
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (typical_price - sma) / (0.015 * mad)
        return cci
    
    @staticmethod
    def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Williams %R.
        
        Returns:
            Williams %R values series
        """
        high_max = df['High'].rolling(window=period).max()
        low_min = df['Low'].rolling(window=period).min()
        
        wr = -100 * ((high_max - df['Close']) / (high_max - low_min))
        return wr
    
    @staticmethod
    def calculate_donchian_channels(df: pd.DataFrame, period: int = 20) -> Dict[str, pd.Series]:
        """
        Calculate Donchian Channels.
        
        Returns:
            Dictionary with upper, middle, lower channels
        """
        upper = df['High'].rolling(window=period).max()
        lower = df['Low'].rolling(window=period).min()
        middle = (upper + lower) / 2
        
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    @staticmethod
    def calculate_keltner_channels(df: pd.DataFrame, period: int = 20, atr_period: int = 10, multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calculate Keltner Channels.
        
        Returns:
            Dictionary with upper, middle, lower channels
        """
        middle = df['Close'].ewm(span=period, adjust=False).mean()
        atr = TechnicalIndicators.calculate_atr(df, atr_period)
        
        upper = middle + (multiplier * atr)
        lower = middle - (multiplier * atr)
        
        return {'upper': upper, 'middle': middle, 'lower': lower}
    
    @staticmethod
    def calculate_pivot_points(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate Pivot Points.
        
        Returns:
            Dictionary with pivot, resistance and support levels
        """
        pivot = (df['High'] + df['Low'] + df['Close']) / 3
        r1 = 2 * pivot - df['Low']
        s1 = 2 * pivot - df['High']
        r2 = pivot + (df['High'] - df['Low'])
        s2 = pivot - (df['High'] - df['Low'])
        r3 = df['High'] + 2 * (pivot - df['Low'])
        s3 = df['Low'] - 2 * (df['High'] - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    @staticmethod
    def calculate_roc(prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Rate of Change (ROC).
        
        Returns:
            ROC values series
        """
        roc = ((prices - prices.shift(period)) / prices.shift(period)) * 100
        return roc
    
    @staticmethod
    def calculate_momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """
        Calculate Momentum.
        
        Returns:
            Momentum values series
        """
        momentum = prices - prices.shift(period)
        return momentum
    
    @staticmethod
    def calculate_trend_strength(prices: pd.Series, period: int = 20) -> pd.Series:
        """
        Calculate Trend Strength using linear regression slope.
        
        Returns:
            Trend strength values series
        """
        def calc_slope(x):
            if len(x) < 2:
                return 0
            return np.polyfit(range(len(x)), x, 1)[0]
        
        trend = prices.rolling(window=period).apply(calc_slope, raw=True)
        return trend
    
    @staticmethod
    def calculate_volatility_ratio(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Volatility Ratio.
        
        Returns:
            Volatility ratio values series
        """
        current_volatility = df['Close'].pct_change().rolling(window=period).std()
        historical_volatility = df['Close'].pct_change().rolling(window=period * 2).std()
        
        ratio = current_volatility / historical_volatility
        return ratio
    
    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Returns:
            VWAP values series
        """
        if 'Volume' not in df.columns or df['Volume'].sum() == 0:
            # Fallback if volume data is unavailable
            return (df['High'] + df['Low'] + df['Close']) / 3
        
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
        return vwap
    
    @staticmethod
    def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators and features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical features
        """
        features = pd.DataFrame(index=df.index)
        
        # Price-based features
        features['returns'] = df['Close'].pct_change()
        features['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        features['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        features['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            if len(df) >= period:
                features[f'sma_{period}'] = df['Close'].rolling(window=period).mean()
                features[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
                features[f'price_to_sma_{period}'] = df['Close'] / features[f'sma_{period}'] - 1
        
        # RSI
        features['rsi_14'] = TechnicalIndicators.calculate_rsi(df['Close'], 14)
        
        # MACD
        macd_data = TechnicalIndicators.calculate_macd(df['Close'])
        features['macd'] = macd_data['macd']
        features['macd_signal'] = macd_data['signal']
        features['macd_histogram'] = macd_data['histogram']
        
        # ATR
        features['atr_14'] = TechnicalIndicators.calculate_atr(df, 14)
        features['atr_percent'] = features['atr_14'] / df['Close'] * 100
        
        # ADX
        adx_data = TechnicalIndicators.calculate_adx(df, 14)
        features['adx'] = adx_data['adx']
        features['plus_di'] = adx_data['plus_di']
        features['minus_di'] = adx_data['minus_di']
        
        # Bollinger Bands
        bb_data = TechnicalIndicators.calculate_bollinger_bands(df['Close'])
        features['bb_percent_b'] = bb_data['percent_b']
        features['bb_bandwidth'] = bb_data['bandwidth']
        
        # Stochastic
        stoch_data = TechnicalIndicators.calculate_stochastic(df)
        features['stoch_k'] = stoch_data['k']
        features['stoch_d'] = stoch_data['d']
        
        # CCI
        features['cci_20'] = TechnicalIndicators.calculate_cci(df, 20)
        
        # Williams %R
        features['williams_r'] = TechnicalIndicators.calculate_williams_r(df, 14)
        
        # Donchian Channels
        dc_data = TechnicalIndicators.calculate_donchian_channels(df)
        features['donchian_position'] = (df['Close'] - dc_data['lower']) / (dc_data['upper'] - dc_data['lower'])
        
        # Keltner Channels
        kc_data = TechnicalIndicators.calculate_keltner_channels(df)
        features['keltner_position'] = (df['Close'] - kc_data['lower']) / (kc_data['upper'] - kc_data['lower'])
        
        # Pivot Points
        pivot_data = TechnicalIndicators.calculate_pivot_points(df)
        features['pivot_distance'] = (df['Close'] - pivot_data['pivot']) / df['Close']
        
        # ROC and Momentum
        features['roc_10'] = TechnicalIndicators.calculate_roc(df['Close'], 10)
        features['momentum_10'] = TechnicalIndicators.calculate_momentum(df['Close'], 10)
        
        # Trend Strength
        features['trend_strength'] = TechnicalIndicators.calculate_trend_strength(df['Close'])
        
        # Volatility
        features['volatility_ratio'] = TechnicalIndicators.calculate_volatility_ratio(df)
        features['historical_volatility'] = df['Close'].pct_change().rolling(window=20).std()
        
        # VWAP
        features['vwap'] = TechnicalIndicators.calculate_vwap(df)
        features['vwap_distance'] = (df['Close'] - features['vwap']) / df['Close']
        
        # Volume features (if available)
        if 'Volume' in df.columns and df['Volume'].sum() > 0:
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
            features['volume_trend'] = df['Volume'].pct_change().rolling(window=5).mean()
        
        # Replace infinities and fill NaN values
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(0)
        
        return features

# ============================================================================
# ENSEMBLE ML MODEL
# ============================================================================

class EnsembleModel:
    """Ensemble learning model combining multiple classifiers."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        """
        Initialize ensemble model with multiple classifiers.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.models = {}
        self.is_trained = False
        self.last_training_time = None
        self.feature_importance = {}
        
        # Create models directory
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all ML models with optimized parameters."""
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=6,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbosity=0
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=6,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=200,
                learning_rate=0.03,
                depth=6,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False,
                thread_count=-1
            ),
            'randomforest': RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1,
                verbose=0
            )
        }
        
        # Model weights (can be adjusted based on performance)
        self.model_weights = {
            'xgboost': 0.30,
            'lightgbm': 0.25,
            'catboost': 0.25,
            'randomforest': 0.20
        }
        
        self.logger.info(f"Initialized {len(self.models)} models for ensemble learning")
    
    def _create_target(self, df: pd.DataFrame, forward_periods: int = 3) -> pd.Series:
        """
        Create binary target variable for training.
        
        Args:
            df: DataFrame with price data
            forward_periods: Number of periods to look ahead
            
        Returns:
            Binary target series (1 for price increase, 0 for decrease)
        """
        future_prices = df['Close'].shift(-forward_periods)
        target = (future_prices > df['Close']).astype(int)
        return target
    
    def train(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """
        Train ensemble model on historical data.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol for model saving
            
        Returns:
            Training metrics dictionary
        """
        try:
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                self.logger.warning(f"Insufficient data for training: {len(df)} samples")
                return None
            
            self.logger.info(f"Starting ensemble training for {symbol} with {len(df)} samples")
            start_time = time.time()
            
            # Prepare features and target
            features_df = TechnicalIndicators.calculate_all_features(df)
            target = self._create_target(df)
            
            # Remove NaN values
            valid_idx = ~(features_df.isna().any(axis=1) | target.isna())
            X = features_df[valid_idx]
            y = target[valid_idx]
            
            if len(X) < 100:
                self.logger.warning("Not enough valid samples after preprocessing")
                return None
            
            # Split data
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=self.config.TEST_SIZE, shuffle=False
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, 
                test_size=self.config.VALIDATION_SIZE / (1 - self.config.TEST_SIZE),
                shuffle=False
            )
            
            self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            
            # Train each model
            model_metrics = {}
            all_predictions = {}
            
            for name, model in self.models.items():
                self.logger.info(f"Training {name}...")
                
                # Train model
                if name == 'catboost':
                    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
                elif name == 'lightgbm':
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                else:
                    model.fit(X_train, y_train)
                
                # Cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='accuracy')
                
                # Validation predictions
                val_pred = model.predict(X_val)
                val_prob = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else val_pred
                
                # Store metrics
                model_metrics[name] = {
                    'accuracy': accuracy_score(y_val, val_pred),
                    'precision': precision_score(y_val, val_pred, zero_division=0),
                    'recall': recall_score(y_val, val_pred, zero_division=0),
                    'f1_score': f1_score(y_val, val_pred, zero_division=0),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                all_predictions[name] = val_prob
                
                self.logger.info(f"{name} - Accuracy: {model_metrics[name]['accuracy']:.3f}, "
                               f"F1: {model_metrics[name]['f1_score']:.3f}")
            
            # Calculate ensemble performance
            ensemble_pred = np.zeros(len(y_val))
            for name, preds in all_predictions.items():
                ensemble_pred += self.model_weights[name] * preds
            
            ensemble_binary = (ensemble_pred > 0.5).astype(int)
            
            # Test set evaluation
            test_metrics = {}
            for name, model in self.models.items():
                test_pred = model.predict(X_test)
                test_metrics[name] = {
                    'test_accuracy': accuracy_score(y_test, test_pred),
                    'test_f1': f1_score(y_test, test_pred, zero_division=0)
                }
            
            # Save models
            self._save_models(symbol)
            
            training_time = time.time() - start_time
            self.is_trained = True
            self.last_training_time = datetime.now()
            
            # Compile training metrics
            metrics = {
                'model_name': f'ensemble_{symbol}',
                'accuracy': accuracy_score(y_val, ensemble_binary),
                'precision': precision_score(y_val, ensemble_binary, zero_division=0),
                'recall': recall_score(y_val, ensemble_binary, zero_division=0),
                'f1_score': f1_score(y_val, ensemble_binary, zero_division=0),
                'training_time': training_time,
                'samples_count': len(X),
                'individual_models': model_metrics,
                'test_metrics': test_metrics
            }
            
            self.logger.info(f"Training completed in {training_time:.2f} seconds")
            self.logger.info(f"Ensemble F1 Score: {metrics['f1_score']:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}", exc_info=True)
            return None
    
    def predict_proba(self, features_df: pd.DataFrame) -> float:
        """
        Get ensemble probability prediction.
        
        Args:
            features_df: DataFrame with technical features
            
        Returns:
            Ensemble probability (0-1)
        """
        if not self.is_trained:
            return 0.5
        
        try:
            ensemble_prob = 0.0
            
            for name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    prob = model.predict_proba(features_df)[:, 1][0]
                    ensemble_prob += self.model_weights[name] * prob
            
            return ensemble_prob
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}", exc_info=True)
            return 0.5
    
    def predict(self, features_df: pd.DataFrame) -> Tuple[str, float]:
        """
        Get ensemble prediction and confidence.
        
        Args:
            features_df: DataFrame with technical features
            
        Returns:
            Tuple of (direction, confidence)
        """
        prob = self.predict_proba(features_df)
        
        if prob > self.config.CONFIDENCE_THRESHOLD:
            return "BUY", prob
        elif prob < (1 - self.config.CONFIDENCE_THRESHOLD):
            return "SELL", 1 - prob
        else:
            return "NEUTRAL", max(prob, 1 - prob)
    
    def _save_models(self, symbol: str) -> None:
        """Save trained models to disk."""
        try:
            for name, model in self.models.items():
                model_path = os.path.join(self.config.MODELS_DIR, f"{symbol}_{name}.pkl")
                joblib.dump(model, model_path)
            
            self.logger.info(f"Models saved for {symbol}")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}", exc_info=True)
    
    def load_models(self, symbol: str) -> bool:
        """
        Load trained models from disk.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if models loaded successfully
        """
        try:
            for name in self.models.keys():
                model_path = os.path.join(self.config.MODELS_DIR, f"{symbol}_{name}.pkl")
                if os.path.exists(model_path):
                    self.models[name] = joblib.load(model_path)
                    self.logger.info(f"Loaded {name} model for {symbol}")
                else:
                    self.logger.warning(f"No saved model found for {symbol}_{name}")
                    return False
            
            self.is_trained = True
            self.last_training_time = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}", exc_info=True)
            return False

# ============================================================================
# CHART GENERATOR
# ============================================================================

class ChartGenerator:
    """Professional chart generation for trading signals."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        """
        Initialize chart generator.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
    
    def create_signal_chart(self, df: pd.DataFrame, symbol: str, signal: Dict[str, Any]) -> str:
        """
        Create professional chart with technical indicators and signal annotation.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            signal: Signal information dictionary
            
        Returns:
            Path to saved chart image
        """
        try:
            # Use last N candles
            chart_df = df.tail(self.config.CHART_CANDLES).copy()
            
            # Set up the plot
            plt.style.use('dark_background')
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.config.CHART_FIGSIZE,
                                                  gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Prepare data
            dates = chart_df.index
            closes = chart_df['Close'].values
            opens = chart_df['Open'].values if 'Open' in chart_df else closes
            highs = chart_df['High'].values if 'High' in chart_df else closes
            lows = chart_df['Low'].values if 'Low' in chart_df else closes
            
            # Calculate colors for candles
            colors = ['#00ff00' if c >= o else '#ff0000' for c, o in zip(closes, opens)]
            
            # Plot candlesticks
            for i in range(len(chart_df)):
                # Wick (high-low)
                ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], 
                        color=colors[i], linewidth=0.8, alpha=0.8)
                # Body (open-close)
                body_height = abs(closes[i] - opens[i])
                body_bottom = min(closes[i], opens[i])
                ax1.add_patch(plt.Rectangle(
                    (mdates.date2num(dates[i]) - 0.3, body_bottom),
                    0.6, body_height,
                    facecolor=colors[i], edgecolor=colors[i], alpha=0.9
                ))
            
            # Add moving averages
            ema20 = chart_df['Close'].ewm(span=20, adjust=False).mean()
            ema50 = chart_df['Close'].ewm(span=50, adjust=False).mean()
            ax1.plot(dates, ema20, color='#00bfff', linewidth=1, label='EMA 20', alpha=0.7)
            ax1.plot(dates, ema50, color='#ff6347', linewidth=1, label='EMA 50', alpha=0.7)
            
            # Bollinger Bands
            bb = TechnicalIndicators.calculate_bollinger_bands(chart_df['Close'])
            ax1.fill_between(dates, bb['upper'], bb['lower'], alpha=0.1, color='gray')
            ax1.plot(dates, bb['upper'], color='gray', linewidth=0.5, alpha=0.5)
            ax1.plot(dates, bb['lower'], color='gray', linewidth=0.5, alpha=0.5)
            
            # Mark entry point
            entry_price = signal['entry_price']
            entry_time = dates[-1]
            ax1.scatter(entry_time, entry_price, color='yellow', s=200, zorder=5, 
                       marker='*', edgecolors='orange', linewidth=2)
            
            # Direction arrow
            arrow_color = '#00ff00' if signal['direction'] == 'BUY' else '#ff0000'
            arrow_direction = 1 if signal['direction'] == 'BUY' else -1
            ax1.annotate('', xy=(entry_time, entry_price * (1 + 0.02 * arrow_direction)),
                        xytext=(entry_time, entry_price),
                        arrowprops=dict(arrowstyle='->', color=arrow_color, lw=3))
            
            # Title and labels
            ax1.set_title(f'{symbol} - {signal["direction"]} Signal\nConfidence: {signal["confidence"]:.1%}',
                         fontsize=14, fontweight='bold', color='white')
            ax1.set_ylabel('Price', color='white')
            ax1.legend(loc='upper left', fontsize=8)
            ax1.grid(True, alpha=0.3)
            
            # RSI subplot
            rsi = TechnicalIndicators.calculate_rsi(chart_df['Close'])
            ax2.plot(dates, rsi, color='#9370db', linewidth=1.5)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax2.fill_between(dates, 70, 100, alpha=0.1, color='red')
            ax2.fill_between(dates, 0, 30, alpha=0.1, color='green')
            ax2.set_ylabel('RSI', color='white')
            ax2.grid(True, alpha=0.3)
            
            # MACD subplot
            macd_data = TechnicalIndicators.calculate_macd(chart_df['Close'])
            ax3.plot(dates, macd_data['macd'], color='#00bfff', linewidth=1.5, label='MACD')
            ax3.plot(dates, macd_data['signal'], color='#ff6347', linewidth=1.5, label='Signal')
            ax3.bar(dates, macd_data['histogram'], 
                   color=['#00ff00' if x > 0 else '#ff0000' for x in macd_data['histogram']],
                   alpha=0.5, width=0.6)
            ax3.set_ylabel('MACD', color='white')
            ax3.set_xlabel('Time', color='white')
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(True, alpha=0.3)
            
            # Format axes
            for ax in [ax1, ax2, ax3]:
                ax.tick_params(colors='white', labelsize=8)
                for spine in ax.spines.values():
                    spine.set_color('#333333')
            
            # Rotate x-axis labels
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            # Save chart
            chart_path = f"chart_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.tight_layout()
            plt.savefig(chart_path, dpi=self.config.CHART_DPI, facecolor='#1a1a1a', 
                       bbox_inches='tight', pad_inches=0.5)
            plt.close()
            
            self.logger.info(f"Chart generated: {chart_path}")
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Chart generation failed: {e}", exc_info=True)
            return None

# ============================================================================
# DATA FETCHER
# ============================================================================

class DataFetcher:
    """Data fetching module with retry logic."""
    
    def __init__(self, config: Config, logger: logging.Logger):
        """
        Initialize data fetcher.
        
        Args:
            config: Configuration object
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
    
    def fetch_data(self, symbol: str, interval: str = '5m', period: str = '7d') -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data from Yahoo Finance with retry logic.
        
        Args:
            symbol: Trading symbol
            interval: Data interval (e.g., '5m', '15m', '1h')
            period: Data period (e.g., '7d', '1mo', '3mo')
            
        Returns:
            DataFrame with OHLCV data or None if fetch fails
        """
        for attempt in range(self.config.MAX_RETRIES):
            try:
                self.logger.debug(f"Fetching {symbol} data ({interval}, {period}) - Attempt {attempt + 1}")
                
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df.empty:
                    raise ValueError(f"Empty DataFrame for {symbol}")
                
                # Standardize column names
                df.columns = [col.capitalize() for col in df.columns]
                
                # Ensure required columns exist
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in df.columns:
                        if col == 'Volume':
                            df[col] = 0
                        else:
                            df[col] = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
                
                self.logger.debug(f"Successfully fetched {len(df)} rows for {symbol}")
                return df
                
            except Exception as e:
                self.logger.warning(f"Fetch attempt {attempt + 1} failed for {symbol}: {e}")
                if attempt < self.config.MAX_RET
