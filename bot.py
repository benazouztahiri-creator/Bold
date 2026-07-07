#!/usr/bin/env python3
"""
Falcon AI Ultimate v2.1 - Full Features + Server Optimized
===========================================================
All features included, optimized for cloud deployment.
"""

import os
import sys
import time
import json
import logging
import sqlite3
import hashlib
import warnings
import threading
import gc
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.signal import argrelextrema

# Matplotlib - memory optimized
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['figure.dpi'] = 72  # Lower DPI for server

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import telebot
from telebot import types
import joblib
import shutil

# Reduce warnings and optimize memory
warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Central configuration with environment variable support."""
    
    # Telegram - Use env vars for security
    TELEGRAM_TOKEN: str = os.environ.get(
        'TELEGRAM_TOKEN', 
        '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
    )
    TELEGRAM_CHAT_ID: str = os.environ.get(
        'TELEGRAM_CHAT_ID', 
        '7553333305'
    )
    
    # Trading
    TRADE_DURATION_MINUTES: int = 10
    TIMEFRAMES: List[str] = field(default_factory=lambda: ['5m', '15m'])
    SCAN_INTERVAL_MINUTES: int = int(os.environ.get('SCAN_INTERVAL', '5'))
    
    # Symbols - Reduced for faster scanning
    SYMBOLS: List[str] = field(default_factory=lambda: [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X',
        'BTC-USD', 'ETH-USD', 'SOL-USD', 'GC=F', 'XAUUSD=X'
    ])
    
    # ML Settings - Adjusted for server
    CONFIDENCE_THRESHOLD: float = 0.65
    RETRAINING_INTERVAL_HOURS: int = 24
    MIN_TRAINING_SAMPLES: int = 500
    TRAINING_PERIOD: str = '3mo'
    
    # Feature Selection
    MAX_FEATURES: int = 40
    FEATURE_SELECTION_METHOD: str = 'mutual_info'
    
    # Model Parameters - Optimized for speed
    CV_FOLDS: int = 3
    VALIDATION_SIZE: float = 0.2
    TEST_SIZE: float = 0.15
    
    # Database
    DB_PATH: str = 'falcon_trading.db'
    MODELS_DIR: str = 'models'
    BACKUP_DIR: str = 'backups'
    
    # Charts - Lower quality for server
    CHART_CANDLES: int = 40
    CHART_DPI: int = 72
    CHART_FIGSIZE: Tuple[int, int] = (12, 8)
    
    # Performance
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5
    MAX_WORKERS: int = 4  # Reduced for server
    
    # Signal Cooldown
    SIGNAL_COOLDOWN_MINUTES: int = 15
    
    # Paths
    LOG_FILE: str = 'falcon_bot.log'

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

class MemoryManager:
    """Memory optimization utilities."""
    
    @staticmethod
    def clear_memory():
        """Clear unused memory."""
        gc.collect()
        plt.close('all')
    
    @staticmethod
    def limit_pandas_memory():
        """Limit pandas memory usage."""
        pd.options.mode.chained_assignment = None
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Downcast dataframe types to save memory."""
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        return df

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Simple but effective logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('FalconAI')

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    """Optimized database with all needed features."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init()
    
    def _init(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    confidence REAL NOT NULL,
                    m5_analysis TEXT,
                    m15_analysis TEXT,
                    trend_filter TEXT,
                    patterns_detected TEXT,
                    models_agreement TEXT,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME,
                    exit_time DATETIME,
                    result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL,
                    model_version TEXT,
                    signal_hash TEXT UNIQUE
                );
                
                CREATE TABLE IF NOT EXISTS model_registry (
                    symbol TEXT PRIMARY KEY,
                    model_version TEXT,
                    features_count INTEGER,
                    training_samples INTEGER,
                    accuracy REAL,
                    precision_score REAL,
                    recall_score REAL,
                    f1_score REAL,
                    brier_score REAL,
                    calibration_error REAL,
                    selected_features TEXT,
                    feature_importance TEXT,
                    model_weights TEXT,
                    trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    training_duration_seconds REAL,
                    is_active INTEGER DEFAULT 1
                );
                
                CREATE TABLE IF NOT EXISTS performance (
                    period TEXT PRIMARY KEY,
                    total_signals INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0.0,
                    avg_confidence REAL DEFAULT 0.0,
                    avg_pnl REAL DEFAULT 0.0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
                CREATE INDEX IF NOT EXISTS idx_signals_result ON signals(result);
                CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(entry_time);
            ''')
            conn.commit()
    
    def save_signal(self, data: Dict) -> Optional[int]:
        """Save signal with duplicate check."""
        try:
            hash_str = f"{data['symbol']}_{data['direction']}_{datetime.now().timestamp()}"
            signal_hash = hashlib.md5(hash_str.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO signals 
                    (symbol, direction, entry_price, confidence, m5_analysis, m15_analysis,
                     trend_filter, patterns_detected, models_agreement, expiry_time, 
                     model_version, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    data['symbol'], data['direction'], data['entry_price'],
                    data['confidence'], data.get('m5_analysis'), data.get('m15_analysis'),
                    data.get('trend_filter'), data.get('patterns_detected'),
                    data.get('models_agreement'), data['expiry_time'],
                    data.get('model_version'), signal_hash
                ))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except:
            return None
    
    def check_active_signal(self, symbol: str) -> bool:
        """Check if symbol has active trade."""
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute('''
                SELECT COUNT(*) FROM signals 
                WHERE symbol = ? AND result = 'PENDING' 
                AND expiry_time > datetime('now', 'localtime')
            ''', (symbol,)).fetchone()[0]
            return count > 0
    
    def check_recent_signal(self, symbol: str, minutes: int) -> bool:
        """Check cooldown period."""
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute('''
                SELECT COUNT(*) FROM signals 
                WHERE symbol = ? AND entry_time > ?
            ''', (symbol, cutoff)).fetchone()[0]
            return count > 0
    
    def update_result(self, signal_id: int, exit_price: float, result: str, pnl: float):
        """Update trade result."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE signals SET 
                exit_price = ?, result = ?, pnl_percent = ?, 
                exit_time = datetime('now', 'localtime')
                WHERE id = ?
            ''', (exit_price, result, pnl, signal_id))
            
            # Update performance
            today = datetime.now().strftime('%Y-%m-%d')
            conn.execute('''
                INSERT INTO performance (period, total_signals, wins, losses)
                VALUES (?, 1, ?, ?)
                ON CONFLICT(period) DO UPDATE SET
                total_signals = total_signals + 1,
                wins = wins + ?,
                losses = losses + ?,
                updated_at = datetime('now', 'localtime')
            ''', (today, 
                  1 if result == 'WIN' else 0,
                  1 if result == 'LOSS' else 0,
                  1 if result == 'WIN' else 0,
                  1 if result == 'LOSS' else 0))
            conn.commit()
    
    def get_pending_trades(self) -> List[Dict]:
        """Get expired pending trades."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM signals 
                WHERE result = 'PENDING' 
                AND expiry_time <= datetime('now', 'localtime')
            ''').fetchall()
            return [dict(r) for r in rows]
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE result != 'PENDING'"
            ).fetchone()[0]
            wins = conn.execute(
                "SELECT COUNT(*) FROM signals WHERE result = 'WIN'"
            ).fetchone()[0]
            
            # Per symbol stats
            symbols = conn.execute('''
                SELECT symbol, COUNT(*) as cnt,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as w
                FROM signals WHERE result != 'PENDING'
                GROUP BY symbol HAVING cnt >= 3
                ORDER BY w*1.0/cnt DESC
            ''').fetchall()
            
            return {
                'total': total,
                'wins': wins,
                'losses': total - wins,
                'win_rate': wins / total if total > 0 else 0,
                'best_symbol': symbols[0][0] if symbols else 'N/A',
                'worst_symbol': symbols[-1][0] if symbols else 'N/A'
            }
    
    def save_model_metrics(self, symbol: str, metrics: Dict):
        """Save model training metrics."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT OR REPLACE INTO model_registry 
                (symbol, model_version, features_count, training_samples,
                 accuracy, precision_score, recall_score, f1_score,
                 brier_score, calibration_error, selected_features,
                 feature_importance, model_weights, trained_at,
                 training_duration_seconds, is_active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'), ?, 1)
            ''', (
                symbol,
                metrics.get('model_version'),
                metrics.get('features_count'),
                metrics.get('training_samples'),
                metrics.get('accuracy'),
                metrics.get('precision'),
                metrics.get('recall'),
                metrics.get('f1_score'),
                metrics.get('brier_score'),
                metrics.get('calibration_error'),
                json.dumps(metrics.get('selected_features', [])),
                json.dumps(metrics.get('feature_importance', {})),
                json.dumps(metrics.get('model_weights', {})),
                metrics.get('training_duration', 0)
            ))
            conn.commit()
    
    def backup(self):
        """Create database backup."""
        try:
            os.makedirs('backups', exist_ok=True)
            backup_path = f"backups/falcon_backup_{datetime.now().strftime('%Y%m%d')}.db"
            shutil.copy2(self.db_path, backup_path)
        except:
            pass

# ============================================================================
# TECHNICAL ANALYZER - FULL FEATURES
# ============================================================================

class TechnicalAnalyzer:
    """Complete technical analysis with all indicators and patterns."""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators."""
        f = pd.DataFrame(index=df.index)
        c, h, l = df['Close'], df['High'], df['Low']
        v = df.get('Volume', pd.Series(0, index=df.index))
        
        # Price features
        for p in [1, 3, 5, 10, 20]:
            f[f'ret_{p}'] = c.pct_change(p)
        f['log_ret'] = np.log(c / c.shift(1))
        f['hl_ratio'] = (h - l) / c
        f['close_pos'] = (c - l) / (h - l + 1e-8)
        
        # Moving Averages
        for p in [5, 10, 20, 50, 100, 200]:
            if len(df) >= p:
                f[f'sma_{p}'] = c.rolling(p).mean()
                f[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
                f[f'price_sma_{p}'] = c / f[f'sma_{p}'] - 1
        
        # RSI (multiple periods)
        for p in [7, 14, 21]:
            delta = c.diff()
            gain = delta.where(delta > 0, 0.0).rolling(p).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(p).mean()
            rs = gain / (loss + 1e-8)
            f[f'rsi_{p}'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = c.ewm(span=12).mean()
        ema26 = c.ewm(span=26).mean()
        f['macd'] = ema12 - ema26
        f['macd_signal'] = f['macd'].ewm(span=9).mean()
        f['macd_hist'] = f['macd'] - f['macd_signal']
        
        # Bollinger Bands
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        f['bb_upper'] = sma20 + 2 * std20
        f['bb_lower'] = sma20 - 2 * std20
        f['bb_pos'] = (c - f['bb_lower']) / (f['bb_upper'] - f['bb_lower'] + 1e-8)
        f['bb_width'] = (f['bb_upper'] - f['bb_lower']) / sma20
        
        # Stochastic
        low14 = l.rolling(14).min()
        high14 = h.rolling(14).max()
        f['stoch_k'] = 100 * (c - low14) / (high14 - low14 + 1e-8)
        f['stoch_d'] = f['stoch_k'].rolling(3).mean()
        
        # ATR
        tr1 = h - l
        tr2 = abs(h - c.shift())
        tr3 = abs(l - c.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        f['atr'] = tr.ewm(span=14).mean()
        f['atr_pct'] = f['atr'] / c
        
        # CCI
        tp = (h + l + c) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        f['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-8)
        
        # Williams %R
        hh14 = h.rolling(14).max()
        ll14 = l.rolling(14).min()
        f['williams_r'] = -100 * (hh14 - c) / (hh14 - ll14 + 1e-8)
        
        # ROC
        for p in [5, 10, 20]:
            f[f'roc_{p}'] = (c - c.shift(p)) / c.shift(p) * 100
        
        # Momentum
        for p in [5, 10, 20]:
            f[f'mom_{p}'] = c - c.shift(p)
        
        # Volatility
        for p in [5, 10, 20]:
            f[f'vol_{p}'] = c.pct_change().rolling(p).std()
        
        # Volume indicators
        if v.sum() > 0:
            f['vol_change'] = v.pct_change()
            f['vol_ratio'] = v / v.rolling(20).mean()
            f['vol_trend'] = v.rolling(5).mean() / v.rolling(20).mean()
        
        # Trend strength
        f['trend_str'] = c.rolling(20).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # Cleanup
        f = f.replace([np.inf, -np.inf], np.nan)
        f = f.ffill().bfill().fillna(0)
        
        return MemoryManager.optimize_dataframe(f)
    
    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict]:
        """Detect candlestick patterns."""
        patterns = []
        if len(df) < 3:
            return patterns
        
        c1, c2, c3 = df.iloc[-1], df.iloc[-2], df.iloc[-3]
        body1 = abs(c1['Close'] - c1['Open'])
        body2 = abs(c2['Close'] - c2['Open'])
        body3 = abs(c3['Close'] - c3['Open'])
        
        upper1 = c1['High'] - max(c1['Close'], c1['Open'])
        lower1 = min(c1['Close'], c1['Open']) - c1['Low']
        total1 = c1['High'] - c1['Low']
        
        # Pin Bars
        if body1 > 0 and total1 > 0:
            if lower1 > body1 * 2 and upper1 < body1 * 0.3:
                patterns.append({'pattern': 'Hammer', 'strength': 'strong', 'direction': 'BUY'})
            if upper1 > body1 * 2 and lower1 < body1 * 0.3:
                patterns.append({'pattern': 'Shooting Star', 'strength': 'strong', 'direction': 'SELL'})
        
        # Engulfing
        if body1 > 0 and body2 > 0:
            if c2['Close'] < c2['Open'] and c1['Close'] > c1['Open']:
                if c1['Open'] <= c2['Close'] and c1['Close'] >= c2['Open']:
                    patterns.append({'pattern': 'Bullish Engulfing', 'strength': 'strong', 'direction': 'BUY'})
            if c2['Close'] > c2['Open'] and c1['Close'] < c1['Open']:
                if c1['Open'] >= c2['Close'] and c1['Close'] <= c2['Open']:
                    patterns.append({'pattern': 'Bearish Engulfing', 'strength': 'strong', 'direction': 'SELL'})
        
        # Doji
        if body1 < total1 * 0.1:
            if abs(upper1 - lower1) < body1:
                patterns.append({'pattern': 'Doji', 'strength': 'moderate', 'direction': 'NEUTRAL'})
            elif upper1 > lower1 * 2:
                patterns.append({'pattern': 'Gravestone Doji', 'strength': 'moderate', 'direction': 'SELL'})
            elif lower1 > upper1 * 2:
                patterns.append({'pattern': 'Dragonfly Doji', 'strength': 'moderate', 'direction': 'BUY'})
        
        # Stars
        if body2 < body3 * 0.3:
            if c3['Close'] < c3['Open'] and c1['Close'] > c1['Open']:
                if c1['Close'] > (c3['Open'] + c3['Close']) / 2:
                    patterns.append({'pattern': 'Morning Star', 'strength': 'strong', 'direction': 'BUY'})
            if c3['Close'] > c3['Open'] and c1['Close'] < c1['Open']:
                if c1['Close'] < (c3['Open'] + c3['Close']) / 2:
                    patterns.append({'pattern': 'Evening Star', 'strength': 'strong', 'direction': 'SELL'})
        
        return patterns
    
    @staticmethod
    def detect_support_resistance(df: pd.DataFrame) -> Dict:
        """Detect S/R levels."""
        try:
            highs, lows = df['High'].values, df['Low'].values
            max_idx = argrelextrema(highs, np.greater, order=15)[0]
            min_idx = argrelextrema(lows, np.less, order=15)[0]
            
            resistance = sorted(
                list(set([round(highs[i], 5) for i in max_idx if i < len(highs)])),
                reverse=True
            )[:3]
            
            support = sorted(
                list(set([round(lows[i], 5) for i in min_idx if i < len(lows)]))
            )[:3]
            
            return {'resistance': resistance, 'support': support}
        except:
            return {'resistance': [], 'support': []}
    
    @staticmethod
    def calculate_trend_filter(df: pd.DataFrame) -> Dict:
        """Calculate trend direction and strength."""
        if len(df) < 50:
            return {'trend': 'NEUTRAL', 'strength': 0}
        
        c = df['Close']
        current = c.iloc[-1]
        ema20 = c.ewm(span=20).mean().iloc[-1]
        ema50 = c.ewm(span=50).mean().iloc[-1]
        ema200 = c.ewm(span=200).mean().iloc[-1] if len(df) >= 200 else ema50
        
        # ADX calculation
        h, l = df['High'], df['Low']
        tr1 = h - l
        tr2 = abs(h - c.shift())
        tr3 = abs(l - c.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=14).mean()
        
        plus_dm = h.diff().clip(lower=0)
        minus_dm = (-l.diff()).clip(lower=0)
        plus_di = 100 * (plus_dm.ewm(span=14).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=14).mean() / atr)
        adx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx_val = adx.ewm(span=14).mean().iloc[-1]
        
        # Score
        score = 0
        reasons = []
        
        if current > ema20: score += 1; reasons.append('Price > EMA20')
        else: score -= 1
        
        if current > ema50: score += 1; reasons.append('Price > EMA50')
        else: score -= 1
        
        if current > ema200: score += 1; reasons.append('Price > EMA200')
        else: score -= 1
        
        if ema20 > ema50: score += 1; reasons.append('EMA20 > EMA50')
        else: score -= 1
        
        if adx_val > 25:
            if score > 0: score += 1
            elif score < 0: score -= 1
        
        if score >= 3:
            trend = 'STRONG_BULLISH'
            strength = min(abs(score) / 5, 1.0)
        elif score >= 1:
            trend = 'BULLISH'
            strength = abs(score) / 5
        elif score <= -3:
            trend = 'STRONG_BEARISH'
            strength = min(abs(score) / 5, 1.0)
        elif score <= -1:
            trend = 'BEARISH'
            strength = abs(score) / 5
        else:
            trend = 'NEUTRAL'
            strength = 0
        
        return {'trend': trend, 'strength': strength, 'score': score, 
                'adx': round(adx_val, 2), 'reasons': reasons}

# ============================================================================
# PER-ASSET MODEL - FULL FEATURES
# ============================================================================

class PerAssetModel:
    """Complete ensemble model per symbol with all features."""
    
    def __init__(self, symbol: str, config: Config, logger: logging.Logger):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        
        self.models = {}
        self.calibrators = {}
        self.scaler = RobustScaler()
        self.selected_features = []
        self.feature_importance = {}
        self.model_weights = {
            'xgboost': 0.30, 'lightgbm': 0.25,
            'catboost': 0.25, 'gradient_boost': 0.20
        }
        self.is_trained = False
        self.model_version = None
        
        self._init_models()
    
    def _init_models(self):
        """Initialize models with server-optimized parameters."""
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.03, max_depth=5,
                min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.5, reg_lambda=1.0, random_state=42,
                n_jobs=2, verbosity=0, tree_method='hist'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.03, max_depth=5,
                num_leaves=31, min_child_samples=20, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=1.0,
                random_state=42, n_jobs=2, verbose=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=200, learning_rate=0.03, depth=5,
                l2_leaf_reg=3, random_seed=42, verbose=False,
                thread_count=2, allow_writing_files=False
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.03, max_depth=4,
                min_samples_split=10, min_samples_leaf=5,
                subsample=0.8, random_state=42
            )
        }
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Automated feature selection."""
        self.logger.info(f"Feature selection for {self.symbol}: {len(X.columns)} features")
        
        # Remove constant/low variance
        constant = [c for c in X.columns if X[c].std() < 1e-8]
        X = X.drop(columns=constant)
        
        # Remove highly correlated
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [c for c in upper.columns if any(upper[c] > 0.95)]
        X = X.drop(columns=high_corr)
        
        # Mutual information
        mi = mutual_info_classif(X, y, random_state=42)
        scores = sorted(zip(X.columns, mi), key=lambda x: x[1], reverse=True)
        
        top_k = min(self.config.MAX_FEATURES, len(scores))
        self.selected_features = [s[0] for s in scores[:top_k]]
        
        self.logger.info(f"Selected {len(self.selected_features)} features for {self.symbol}")
        return self.selected_features
    
    def train(self, df: pd.DataFrame) -> Optional[Dict]:
        """Train model with all metrics."""
        try:
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                return None
            
            self.logger.info(f"Training {self.symbol}: {len(df)} samples")
            start = time.time()
            
            features = TechnicalAnalyzer.calculate_all_indicators(df)
            future_ret = df['Close'].shift(-3) / df['Close'] - 1
            target = (future_ret > 0.001).astype(int)
            
            valid = ~(features.isna().any(axis=1) | target.isna())
            X, y = features[valid], target[valid]
            
            if len(X) < 100:
                return None
            
            # Feature selection
            self.select_features(X, y)
            X = X[self.selected_features]
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.TEST_SIZE, shuffle=False
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, 
                test_size=self.config.VALIDATION_SIZE / (1 - self.config.TEST_SIZE),
                shuffle=False
            )
            
            # Scale
            X_train_s = self.scaler.fit_transform(X_train)
            X_val_s = self.scaler.transform(X_val)
            X_test_s = self.scaler.transform(X_test)
            
            # Train models
            predictions, probas = {}, {}
            
            for name, model in self.models.items():
                try:
                    model.fit(X_train_s, y_train)
                    
                    cal = CalibratedClassifierCV(model, cv=3, method='isotonic')
                    cal.fit(X_train_s, y_train)
                    self.calibrators[name] = cal
                    
                    probas[name] = cal.predict_proba(X_val_s)[:, 1]
                    predictions[name] = cal.predict(X_val_s)
                except Exception as e:
                    self.logger.error(f"Train {name} error: {e}")
            
            if len(probas) < 2:
                return None
            
            # Ensemble
            ensemble_prob = sum(
                self.model_weights.get(n, 0) * p for n, p in probas.items()
            )
            ensemble_pred = (ensemble_prob > 0.5).astype(int)
            
            # Metrics
            acc = accuracy_score(y_val, ensemble_pred)
            prec = precision_score(y_val, ensemble_pred, zero_division=0)
            rec = recall_score(y_val, ensemble_pred, zero_division=0)
            f1 = f1_score(y_val, ensemble_pred, zero_division=0)
            brier = brier_score_loss(y_val, ensemble_prob)
            cal_err = self._calc_calibration_error(y_val, ensemble_prob)
            
            # Update weights
            scores = {}
            for name in predictions:
                scores[name] = f1_score(y_val, predictions[name], zero_division=0)
            total = sum(scores.values())
            if total > 0:
                for name in self.model_weights:
                    self.model_weights[name] = scores.get(name, 0) / total
            
            # Feature importance
            if 'xgboost' in self.models:
                self.feature_importance = dict(
                    zip(self.selected_features, self.models['xgboost'].feature_importances_)
                )
            
            self.is_trained = True
            self.model_version = datetime.now().strftime('v%Y%m%d_%H%M%S')
            train_time = time.time() - start
            
            metrics = {
                'model_version': self.model_version,
                'features_count': len(self.selected_features),
                'training_samples': len(X_train),
                'accuracy': acc, 'precision': prec, 'recall': rec,
                'f1_score': f1, 'brier_score': brier,
                'calibration_error': cal_err,
                'selected_features': self.selected_features,
                'feature_importance': self.feature_importance,
                'model_weights': self.model_weights.copy(),
                'training_duration': train_time
            }
            
            self.logger.info(f"{self.symbol} trained: F1={f1:.3f}, Time={train_time:.1f}s")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Train {self.symbol} failed: {e}", exc_info=True)
            return None
    
    def predict(self, df: pd.DataFrame) -> Tuple[str, float, Dict]:
        """Make calibrated prediction."""
        if not self.is_trained:
            return "NEUTRAL", 0.0, {}
        
        try:
            features = TechnicalAnalyzer.calculate_all_indicators(df).iloc[[-1]]
            available = [f for f in self.selected_features if f in features.columns]
            
            if len(available) < 10:
                return "NEUTRAL", 0.0, {}
            
            X = features[available].fillna(0)
            X_s = self.scaler.transform(X)
            
            model_probas = {}
            for name in self.calibrators:
                try:
                    model_probas[name] = float(
                        self.calibrators[name].predict_proba(X_s)[0, 1]
                    )
                except:
                    model_probas[name] = 0.5
            
            if not model_probas:
                return "NEUTRAL", 0.0, {}
            
            prob = sum(
                self.model_weights.get(n, 0) * p for n, p in model_probas.items()
            )
            
            if prob > self.config.CONFIDENCE_THRESHOLD:
                return "BUY", prob, model_probas
            elif prob < (1 - self.config.CONFIDENCE_THRESHOLD):
                return "SELL", 1 - prob, model_probas
            return "NEUTRAL", max(prob, 1 - prob), model_probas
            
        except Exception as e:
            self.logger.error(f"Predict {self.symbol}: {e}")
            return "NEUTRAL", 0.0, {}
    
    def _calc_calibration_error(self, y_true, y_prob):
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            return np.mean(np.abs(prob_true - prob_pred))
        except:
            return 1.0
    
    def save(self):
        """Save model to disk."""
        path = os.path.join(self.config.MODELS_DIR, self.symbol)
        os.makedirs(path, exist_ok=True)
        
        data = {
            'models': self.models,
            'calibrators': self.calibrators,
            'scaler': self.scaler,
            'features': self.selected_features,
            'importance': self.feature_importance,
            'weights': self.model_weights,
            'version': self.model_version
        }
        joblib.dump(data, os.path.join(path, 'model.pkl'))
    
    def load(self) -> bool:
        """Load model from disk."""
        path = os.path.join(self.config.MODELS_DIR, self.symbol, 'model.pkl')
        if not os.path.exists(path):
            return False
        
        data = joblib.load(path)
        self.models = data['models']
        self.calibrators = data['calibrators']
        self.scaler = data['scaler']
        self.selected_features = data['features']
        self.feature_importance = data['importance']
        self.model_weights = data['weights']
        self.model_version = data['version']
        self.is_trained = True
        return True

# ============================================================================
# CHART GENERATOR
# ============================================================================

class ChartGenerator:
    """Generate trading charts."""
    
    @staticmethod
    def create_chart(df: pd.DataFrame, symbol: str, signal: Dict) -> Optional[str]:
        """Create and save chart image."""
        try:
            chart_df = df.tail(40)
            
            fig, (ax1, ax2, ax3) = plt.subplots(
                3, 1, figsize=(12, 8), 
                gridspec_kw={'height_ratios': [3, 1, 1]}
            )
            fig.patch.set_facecolor('#1a1a2e')
            
            # Price chart
            dates = chart_df.index
            closes = chart_df['Close'].values
            
            colors = ['#00ff88' if closes[i] >= closes[i-1] else '#ff4444' 
                     for i in range(1, len(closes))]
            colors.insert(0, colors[0])
            
            ax1.bar(range(len(dates)), closes - chart_df['Open'].values,
                   bottom=chart_df['Open'].values, color=colors, width=0.8)
            
            # EMAs
            ema20 = chart_df['Close'].ewm(span=20).mean()
            ema50 = chart_df['Close'].ewm(span=50).mean()
            ax1.plot(range(len(dates)), ema20, '#00bfff', linewidth=1, alpha=0.7)
            ax1.plot(range(len(dates)), ema50, '#ff6347', linewidth=1, alpha=0.7)
            
            # Entry point
            ax1.scatter(len(dates)-1, signal['entry_price'], 
                       color='yellow', s=150, marker='*', zorder=5)
            
            ax1.set_title(f'{symbol} - {signal["direction"]} Signal\nConfidence: {signal["confidence"]:.1%}',
                         color='white', fontweight='bold')
            ax1.set_facecolor('#1a1a2e')
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.2)
            
            # RSI
            delta = chart_df['Close'].diff()
            gain = delta.where(delta > 0, 0.0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
            rs = gain / (loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            ax2.plot(range(len(dates)), rsi, '#9370db', linewidth=1.5)
            ax2.axhline(y=70, color='red', linestyle='--', alpha=0.5)
            ax2.axhline(y=30, color='green', linestyle='--', alpha=0.5)
            ax2.set_facecolor('#1a1a2e')
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.2)
            
            # MACD
            ema12 = chart_df['Close'].ewm(span=12).mean()
            ema26 = chart_df['Close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal_line = macd.ewm(span=9).mean()
            hist = macd - signal_line
            
            ax3.bar(range(len(dates)), hist, 
                   color=['#00ff88' if x > 0 else '#ff4444' for x in hist], 
                   alpha=0.7, width=0.8)
            ax3.plot(range(len(dates)), macd, '#00bfff', linewidth=1.5)
            ax3.plot(range(len(dates)), signal_line, '#ff6347', linewidth=1.5)
            ax3.set_facecolor('#1a1a2e')
            ax3.tick_params(colors='white')
            ax3.grid(True, alpha=0.2)
            
            plt.tight_layout()
            
            chart_path = f"chart_{symbol}_{datetime.now().strftime('%H%M%S')}.png"
            plt.savefig(chart_path, dpi=72, facecolor='#1a1a2e', bbox_inches='tight')
            plt.close('all')
            gc.collect()
            
            return chart_path
            
        except Exception as e:
            plt.close('all')
            gc.collect()
            return None

# ============================================================================
# MAIN BOT
# ============================================================================

class FalconBot:
    """Main trading bot orchestrator."""
    
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.db = Database(config.DB_PATH)
        self.models: Dict[str, PerAssetModel] = {}
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        self.chart_gen = ChartGenerator()
        
        # Initialize Telegram
        self.tb = telebot.TeleBot(config.TELEGRAM_TOKEN)
        self._setup_commands()
        
        # Load models
        for symbol in config.SYMBOLS:
            model = PerAssetModel(symbol, config, self.logger)
            if model.load():
                self.logger.info(f"Loaded model: {symbol}")
            else:
                self.logger.info(f"New model: {symbol} (needs training)")
            self.models[symbol] = model
        
        self.running = False
        self.last_retrain = None
    
    def _setup_commands(self):
        @self.tb.message_handler(commands=['start', 'status'])
        def handle(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            trained = sum(1 for m in self.models.values() if m.is_trained)
            stats = self.db.get_stats()
            
            text = f"""
🦅 **Falcon AI Ultimate**

✅ الحالة: يعمل
🤖 النماذج: {trained}/{len(self.models)}
📊 الإشارات: {stats.get('total', 0)}
📈 نسبة النجاح: {stats.get('win_rate', 0):.1%}
⭐ أفضل أصل: {stats.get('best_symbol', 'N/A')}

⚡️ جاري التحليل...
"""
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['stats'])
        def stats_cmd(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            s = self.db.get_stats()
            text = f"""
📊 **إحصائيات الأداء**

📈 الإشارات: {s['total']}
✅ رابحة: {s['wins']}
❌ خاسرة: {s['losses']}
📊 النسبة: {s['win_rate']:.1%}
⭐ الأفضل: {s['best_symbol']}
👎 الأسوأ: {s['worst_symbol']}
"""
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def fetch_data(self, symbol: str, interval: str = '5m', period: str = '5d') -> Optional[pd.DataFrame]:
        """Fetch data with retry."""
        for attempt in range(self.config.MAX_RETRIES):
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                if not df.empty:
                    df.columns = [c.capitalize() for c in df.columns]
                    return MemoryManager.optimize_dataframe(df)
            except Exception as e:
                self.logger.warning(f"Fetch {symbol} attempt {attempt+1}: {e}")
                time.sleep(self.config.RETRY_DELAY)
        return None
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Full analysis pipeline."""
        try:
            model = self.models.get(symbol)
            if not model or not model.is_trained:
                return None
            
            # Check active trade
            if self.db.check_active_signal(symbol):
                return None
            
            # Check cooldown
            if self.db.check_recent_signal(symbol, self.config.SIGNAL_COOLDOWN_MINUTES):
                return None
            
            # Fetch data
            df_5m = self.fetch_data(symbol, '5m', '5d')
            df_15m = self.fetch_data(symbol, '15m', '10d')
            
            if df_5m is None or df_15m is None or len(df_5m) < 30:
                return None
            
            # Predictions
            dir_5m, conf_5m, probas_5m = model.predict(df_5m)
            dir_15m, conf_15m, probas_15m = model.predict(df_15m)
            
            if dir_5m != dir_15m or dir_5m == "NEUTRAL":
                return None
            
            # Trend filter
            trend = TechnicalAnalyzer.calculate_trend_filter(df_15m)
            if (dir_5m == "BUY" and 'BEARISH' in trend['trend']) or \
               (dir_5m == "SELL" and 'BULLISH' in trend['trend']):
                return None
            
            # Patterns
            patterns = TechnicalAnalyzer.detect_candlestick_patterns(df_5m)
            pattern_names = [p['pattern'] for p in patterns if p['direction'] == dir_5m]
            
            # S/R
            sr = TechnicalAnalyzer.detect_support_resistance(df_15m)
            
            # Confidence adjustment
            confidence = (conf_5m + conf_15m) / 2
            if trend['strength'] > 0.6:
                confidence = min(confidence * 1.1, 0.95)
            if pattern_names:
                confidence = min(confidence * 1.05, 0.95)
            
            if confidence < self.config.CONFIDENCE_THRESHOLD:
                return None
            
            return {
                'symbol': symbol,
                'direction': dir_5m,
                'entry_price': float(df_5m['Close'].iloc[-1]),
                'confidence': confidence,
                'm5_analysis': dir_5m,
                'm15_analysis': dir_15m,
                'trend_filter': trend['trend'],
                'patterns_detected': ','.join(pattern_names) if pattern_names else 'None',
                'models_agreement': f"{len(probas_5m)}/4 models",
                'expiry_time': (datetime.now() + timedelta(
                    minutes=self.config.TRADE_DURATION_MINUTES
                )).strftime('%Y-%m-%d %H:%M:%S'),
                'model_version': model.model_version
            }
            
        except Exception as e:
            self.logger.error(f"Analyze {symbol}: {e}")
            return None
    
    def send_signal(self, signal: Dict):
        """Send signal with chart."""
        try:
            emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
            direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
            
            msg = f"""
{emoji} **{signal['symbol']}** - {direction}

💰 السعر: {signal['entry_price']:.5f}
⏳ المدة: {self.config.TRADE_DURATION_MINUTES}:00 دقيقة
💪 الثقة: {signal['confidence']:.1%}

📊 التحليل:
• M5: {signal['m5_analysis']}
• M15: {signal['m15_analysis']}
• الاتجاه: {signal['trend_filter']}
• الأنماط: {signal['patterns_detected']}

🤖 {signal['model_version']} | Falcon AI Ultimate
"""
            
            # Generate chart
            df = self.fetch_data(signal['symbol'], '5m', '3d')
            if df is not None:
                chart_path = self.chart_gen.create_chart(df, signal['symbol'], signal)
                
                if chart_path:
                    with open(chart_path, 'rb') as photo:
                        self.tb.send_photo(
                            self.config.TELEGRAM_CHAT_ID, photo,
                            caption=msg, parse_mode='Markdown'
                        )
                    try:
                        os.remove(chart_path)
                    except:
                        pass
                else:
                    self.tb.send_message(
                        self.config.TELEGRAM_CHAT_ID, msg, parse_mode='Markdown'
                    )
            
            self.logger.info(f"Signal sent: {signal['symbol']} {signal['direction']}")
            
        except Exception as e:
            self.logger.error(f"Send error: {e}")
    
    def check_trades(self):
        """Check and close expired trades."""
        for trade in self.db.get_pending_trades():
            try:
                df = self.fetch_data(trade['symbol'], '5m', '1d')
                if df is None:
                    continue
                
                current = float(df['Close'].iloc[-1])
                entry = trade['entry_price']
                
                if trade['direction'] == 'BUY':
                    pnl = (current - entry) / entry * 100
                    result = 'WIN' if current > entry else 'LOSS'
                else:
                    pnl = (entry - current) / entry * 100
                    result = 'WIN' if current < entry else 'LOSS'
                
                self.db.update_result(trade['id'], current, result, pnl)
                self.logger.info(f"Trade {trade['id']}: {result} ({pnl:.2f}%)")
                
            except Exception as e:
                self.logger.error(f"Check trade {trade.get('id')}: {e}")
    
    def retrain_models(self):
        """Retrain all models."""
        self.logger.info("Starting retraining cycle...")
        
        for symbol in self.config.SYMBOLS:
            try:
                df = self.fetch_data(symbol, '1h', self.config.TRAINING_PERIOD)
                if df is None or len(df) < self.config.MIN_TRAINING_SAMPLES:
                    continue
                
                model = PerAssetModel(symbol, self.config, self.logger)
                metrics = model.train(df)
                
                if metrics:
                    model.save()
                    self.models[symbol] = model
                    self.db.save_model_metrics(symbol, metrics)
                    self.logger.info(f"Retrained: {symbol} (F1: {metrics['f1_score']:.3f})")
                    
            except Exception as e:
                self.logger.error(f"Retrain {symbol}: {e}")
        
        MemoryManager.clear_memory()
        self.last_retrain = datetime.now()
        
        # Notify
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(
                self.config.TELEGRAM_CHAT_ID,
                f"🔄 **إعادة تدريب مكتملة**\n✅ النماذج: {trained}/{len(self.models)}",
                parse_mode='Markdown'
            )
        except:
            pass
    
    def start_telegram(self):
        """Start Telegram polling in thread."""
        def poll():
            self.logger.info("Telegram polling started")
            while True:
                try:
                    self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except Exception as e:
                    self.logger.error(f"Polling error: {e}")
                    time.sleep(10)
        
        thread = threading.Thread(target=poll, daemon=True)
        thread.start()
    
    def run(self):
        """Main execution loop."""
        self.running = True
        
        self.logger.info("=" * 50)
        self.logger.info("🦅 Falcon AI Ultimate Starting...")
        self.logger.info(f"Symbols: {len(self.config.SYMBOLS)}")
        self.logger.info(f"Workers: {self.config.MAX_WORKERS}")
        self.logger.info("=" * 50)
        
        # Start Telegram
        self.start_telegram()
        time.sleep(2)
        
        # Startup message
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(
                self.config.TELEGRAM_CHAT_ID,
                f"🦅 **Falcon AI Ultimate**\n✅ جاهز | نماذج: {trained}\n⚡️ بدء التحليل...",
                parse_mode='Markdown'
            )
        except:
            pass
        
        # Initial training if needed
        if not any(m.is_trained for m in self.models.values()):
            self.logger.info("No trained models found. Starting initial training...")
 
