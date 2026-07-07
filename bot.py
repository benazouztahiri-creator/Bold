#!/usr/bin/env python3
"""
Falcon AI Ultimate v2.0 - Professional Trading Signal Bot
=========================================================
Advanced ML-powered trading signal generator with per-asset training,
feature selection, probability calibration, and pattern recognition.

Key Features:
- Per-asset dedicated models with automated feature selection
- Probability calibration for accurate confidence scores
- Candlestick pattern recognition
- Support/Resistance detection
- Multi-threaded parallel analysis
- Advanced trend filtering
- Smart signal cooldown per symbol
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
import queue
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import pickle

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from scipy.signal import argrelextrema

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from sklearn.model_selection import (
    train_test_split, cross_val_score, TimeSeriesSplit, GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, mutual_info_classif, RFECV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

import telebot
from telebot import types
import joblib
import shutil

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    """Central configuration for Falcon AI Ultimate."""
    
    # Telegram
    TELEGRAM_TOKEN: str = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
    TELEGRAM_CHAT_ID: str = '7553333305'
    
    # Trading
    TRADE_DURATION_MINUTES: int = 10
    TIMEFRAMES: List[str] = field(default_factory=lambda: ['5m', '15m'])
    SCAN_INTERVAL_MINUTES: int = 5
    
    # Symbols per category (dedicated model for each)
    SYMBOLS: Dict[str, List[str]] = field(default_factory=lambda: {
        'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 
                  'NZDUSD=X', 'EURGBP=X', 'EURJPY=X'],
        'crypto': ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'BNB-USD'],
        'metals': ['GC=F', 'SI=F', 'XAUUSD=X']
    })
    
    # ML Settings
    CONFIDENCE_THRESHOLD: float = 0.70  # Minimum 70% confidence
    RETRAINING_INTERVAL_HOURS: int = 24
    MIN_TRAINING_SAMPLES: int = 1000
    TRAINING_PERIOD: str = '6mo'  # 6 months minimum
    
    # Feature Selection
    MAX_FEATURES: int = 50
    FEATURE_SELECTION_METHOD: str = 'mutual_info'  # mutual_info, rfecv, l1
    
    # Cross Validation
    CV_FOLDS: int = 5
    VALIDATION_SIZE: float = 0.15
    TEST_SIZE: float = 0.15
    
    # Probability Calibration
    CALIBRATION_METHOD: str = 'isotonic'  # isotonic, sigmoid
    CALIBRATION_CV: int = 5
    
    # Database
    DB_PATH: str = 'falcon_trading.db'
    MODELS_DIR: str = 'models'
    BACKUP_DIR: str = 'backups'
    CACHE_DIR: str = 'cache'
    
    # Charts
    CHART_CANDLES: int = 60
    CHART_DPI: int = 150
    CHART_FIGSIZE: Tuple[int, int] = (16, 10)
    
    # Performance
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5
    MAX_WORKERS: int = 8
    
    # Signal Cooldown
    SIGNAL_COOLDOWN_MINUTES: int = 15  # Per symbol
    SIGNAL_COOLDOWN_AFTER_WIN: int = 30
    SIGNAL_COOLDOWN_AFTER_LOSS: int = 60
    
    # Paths
    LOG_FILE: str = 'falcon_bot.log'
    
    # Daily Summary
    DAILY_SUMMARY_HOUR: int = 22  # Send summary at 10 PM
    PERFORMANCE_UPDATE_INTERVAL: int = 4  # Hours

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    """Professional logging with rotation and formatting."""
    logger = logging.getLogger('FalconAI')
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

# ============================================================================
# DATABASE MANAGER (Enhanced)
# ============================================================================

class DatabaseManager:
    """Advanced database manager with backup functionality."""
    
    def __init__(self, db_path: str, logger: logging.Logger):
        self.db_path = db_path
        self.logger = logger
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize all database tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Signals table with enhanced fields
                cursor.execute('''
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
                        signal_hash TEXT UNIQUE,
                        candle_close_time DATETIME
                    )
                ''')
                
                # Model registry
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS model_registry (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL UNIQUE,
                        model_version TEXT NOT NULL,
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
                        trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        next_training_at DATETIME,
                        training_duration_seconds REAL,
                        is_active INTEGER DEFAULT 1
                    )
                ''')
                
                # Performance tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        period TEXT NOT NULL,
                        total_signals INTEGER DEFAULT 0,
                        wins INTEGER DEFAULT 0,
                        losses INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0.0,
                        avg_confidence REAL DEFAULT 0.0,
                        avg_pnl REAL DEFAULT 0.0,
                        best_symbol TEXT,
                        worst_symbol TEXT,
                        total_pnl REAL DEFAULT 0.0,
                        sharpe_ratio REAL DEFAULT 0.0,
                        max_drawdown REAL DEFAULT 0.0,
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Create indexes
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_result ON signals(result)')
                cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_time ON signals(entry_time)')
                
                conn.commit()
                self.logger.info("Database initialized with enhanced schema")
                
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}", exc_info=True)
            raise
    
    def save_signal(self, signal_data: Dict[str, Any]) -> Optional[int]:
        """Save signal with duplicate prevention."""
        try:
            # Generate unique hash
            hash_str = f"{signal_data['symbol']}_{signal_data['direction']}_{signal_data['entry_time']}"
            signal_hash = hashlib.md5(hash_str.encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check for existing signal
                cursor.execute('SELECT id FROM signals WHERE signal_hash = ?', (signal_hash,))
                if cursor.fetchone():
                    self.logger.debug(f"Duplicate signal prevented: {signal_hash}")
                    return None
                
                cursor.execute('''
                    INSERT INTO signals 
                    (symbol, direction, entry_price, confidence, m5_analysis, m15_analysis,
                     trend_filter, patterns_detected, models_agreement, expiry_time, 
                     model_version, signal_hash, candle_close_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data['symbol'],
                    signal_data['direction'],
                    signal_data['entry_price'],
                    signal_data['confidence'],
                    signal_data.get('m5_analysis'),
                    signal_data.get('m15_analysis'),
                    signal_data.get('trend_filter'),
                    signal_data.get('patterns_detected'),
                    signal_data.get('models_agreement'),
                    signal_data['expiry_time'],
                    signal_data.get('model_version'),
                    signal_hash,
                    signal_data.get('candle_close_time')
                ))
                
                signal_id = cursor.lastrowid
                conn.commit()
                
                self.logger.info(f"Signal saved: ID={signal_id}, {signal_data['symbol']} {signal_data['direction']}")
                return signal_id
                
        except Exception as e:
            self.logger.error(f"Failed to save signal: {e}", exc_info=True)
            return None
    
    def check_active_signal(self, symbol: str) -> bool:
        """Check if there's an active trade for this symbol."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) FROM signals
                    WHERE symbol = ? AND result = 'PENDING'
                    AND expiry_time > datetime('now', 'localtime')
                ''', (symbol,))
                return cursor.fetchone()[0] > 0
        except Exception as e:
            self.logger.error(f"Error checking active signal: {e}")
            return True  # Conservative approach
    
    def check_recent_signal(self, symbol: str, direction: str, minutes: int) -> bool:
        """Check if similar signal was sent within cooldown period."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cutoff = datetime.now() - timedelta(minutes=minutes)
                cursor.execute('''
                    SELECT COUNT(*) FROM signals
                    WHERE symbol = ? AND direction = ?
                    AND entry_time > ?
                ''', (symbol, direction, cutoff))
                return cursor.fetchone()[0] > 0
        except Exception as e:
            self.logger.error(f"Error checking recent signal: {e}")
            return True
    
    def update_signal_result(self, signal_id: int, exit_price: float, result: str, pnl: float) -> None:
        """Update trade result and performance metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE signals SET 
                    exit_price = ?, result = ?, pnl_percent = ?, exit_time = datetime('now', 'localtime')
                    WHERE id = ?
                ''', (exit_price, result, pnl, signal_id))
                
                # Update daily performance
                today = datetime.now().strftime('%Y-%m-%d')
                cursor.execute('''
                    INSERT INTO performance (period, total_signals, wins, losses)
                    VALUES (?, 1, ?, ?)
                    ON CONFLICT(period) DO UPDATE SET
                    total_signals = total_signals + 1,
                    wins = wins + ?,
                    losses = losses + ?
                ''', (today, 
                      1 if result == 'WIN' else 0,
                      1 if result == 'LOSS' else 0,
                      1 if result == 'WIN' else 0,
                      1 if result == 'LOSS' else 0))
                
                conn.commit()
                self.logger.info(f"Trade {signal_id} result: {result}, PnL: {pnl:.2f}%")
                
        except Exception as e:
            self.logger.error(f"Failed to update signal result: {e}", exc_info=True)
    
    def save_model_metrics(self, symbol: str, metrics: Dict[str, Any]) -> None:
        """Save model training metrics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO model_registry 
                    (symbol, model_version, features_count, training_samples, 
                     accuracy, precision_score, recall_score, f1_score,
                     brier_score, calibration_error, selected_features, 
                     feature_importance, trained_at, next_training_at, 
                     training_duration_seconds, is_active)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now', 'localtime'), 
                            datetime('now', 'localtime', '+24 hours'), ?, 1)
                ''', (
                    symbol,
                    metrics.get('model_version', 'v1.0'),
                    metrics.get('features_count', 0),
                    metrics.get('training_samples', 0),
                    metrics.get('accuracy', 0),
                    metrics.get('precision', 0),
                    metrics.get('recall', 0),
                    metrics.get('f1_score', 0),
                    metrics.get('brier_score', 0),
                    metrics.get('calibration_error', 0),
                    json.dumps(metrics.get('selected_features', [])),
                    json.dumps(metrics.get('feature_importance', {})),
                    metrics.get('training_duration', 0)
                ))
                conn.commit()
                self.logger.info(f"Model metrics saved for {symbol}")
                
        except Exception as e:
            self.logger.error(f"Failed to save model metrics: {e}", exc_info=True)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total stats
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN result='LOSS' THEN 1 ELSE 0 END) as losses,
                        AVG(confidence) as avg_confidence,
                        AVG(CASE WHEN result != 'PENDING' THEN pnl_percent END) as avg_pnl
                    FROM signals WHERE result != 'PENDING'
                ''')
                totals = cursor.fetchone()
                
                # Per symbol stats
                cursor.execute('''
                    SELECT symbol,
                        COUNT(*) as total,
                        SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                        AVG(confidence) as avg_confidence,
                        AVG(pnl_percent) as avg_pnl
                    FROM signals WHERE result != 'PENDING'
                    GROUP BY symbol HAVING total >= 3
                    ORDER BY wins * 1.0 / total DESC
                ''')
                symbol_stats = cursor.fetchall()
                
                return {
                    'total_signals': totals[0] or 0,
                    'wins': totals[1] or 0,
                    'losses': totals[2] or 0,
                    'win_rate': (totals[1] or 0) / (totals[0] or 1),
                    'avg_confidence': totals[3] or 0,
                    'avg_pnl': totals[4] or 0,
                    'best_symbol': symbol_stats[0] if symbol_stats else None,
                    'worst_symbol': symbol_stats[-1] if symbol_stats else None
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get performance summary: {e}")
            return {}
    
    def backup_database(self) -> bool:
        """Create automatic database backup."""
        try:
            os.makedirs(self.db_path.replace('.db', '_backups'), exist_ok=True)
            backup_path = f"backups/falcon_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            shutil.copy2(self.db_path.replace('.db', '.db'), backup_path)
            self.logger.info(f"Database backed up to {backup_path}")
            return True
        except Exception as e:
            self.logger.error(f"Database backup failed: {e}")
            return False

# ============================================================================
# TECHNICAL INDICATORS (Enhanced)
# ============================================================================

class TechnicalAnalyzer:
    """Advanced technical analysis with pattern recognition."""
    
    @staticmethod
    def detect_support_resistance(df: pd.DataFrame, window: int = 20, sensitivity: int = 3) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels using local extrema.
        
        Args:
            df: DataFrame with OHLCV data
            window: Window for local extrema detection
            sensitivity: Number of touches to confirm level
            
        Returns:
            Dictionary with support and resistance levels
        """
        try:
            highs = df['High'].values
            lows = df['Low'].values
            closes = df['Close'].values
            
            # Find local maxima and minima
            local_max_idx = argrelextrema(highs, np.greater, order=window)[0]
            local_min_idx = argrelextrema(lows, np.less, order=window)[0]
            
            resistance_levels = []
            support_levels = []
            
            # Cluster nearby levels
            if len(local_max_idx) > 0:
                resistance_clusters = TechnicalAnalyzer._cluster_levels(highs[local_max_idx], sensitivity)
                for cluster in resistance_clusters:
                    if len(cluster) >= sensitivity:
                        resistance_levels.append(np.mean(cluster))
            
            if len(local_min_idx) > 0:
                support_clusters = TechnicalAnalyzer._cluster_levels(lows[local_min_idx], sensitivity)
                for cluster in support_clusters:
                    if len(cluster) >= sensitivity:
                        support_levels.append(np.mean(cluster))
            
            return {
                'resistance': sorted(resistance_levels, reverse=True)[:3],
                'support': sorted(support_levels)[:3]
            }
            
        except Exception as e:
            return {'resistance': [], 'support': []}
    
    @staticmethod
    def _cluster_levels(levels: np.ndarray, sensitivity: int) -> List[np.ndarray]:
        """Cluster nearby price levels."""
        if len(levels) < 2:
            return [levels] if len(levels) > 0 else []
        
        sorted_levels = np.sort(levels)
        clusters = []
        current_cluster = [sorted_levels[0]]
        
        for i in range(1, len(sorted_levels)):
            if abs(sorted_levels[i] - current_cluster[-1]) / current_cluster[-1] < 0.01:
                current_cluster.append(sorted_levels[i])
            else:
                clusters.append(np.array(current_cluster))
                current_cluster = [sorted_levels[i]]
        
        clusters.append(np.array(current_cluster))
        return clusters
    
    @staticmethod
    def detect_candlestick_patterns(df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect candlestick patterns.
        
        Returns:
            List of detected patterns with type and strength
        """
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        # Get recent candles
        c1 = df.iloc[-1]  # Current
        c2 = df.iloc[-2]  # Previous
        c3 = df.iloc[-3]  # Two before
        
        body1 = abs(c1['Close'] - c1['Open'])
        body2 = abs(c2['Close'] - c2['Open'])
        body3 = abs(c3['Close'] - c3['Open'])
        
        upper_wick1 = c1['High'] - max(c1['Close'], c1['Open'])
        lower_wick1 = min(c1['Close'], c1['Open']) - c1['Low']
        
        upper_wick2 = c2['High'] - max(c2['Close'], c2['Open'])
        lower_wick2 = min(c2['Close'], c2['Open']) - c2['Low']
        
        # Pin Bar (Hammer / Shooting Star)
        if body1 > 0:
            total_range1 = c1['High'] - c1['Low']
            if total_range1 > 0:
                # Hammer (bullish)
                if lower_wick1 > body1 * 2 and upper_wick1 < body1 * 0.3:
                    if c1['Close'] > c1['Open']:
                        patterns.append({'pattern': 'Hammer', 'strength': 'strong', 'direction': 'BUY'})
                
                # Shooting Star (bearish)
                if upper_wick1 > body1 * 2 and lower_wick1 < body1 * 0.3:
                    if c1['Close'] < c1['Open']:
                        patterns.append({'pattern': 'Shooting Star', 'strength': 'strong', 'direction': 'SELL'})
        
        # Engulfing
        if body2 > 0 and body1 > 0:
            # Bullish Engulfing
            if c2['Close'] < c2['Open'] and c1['Close'] > c1['Open']:
                if c1['Open'] <= c2['Close'] and c1['Close'] >= c2['Open']:
                    patterns.append({'pattern': 'Bullish Engulfing', 'strength': 'strong', 'direction': 'BUY'})
            
            # Bearish Engulfing
            if c2['Close'] > c2['Open'] and c1['Close'] < c1['Open']:
                if c1['Open'] >= c2['Close'] and c1['Close'] <= c2['Open']:
                    patterns.append({'pattern': 'Bearish Engulfing', 'strength': 'strong', 'direction': 'SELL'})
        
        # Doji
        if body1 < (c1['High'] - c1['Low']) * 0.1:
            if abs(upper_wick1 - lower_wick1) < body1:
                patterns.append({'pattern': 'Doji', 'strength': 'moderate', 'direction': 'NEUTRAL'})
            elif upper_wick1 > lower_wick1 * 2:
                patterns.append({'pattern': 'Gravestone Doji', 'strength': 'moderate', 'direction': 'SELL'})
            elif lower_wick1 > upper_wick1 * 2:
                patterns.append({'pattern': 'Dragonfly Doji', 'strength': 'moderate', 'direction': 'BUY'})
        
        # Morning/Evening Star
        if len(df) >= 3:
            if (c3['Close'] < c3['Open'] and  # First red
                body2 < body3 * 0.3 and  # Small middle
                c1['Close'] > c1['Open'] and  # Third green
                c1['Close'] > (c3['Open'] + c3['Close']) / 2):  # Close above midpoint
                patterns.append({'pattern': 'Morning Star', 'strength': 'strong', 'direction': 'BUY'})
            
            if (c3['Close'] > c3['Open'] and  # First green
                body2 < body3 * 0.3 and  # Small middle
                c1['Close'] < c1['Open'] and  # Third red
                c1['Close'] < (c3['Open'] + c3['Close']) / 2):  # Close below midpoint
                patterns.append({'pattern': 'Evening Star', 'strength': 'strong', 'direction': 'SELL'})
        
        return patterns
    
    @staticmethod
    def calculate_trend_filter(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate multi-timeframe trend filter.
        
        Returns:
            Dictionary with trend direction and strength
        """
        if len(df) < 50:
            return {'trend': 'NEUTRAL', 'strength': 0}
        
        closes = df['Close']
        
        # EMAs
        ema20 = closes.ewm(span=20, adjust=False).mean().iloc[-1]
        ema50 = closes.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = closes.ewm(span=200, adjust=False).mean().iloc[-1] if len(df) >= 200 else ema50
        
        current_price = closes.iloc[-1]
        
        # ADX for trend strength
        adx_data = TechnicalAnalyzer._calculate_adx(df)
        adx_value = adx_data['adx'].iloc[-1] if not adx_data['adx'].empty else 0
        
        # Trend score
        score = 0
        reasons = []
        
        if current_price > ema20:
            score += 1
            reasons.append('Price above EMA20')
        else:
            score -= 1
        
        if current_price > ema50:
            score += 1
            reasons.append('Price above EMA50')
        else:
            score -= 1
        
        if current_price > ema200:
            score += 1
            reasons.append('Price above EMA200')
        else:
            score -= 1
        
        if ema20 > ema50:
            score += 1
            reasons.append('EMA20 > EMA50')
        else:
            score -= 1
        
        # ADX strength
        if adx_value > 25:
            if score > 0:
                score += 1
            elif score < 0:
                score -= 1
        
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
        
        return {
            'trend': trend,
            'strength': strength,
            'score': score,
            'adx': adx_value,
            'reasons': reasons
        }
    
    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> Dict[str, pd.Series]:
        """Calculate ADX indicator."""
        high = df['High']
        low = df['Low']
        close = df['Close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = -minus_dm
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = dx.ewm(span=period, adjust=False).mean()
        
        return {'adx': adx, 'plus_di': plus_di, 'minus_di': minus_di}
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive set of technical indicators."""
        features = pd.DataFrame(index=df.index)
        
        # Price features
        features['returns_1'] = df['Close'].pct_change(1)
        features['returns_3'] = df['Close'].pct_change(3)
        features['returns_5'] = df['Close'].pct_change(5)
        features['returns_10'] = df['Close'].pct_change(10)
        features['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Volatility
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = df['Close'].pct_change().rolling(period).std()
            features[f'high_low_range_{period}'] = (df['High'] - df['Low']).rolling(period).mean()
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            if len(df) >= period:
                features[f'sma_{period}'] = df['Close'].rolling(period).mean()
                features[f'ema_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
                features[f'price_sma_{period}'] = df['Close'] / features[f'sma_{period}'] - 1
                features[f'price_ema_{period}'] = df['Close'] / features[f'ema_{period}'] - 1
        
        # RSI
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = TechnicalAnalyzer._calculate_rsi(df['Close'], period)
        
        # MACD
        macd = TechnicalAnalyzer._calculate_macd(df['Close'])
        features['macd'] = macd['macd']
        features['macd_signal'] = macd['signal']
        features['macd_hist'] = macd['histogram']
        
        # Bollinger Bands
        bb = TechnicalAnalyzer._calculate_bollinger(df['Close'])
        features['bb_percent_b'] = bb['percent_b']
        features['bb_width'] = bb['width']
        
        # Stochastic
        stoch = TechnicalAnalyzer._calculate_stochastic(df)
        features['stoch_k'] = stoch['k']
        features['stoch_d'] = stoch['d']
        
        # ATR
        features['atr'] = TechnicalAnalyzer._calculate_atr(df)
        features['atr_percent'] = features['atr'] / df['Close']
        
        # CCI
        features['cci'] = TechnicalAnalyzer._calculate_cci(df, 20)
        
        # Williams %R
        features['williams_r'] = TechnicalAnalyzer._calculate_williams_r(df)
        
        # ROC
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = TechnicalAnalyzer._calculate_roc(df['Close'], period)
        
        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        
        # Volume (if available)
        if 'Volume' in df.columns:
            features['volume_change'] = df['Volume'].pct_change()
            features['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
            features['volume_trend'] = df['Volume'].rolling(5).mean() / df['Volume'].rolling(20).mean()
        
        # Clean up
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return features
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def _calculate_macd(prices: pd.Series) -> Dict[str, pd.Series]:
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return {'macd': macd_line, 'signal': signal_line, 'histogram': macd_line - signal_line}
    
    @staticmethod
    def _calculate_bollinger(prices: pd.Series, period: int = 20, std: float = 2) -> Dict[str, pd.Series]:
        sma = prices.rolling(period).mean()
        std_dev = prices.rolling(period).std()
        upper = sma + std * std_dev
        lower = sma - std * std_dev
        percent_b = (prices - lower) / (upper - lower + 1e-8)
        width = (upper - lower) / sma
        return {'upper': upper, 'middle': sma, 'lower': lower, 'percent_b': percent_b, 'width': width}
    
    @staticmethod
    def _calculate_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        low_min = df['Low'].rolling(k_period).min()
        high_max = df['High'].rolling(k_period).max()
        k = 100 * (df['Close'] - low_min) / (high_max - low_min + 1e-8)
        d = k.rolling(d_period).mean()
        return {'k': k, 'd': d}
    
    @staticmethod
    def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_low = df['High'] - df['Low']
        high_close = abs(df['High'] - df['Close'].shift())
        low_close = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def _calculate_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        sma = tp.rolling(period).mean()
        mad = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - sma) / (0.015 * mad + 1e-8)
    
    @staticmethod
    def _calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hh = df['High'].rolling(period).max()
        ll = df['Low'].rolling(period).min()
        return -100 * (hh - df['Close']) / (hh - ll + 1e-8)
    
    @staticmethod
    def _calculate_roc(prices: pd.Series, period: int) -> pd.Series:
        return ((prices - prices.shift(period)) / prices.shift(period)) * 100

# ============================================================================
# PER-ASSET ENSEMBLE MODEL (Enhanced)
# ============================================================================

class PerAssetModel:
    """Dedicated ensemble model for each trading symbol."""
    
    def __init__(self, symbol: str, config: Config, logger: logging.Logger):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        
        self.models = {}
        self.calibrators = {}
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.selected_features = []
        self.feature_importance = {}
        self.is_trained = False
        self.training_metrics = {}
        self.model_version = None
        
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize ensemble models with optimized hyperparameters."""
        self.models = {
            'xgboost': xgb.XGBClassifier(
                n_estimators=300, learning_rate=0.02, max_depth=5,
                min_child_weight=4, subsample=0.75, colsample_bytree=0.75,
                gamma=0.2, reg_alpha=0.5, reg_lambda=2.0,
                random_state=42, n_jobs=-1, verbosity=0
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=300, learning_rate=0.02, max_depth=5,
                num_leaves=31, min_child_samples=25, subsample=0.75,
                colsample_bytree=0.75, reg_alpha=0.5, reg_lambda=2.0,
                random_state=42, n_jobs=-1, verbose=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=300, learning_rate=0.02, depth=5,
                l2_leaf_reg=5, random_seed=42, verbose=False, thread_count=-1
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.02, max_depth=4,
                min_samples_split=10, min_samples_leaf=5,
                subsample=0.75, random_state=42
            )
        }
        
        # Adaptive weights (will be updated based on performance)
        self.model_weights = {
            'xgboost': 0.30,
            'lightgbm': 0.25,
            'catboost': 0.25,
            'gradient_boost': 0.20
        }
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """
        Automated feature selection using mutual information and RFECV.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Selecting features for {self.symbol} from {len(X.columns)} features")
        
        # Remove constant and highly correlated features
        constant_features = [col for col in X.columns if X[col].std() < 1e-8]
        X_clean = X.drop(columns=constant_features)
        
        # Remove highly correlated features
        corr_matrix = X_clean.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
        X_clean = X_clean.drop(columns=high_corr_features)
        
        # Mutual Information Selection
        mi_scores = mutual_info_classif(X_clean, y, random_state=42)
        mi_df = pd.DataFrame({'feature': X_clean.columns, 'mi_score': mi_scores})
        mi_df = mi_df.sort_values('mi_score', ascending=False)
        
        # Select top features
        top_k = min(self.config.MAX_FEATURES, len(mi_df))
        selected_by_mi = mi_df.head(top_k)['feature'].tolist()
        
        # RFECV for final selection
        if len(selected_by_mi) > 20:
            base_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rfecv = RFECV(
                estimator=base_model,
                step=5,
                cv=TimeSeriesSplit(n_splits=3),
                scoring='f1',
                min_features_to_select=max(15, top_k // 3),
                n_jobs=-1
            )
            
            try:
                rfecv.fit(X_clean[selected_by_mi], y)
                final_features = [selected_by_mi[i] for i in range(len(selected_by_mi)) 
                                 if rfecv.support_[i]]
                
                if len(final_features) < 10:
                    final_features = selected_by_mi[:20]
            except Exception as e:
                self.logger.warning(f"RFECV failed: {e}, using MI selection")
                final_features = selected_by_mi[:30]
        else:
            final_features = selected_by_mi
        
        self.selected_features = final_features
        self.logger.info(f"Selected {len(final_features)} features for {self.symbol}")
        
        return final_features
    
    def train(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """
        Train per-asset model with feature selection and calibration.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Training metrics or None if failed
        """
        try:
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                self.logger.warning(f"Insufficient data for {self.symbol}: {len(df)} samples")
                return None
            
            self.logger.info(f"Training model for {self.symbol} with {len(df)} samples")
            start_time = time.time()
            
            # Prepare features and target
            features_df = TechnicalAnalyzer.calculate_all_indicators(df)
            
            # Create target (future price movement)
            future_returns = df['Close'].shift(-3) / df['Close'] - 1
            target = (future_returns > 0.001).astype(int)  # Buy if > 0.1% gain
            
            # Remove NaN
            valid_idx = ~(features_df.isna().any(axis=1) | target.isna() | future_returns.isna())
            X = features_df[valid_idx]
            y = target[valid_idx]
            
            if len(X) < 100:
                return None
            
            # Feature selection
            if not self.selected_features:
                self.select_features(X, y)
            
            X_selected = X[self.selected_features]
            
            # Split data
            X_temp, X_test, y_temp, y_test = train_test_split(
                X_selected, y, test_size=self.config.TEST_SIZE, shuffle=False
            )
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, 
                test_size=self.config.VALIDATION_SIZE / (1 - self.config.TEST_SIZE),
                shuffle=False
            )
            
            self.logger.info(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
            
            # Scale features
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train each model
            model_predictions = {}
            model_probas = {}
            
            for name, model in self.models.items():
                self.logger.debug(f"Training {name} for {self.symbol}")
                
                try:
                    if name == 'catboost':
                        model.fit(X_train_scaled, y_train, verbose=False)
                    elif name == 'lightgbm':
                        model.fit(X_train_scaled, y_train, eval_set=[(X_val_scaled, y_val)])
                    else:
                        model.fit(X_train_scaled, y_train)
                    
                    # Get predictions
                    val_pred = model.predict(X_val_scaled)
                    val_proba = model.predict_proba(X_val_scaled)[:, 1]
                    
                    model_predictions[name] = val_pred
                    model_probas[name] = val_proba
                    
                    # Cross-validation
                    cv_scores = cross_val_score(
                        model, X_train_scaled, y_train,
                        cv=TimeSeriesSplit(n_splits=min(3, self.config.CV_FOLDS)),
                        scoring='accuracy'
                    )
                    
                    # Calibration
                    self.calibrators[name] = CalibratedClassifierCV(
                        model, method=self.config.CALIBRATION_METHOD,
                        cv=self.config.CALIBRATION_CV
                    )
                    self.calibrators[name].fit(X_train_scaled, y_train)
                    
                except Exception as e:
                    self.logger.error(f"Error training {name}: {e}")
                    continue
            
            if not model_probas:
                return None
            
            # Calculate ensemble predictions
            ensemble_proba = np.zeros(len(y_val))
            for name, proba in model_probas.items():
                ensemble_proba += self.model_weights[name] * proba
            
            ensemble_pred = (ensemble_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, ensemble_pred)
            precision = precision_score(y_val, ensemble_pred, zero_division=0)
            recall = recall_score(y_val, ensemble_pred, zero_division=0)
            f1 = f1_score(y_val, ensemble_pred, zero_division=0)
            brier = brier_score_loss(y_val, ensemble_proba)
            
            # Test set performance
            test_ensemble_proba = np.zeros(len(y_test))
            for name in model_probas.keys():
                test_proba = self.calibrators[name].predict_proba(X_test_scaled)[:, 1]
                test_ensemble_proba += self.model_weights[name] * test_proba
            
            test_pred = (test_ensemble_proba > 0.5).astype(int)
            test_accuracy = accuracy_score(y_test, test_pred)
            
            # Update model weights based on validation performance
            model_scores = {}
            for name in model_probas.keys():
                model_scores[name] = f1_score(y_val, model_predictions[name], zero_division=0)
            
            total_score = sum(model_scores.values())
            if total_score > 0:
                for name in self.model_weights:
                    self.model_weights[name] = model_scores.get(name, 0) / total_score
            
            # Save feature importance
            if 'xgboost' in self.models:
                importances = self.models['xgboost'].feature_importances_
                self.feature_importance = dict(zip(self.selected_features, importances))
            
            self.is_trained = True
            training_time = time.time() - start_time
            self.model_version = datetime.now().strftime('v%Y%m%d_%H%M%S')
            
            # Compile metrics
            self.training_metrics = {
                'model_version': self.model_version,
                'features_count': len(self.selected_features),
                'training_samples': len(X_train),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'brier_score': brier,
                'calibration_error': self._calculate_calibration_error(y_val, ensemble_proba),
                'test_accuracy': test_accuracy,
                'training_duration': training_time,
                'selected_features': self.selected_features,
                'feature_importance': self.feature_importance,
                'model_weights': self.model_weights.copy()
            }
            
            self.logger.info(f"Training complete for {self.symbol}: F1={f1:.3f}, "
                           f"Time={training_time:.1f}s, Features={len(self.selected_features)}")
            
            return self.training_metrics
            
        except Exception as e:
            self.logger.error(f"Training failed for {self.symbol}: {e}", exc_info=True)
            return None
    
    def predict(self, features_df: pd.DataFrame) -> Tuple[str, float, Dict[str, float]]:
        """
        Make calibrated ensemble prediction.
        
        Returns:
            Tuple of (direction, confidence, model_probabilities)
        """
        if not self.is_trained or not self.selected_features:
            return "NEUTRAL", 0.0, {}
        
        try:
            # Select and scale features
            available_features = [f for f in self.selected_features if f in features_df.columns]
            if len(available_features) < 10:
                return "NEUTRAL", 0.0, {}
            
            X = features_df[available_features].fillna(0).iloc[[-1]]
            X_scaled = self.scaler.transform(X)
            
            # Get calibrated predictions from each model
            model_probas = {}
            for name in self.models.keys():
                if name in self.calibrators:
                    try:
                        proba = self.calibrators[name].predict_proba(X_scaled)[0, 1]
                        model_probas[name] = float(proba)
                    except:
                        model_probas[name] = 0.5
            
            if not model_probas:
                return "NEUTRAL", 0.0, {}
            
            # Calculate weighted ensemble probability
            ensemble_proba = sum(
                self.model_weights.get(name, 0) * proba 
                for name, proba in model_probas.items()
            )
            
            # Determine direction
            if ensemble_proba > self.config.CONFIDENCE_THRESHOLD:
                direction = "BUY"
                confidence = ensemble_proba
            elif ensemble_proba < (1 - self.config.CONFIDENCE_THRESHOLD):
                direction = "SELL"
                confidence = 1 - ensemble_proba
            else:
                direction = "NEUTRAL"
                confidence = max(ensemble_proba, 1 - ensemble_proba)
            
            return direction, confidence, model_probas
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {self.symbol}: {e}")
            return "NEUTRAL", 0.0, {}
    
    def _calculate_calibration_error(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Calculate calibration error."""
        try:
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            return np.mean(np.abs(prob_true - prob_pred))
        except:
            return 1.0
    
    def save(self) ->