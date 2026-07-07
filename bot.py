#!/usr/bin/env python3
"""
Falcon AI Ultimate v4.0 - Professional Forex Bot
==================================================
- 12-24 Month Training
- Walk-Forward Validation
- Meta Model (Stacking)
- Probability Calibration
- Market Regime Detection
- Dynamic Threshold
- Weighted Probabilities
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import telebot
import joblib

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '2'

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    TELEGRAM_TOKEN: str = os.environ.get('TELEGRAM_TOKEN', '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '7553333305')
    
    TRADE_DURATION_MINUTES: int = 10
    SCAN_INTERVAL_MINUTES: int = 3
    
    SYMBOLS: List[str] = field(default_factory=lambda: [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
        'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X'
    ])
    
    # ✅ Training: 12-24 months
    TRAINING_PERIOD_1H: str = '12mo'  # ✅ 12 شهر للفريم الكبير
    TRAINING_PERIOD_15M: str = '2mo'  # ✅ شهرين للفريم المتوسط
    
    # ✅ Walk-Forward
    WALK_FORWARD_WINDOWS: int = 5
    MIN_TRAINING_SAMPLES: int = 500
    
    # ✅ Dynamic Threshold
    CONFIDENCE_THRESHOLD_INITIAL: float = 0.65
    CONFIDENCE_THRESHOLD_MIN: float = 0.55
    CONFIDENCE_THRESHOLD_MAX: float = 0.80
    
    # ✅ Market Regime
    ADX_TREND_THRESHOLD: float = 25
    VOLATILITY_HIGH_THRESHOLD: float = 1.5
    
    # ✅ Performance tracking
    PERFORMANCE_WINDOW: int = 50  # آخر 50 صفقة لتقييم الأداء
    
    RETRAINING_INTERVAL_HOURS: int = 24
    MAX_FEATURES: int = 80
    FORECAST_PERIODS: int = 5
    
    DB_PATH: str = 'falcon_trading.db'
    MODELS_DIR: str = 'models'
    
    MAX_RETRIES: int = 5
    RETRY_DELAY: int = 10
    MAX_WORKERS: int = 2
    SIGNAL_COOLDOWN_MINUTES: int = 10
    
    LOG_FILE: str = 'falcon_bot.log'

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('FalconPro')

# ============================================================================
# DATABASE WITH PERFORMANCE TRACKING
# ============================================================================

class Database:
    def __init__(self, db_path: str, logger: logging.Logger):
        self.db_path = db_path
        self.logger = logger
        self._init()
    
    def _init(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    confidence REAL, regime TEXT, entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, meta_proba REAL, signal_hash TEXT UNIQUE
                );
                CREATE TABLE IF NOT EXISTS model_performance (
                    symbol TEXT PRIMARY KEY,
                    last_50_wins INTEGER DEFAULT 0,
                    last_50_total INTEGER DEFAULT 0,
                    current_threshold REAL DEFAULT 0.65,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            conn.commit()
    
    def save_signal(self, data: Dict) -> Optional[int]:
        try:
            hash_str = f"{data['symbol']}_{data['direction']}_{datetime.now().timestamp()}"
            signal_hash = hashlib.md5(hash_str.encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO signals 
                    (symbol, direction, entry_price, confidence, regime, expiry_time, meta_proba, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data.get('regime', ''), data['expiry_time'],
                      data.get('meta_proba', 0), signal_hash))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except:
            return None
    
    def update_result(self, signal_id: int, exit_price: float, result: str, pnl: float, symbol: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE signals SET exit_price=?, result=?, pnl_percent=? WHERE id=?
            ''', (exit_price, result, pnl, signal_id))
            
            # Update performance tracker
            conn.execute('''
                INSERT INTO model_performance (symbol, last_50_wins, last_50_total, current_threshold)
                VALUES (?, ?, 1, 0.65)
                ON CONFLICT(symbol) DO UPDATE SET
                last_50_wins = CASE WHEN last_50_total >= 50 
                    THEN last_50_wins + ? - (SELECT CASE WHEN result='WIN' THEN 1 ELSE 0 END 
                        FROM signals WHERE symbol=? AND result!='PENDING' 
                        ORDER BY entry_time DESC LIMIT 1 OFFSET 49)
                    ELSE last_50_wins + ? END,
                last_50_total = CASE WHEN last_50_total >= 50 
                    THEN 50 ELSE last_50_total + 1 END
            ''', (symbol, 
                  1 if result == 'WIN' else 0, symbol,
                  1 if result == 'WIN' else 0))
            conn.commit()
    
    def get_dynamic_threshold(self, symbol: str) -> float:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute('''
                SELECT last_50_wins, last_50_total FROM model_performance WHERE symbol=?
            ''', (symbol,)).fetchone()
            
            if not row or row[1] < 10:
                return 0.65
            
            win_rate = row[0] / row[1]
            
            # Dynamic threshold based on performance
            if win_rate > 0.70:
                return 0.58  # أداء ممتاز → عتبة أقل
            elif win_rate > 0.60:
                return 0.62
            elif win_rate > 0.50:
                return 0.68
            else:
                return 0.75  # أداء ضعيف → عتبة أعلى
    
    def check_active_signal(self, symbol: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? 
                AND result='PENDING' AND expiry_time > datetime('now', 'localtime')
            ''', (symbol,)).fetchone()[0]
            return count > 0
    
    def check_recent_signal(self, symbol: str, minutes: int) -> bool:
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?
            ''', (symbol, cutoff)).fetchone()[0]
            return count > 0
    
    def get_pending_trades(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM signals WHERE result='PENDING' 
                AND expiry_time <= datetime('now', 'localtime')
            ''').fetchall()
            return [dict(r) for r in rows]
    
    def get_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM signals WHERE result!='PENDING'").fetchone()[0]
            wins = conn.execute("SELECT COUNT(*) FROM signals WHERE result='WIN'").fetchone()[0]
            return {'total': total, 'wins': wins, 'losses': total-wins,
                    'win_rate': wins/total if total > 0 else 0}

# ============================================================================
# MARKET REGIME DETECTION
# ============================================================================

class MarketRegime:
    """Detect market regime: TREND, RANGE, HIGH_VOL, LOW_VOL"""
    
    @staticmethod
    def detect(df: pd.DataFrame) -> Dict:
        if len(df) < 50:
            return {'regime': 'UNKNOWN', 'adx': 0, 'volatility_ratio': 1}
        
        c = df['Close']
        h = df['High']
        l = df['Low']
        
        # ADX
        tr1 = h - l
        tr2 = abs(h - c.shift())
        tr3 = abs(l - c.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=14).mean()
        
        plus_dm = h.diff().clip(lower=0)
        minus_dm = (-l.diff()).clip(lower=0)
        plus_di = 100 * (plus_dm.ewm(span=14).mean()) / (atr + 1e-8)
        minus_di = 100 * (minus_dm.ewm(span=14).mean()) / (atr + 1e-8)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = float(dx.ewm(span=14).mean().iloc[-1])
        
        # Volatility ratio (current vs historical)
        current_vol = c.pct_change().rolling(20).std().iloc[-1]
        historical_vol = c.pct_change().rolling(100).std().iloc[-1] if len(c) >= 100 else current_vol
        vol_ratio = current_vol / (historical_vol + 1e-8)
        
        # Bollinger width
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        bb_width = (4 * std20.iloc[-1]) / (sma20.iloc[-1] + 1e-8)
        
        # Regime classification
        if adx > 25:
            if vol_ratio > 1.3:
                regime = 'TREND_HIGH_VOL'
            else:
                regime = 'TREND_LOW_VOL'
        else:
            if vol_ratio > 1.3:
                regime = 'RANGE_HIGH_VOL'
            else:
                regime = 'RANGE_LOW_VOL'
        
        return {
            'regime': regime,
            'adx': round(adx, 1),
            'volatility_ratio': round(vol_ratio, 2),
            'bb_width': round(bb_width, 4),
            'is_trend': adx > 25,
            'is_volatile': vol_ratio > 1.3
        }

# ============================================================================
# ADVANCED FEATURES (80+)
# ============================================================================

def calculate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """80+ professional features."""
    f = pd.DataFrame(index=df.index)
    c, h, l, o = df['Close'], df['High'], df['Low'], df['Open']
    v = df.get('Volume', pd.Series(1, index=df.index))
    
    # Returns (6)
    for p in [1, 2, 3, 5, 10, 20]:
        f[f'ret_{p}'] = c.pct_change(p)
    f['log_ret'] = np.log(c / c.shift(1))
    f['hl_ratio'] = (h - l) / (c + 1e-8)
    f['close_pos'] = (c - l) / (h - l + 1e-8)
    f['gap'] = (o - c.shift(1)) / c.shift(1)
    
    # Moving Averages (18)
    for p in [5, 10, 20, 50, 100, 200]:
        if len(df) >= p:
            f[f'sma_{p}'] = c.rolling(p).mean()
            f[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
            f[f'dist_sma_{p}'] = (c - f[f'sma_{p}']) / (f[f'sma_{p}'] + 1e-8)
    
    # RSI (3)
    for p in [7, 14, 21]:
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).rolling(p).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(p).mean()
        f[f'rsi_{p}'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    # MACD (3)
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    f['macd'] = ema12 - ema26
    f['macd_signal'] = f['macd'].ewm(span=9).mean()
    f['macd_hist'] = f['macd'] - f['macd_signal']
    
    # Bollinger (4)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f['bb_upper'] = sma20 + 2 * std20
    f['bb_lower'] = sma20 - 2 * std20
    f['bb_pos'] = (c - f['bb_lower']) / (f['bb_upper'] - f['bb_lower'] + 1e-8)
    f['bb_width'] = (f['bb_upper'] - f['bb_lower']) / (sma20 + 1e-8)
    
    # ATR (3)
    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    f['atr_14'] = tr.ewm(span=14).mean()
    f['atr_pct'] = f['atr_14'] / (c + 1e-8)
    f['atr_ratio'] = f['atr_14'] / f['atr_14'].rolling(50).mean()
    
    # Stochastic (2)
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    f['stoch_k'] = 100 * (c - low14) / (high14 - low14 + 1e-8)
    f['stoch_d'] = f['stoch_k'].rolling(3).mean()
    
    # CCI (1)
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    f['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-8)
    
    # Williams %R (1)
    hh14 = h.rolling(14).max()
    ll14 = l.rolling(14).min()
    f['williams_r'] = -100 * (hh14 - c) / (hh14 - ll14 + 1e-8)
    
    # ADX (3)
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    atr14 = tr.ewm(span=14).mean()
    plus_di = 100 * (plus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    minus_di = 100 * (minus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    f['adx'] = dx.ewm(span=14).mean()
    f['di_plus'] = plus_di
    f['di_minus'] = minus_di
    
    # MFI (1)
    tp_mfi = (h + l + c) / 3
    mf = tp_mfi * v
    pos_mf = mf.where(tp_mfi > tp_mfi.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(tp_mfi < tp_mfi.shift(1), 0).rolling(14).sum()
    f['mfi'] = 100 - (100 / (1 + pos_mf / (neg_mf + 1e-8)))
    
    # ROC & Momentum (6)
    for p in [5, 10, 20]:
        f[f'roc_{p}'] = (c - c.shift(p)) / (c.shift(p) + 1e-8) * 100
        f[f'mom_{p}'] = c - c.shift(p)
    
    # Volatility (3)
    for p in [5, 10, 20]:
        f[f'vol_{p}'] = c.pct_change().rolling(p).std()
    
    # Donchian (2)
    f['dc_upper'] = h.rolling(20).max()
    f['dc_lower'] = l.rolling(20).min()
    f['dc_pos'] = (c - f['dc_lower']) / (f['dc_upper'] - f['dc_lower'] + 1e-8)
    
    # Keltner (2)
    kc_mid = c.ewm(span=20).mean()
    f['kc_upper'] = kc_mid + 1.5 * f['atr_14']
    f['kc_lower'] = kc_mid - 1.5 * f['atr_14']
    f['kc_pos'] = (c - f['kc_lower']) / (f['kc_upper'] - f['kc_lower'] + 1e-8)
    
    # SuperTrend (1)
    f['supertrend'] = ((c > (h+l)/2 + 2*f['atr_14']).astype(int) - 
                       (c < (h+l)/2 - 2*f['atr_14']).astype(int))
    
    # Candlestick (4)
    f['body_size'] = abs(c - o) / (h - l + 1e-8)
    f['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-8)
    f['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-8)
    f['candle_type'] = (c > o).astype(int) - (c < o).astype(int)
    
    # Support/Resistance (4)
    for p in [20, 50]:
        f[f'high_{p}d'] = c / (h.rolling(p).max() + 1e-8)
        f[f'low_{p}d'] = c / (l.rolling(p).min() + 1e-8)
    
    # Trend Strength (2)
    f['trend_str'] = c.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    f['hh_ll_ratio'] = (h.rolling(20).max() - c) / (c - l.rolling(20).min() + 1e-8)
    
    # Volume (2)
    f['vol_ratio'] = v / (v.rolling(20).mean() + 1e-8)
    f['vol_trend'] = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-8)
    
    return f.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# ============================================================================
# SMART TARGET
# ============================================================================

def create_smart_target(df: pd.DataFrame, periods: int) -> pd.Series:
    """Target based on meaningful ATR movement."""
    atr = df['High'] - df['Low']
    atr = atr.rolling(14).mean()
    
    future_price = df['Close'].shift(-periods)
    price_change = future_price - df['Close']
    
    threshold = atr * 0.5  # 0.5 × ATR
    
    buy_signal = price_change > threshold
    sell_signal = price_change < -threshold
    
    target = pd.Series(np.nan, index=df.index)
    target[buy_signal] = 1
    target[sell_signal] = 0
    
    return target

# ============================================================================
# PRO ENSEMBLE MODEL WITH META-STACKING
# ============================================================================

class ProEnsembleModel:
    """
    Professional ensemble with:
    - Walk-Forward Validation
    - Meta Model (Stacking)
    - Probability Calibration
    - Dynamic Threshold
    """
    
    def __init__(self, symbol: str, config: Config, logger: logging.Logger):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        
        self.base_models = {}
        self.meta_model = None
        self.calibrators = {}
        self.scaler = RobustScaler()
        self.selected_features = []
        self.is_trained = False
        self.version = None
        self.walk_forward_score = 0
        self.current_threshold = config.CONFIDENCE_THRESHOLD_INITIAL
    
    def _init_base_models(self):
        self.base_models = {
            'xgboost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=5,
                                          random_state=42, n_jobs=2, verbosity=0, tree_method='hist'),
            'catboost': CatBoostClassifier(iterations=200, learning_rate=0.03, depth=5,
                                            random_seed=42, verbose=False, thread_count=2, allow_writing_files=False),
            'lightgbm': lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, max_depth=5,
                                            random_state=42, n_jobs=2, verbose=-1),
            'randomforest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=2)
        }
    
    def _walk_forward_train(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Walk-forward validation."""
        tscv = TimeSeriesSplit(n_splits=self.config.WALK_FORWARD_WINDOWS)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            X_train_s = self.scaler.fit_transform(X_train)
            X_val_s = self.scaler.transform(X_val)
            
            # Train base models
            base_preds_val = np.zeros((len(X_val), len(self.base_models)))
            
            for i, (name, model) in enumerate(self.base_models.items()):
                try:
                    if name == 'catboost':
                        model.fit(X_train_s, y_train, verbose=False)
                    elif name == 'lightgbm':
                        model.fit(X_train_s, y_train)
                    else:
                        model.fit(X_train_s, y_train)
                    base_preds_val[:, i] = model.predict_proba(X_val_s)[:, 1]
                except:
                    base_preds_val[:, i] = 0.5
            
            # Meta model
            meta = LogisticRegression()
            meta.fit(base_preds_val, y_val)
            meta_pred = meta.predict(base_preds_val)
            scores.append(accuracy_score(y_val, meta_pred))
        
        return np.mean(scores)
    
    def train(self, df: pd.DataFrame) -> bool:
        try:
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                self.logger.warning(f"{self.symbol}: بيانات غير كافية ({len(df)})")
                return False
            
            self.logger.info(f"🎓 {self.symbol}: {len(df)} عينة - تدريب متقدم...")
            
            features = calculate_advanced_features(df)
            target = create_smart_target(df, self.config.FORECAST_PERIODS)
            
            valid = ~(features.isna().any(axis=1) | target.isna())
            X = features[valid]
            y = target[valid]
            
            neutral_pct = target.isna().sum() / len(target) * 100
            self.logger.info(f"📊 {self.symbol}: {len(X)} صالحة ({neutral_pct:.0f}% محايدة)")
            
            if len(X) < 200:
                return False
            
            # Feature selection
            mi = mutual_info_classif(X, y, random_state=42)
            scores = sorted(zip(X.columns, mi), key=lambda x: x[1], reverse=True)
            self.selected_features = [s[0] for s in scores[:self.config.MAX_FEATURES]]
            X = X[self.selected_features]
            
            # Walk-Forward validation
            self._init_base_models()
            self.logger.info(f"🔄 {self.symbol}: Walk-Forward ({self.config.WALK_FORWARD_WINDOWS} windows)...")
            self.walk_forward_score = self._walk_forward_train(X, y)
            self.logger.info(f"📈 {self.symbol}: Walk-Forward Score = {self.walk_forward_score:.3f}")
            
            # Final train on all data
            X_s = self.scaler.fit_transform(X)
            
            base_preds_all = np.zeros((len(X), len(self.base_models)))
            
            for i, (name, model) in enumerate(self.base_models.items()):
                try:
                    if name == 'catboost':
                        model.fit(X_s, y, verbose=False)
                    elif name == 'lightgbm':
                        model.fit(X_s, y)
                    else:
                        model.fit(X_s, y)
                    base_preds_all[:, i] = model.predict_proba(X_s)[:, 1]
                    
                    # ✅ Probability Calibration
                    self.calibrators[name] = CalibratedClassifierCV(model, cv=3, method='isotonic')
                    self.calibrators[name].fit(X_s, y)
                except:
                    pass
            
            # ✅ Meta Model (Stacking)
            self.meta_model = LogisticRegression()
            self.meta_model.fit(base_preds_all, y)
            
            self.is_trained = True
            self.version = datetime.now().strftime('v%Y%m%d_%H%M%S')
            
            meta_pred = self.meta_model.predict(base_preds_all)
            acc = accuracy_score(y, meta_pred)
            
            self.logger.info(f"✅ {self.symbol}: دقة={acc:.1%}, ميزات={len(self.selected_features)}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.symbol}: {e}", exc_info=True)
            return False
    
    def predict(self, df: pd.DataFrame, dynamic_threshold: float = None) -> Tuple[str, float, Dict]:
        """Predict with meta-model and calibration."""
        if not self.is_trained:
            return "NEUTRAL", 0.0, {}
        
        threshold = dynamic_threshold or self.config.CONFIDENCE_THRESHOLD_INITIAL
        
        try:
            features = calculate_advanced_features(df).iloc[[-1]]
            available = [f for f in self.selected_features if f in features.columns]
            
            if len(available) < 15:
                return "NEUTRAL", 0.0, {}
            
            X = features[available].fillna(0)
            X_s = self.scaler.transform(X)
            
            # Get calibrated probabilities
            base_probas = []
            
            for name, calibrator in self.calibrators.items():
                try:
                    proba = float(calibrator.predict_proba(X_s)[0, 1])
                    base_probas.append(proba)
                except:
                    base_probas.append(0.5)
            
            # ✅ Meta model prediction (weighted, not voting!)
            meta_proba = float(self.meta_model.predict_proba(np.array([base_probas]))[0, 1])
            
            # Determine direction
            if meta_proba > threshold:
                direction = "BUY"
                confidence = meta_proba
            elif meta_proba < (1 - threshold):
                direction = "SELL"
                confidence = 1 - meta_proba
            else:
                direction = "NEUTRAL"
                confidence = max(meta_proba, 1 - meta_proba)
            
            return direction, confidence, {
                'meta_proba': meta_proba,
                'base_probas': dict(zip(self.calibrators.keys(), base_probas))
            }
            
        except Exception as e:
            return "NEUTRAL", 0.0, {}
    
    def save(self):
        path = os.path.join(self.config.MODELS_DIR, self.symbol)
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'calibrators': self.calibrators,
            'scaler': self.scaler,
            'features': self.selected_features,
            'version': self.version,
            'walk_forward_score': self.walk_forward_score
        }, os.path.join(path, 'pro_model.pkl'))
    
    def load(self) -> bool:
        path = os.path.join(self.config.MODELS_DIR, self.symbol, 'pro_model.pkl')
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.base_models = data['base_models']
        self.meta_model = data['meta_model']
        self.calibrators = data['calibrators']
        self.scaler = data['scaler']
        self.selected_features = data['features']
        self.version = data['version']
        self.walk_forward_score = data.get('walk_forward_score', 0)
        self.is_trained = True
        return True

# ============================================================================
# MAIN BOT
# ============================================================================

class FalconProBot:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.db = Database(config.DB_PATH, self.logger)
        self.models = {}
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
        self.tb = telebot.TeleBot(config.TELEGRAM_TOKEN)
        self._setup_commands()
        
        for symbol in config.SYMBOLS:
            model = ProEnsembleModel(symbol, config, self.logger)
            loaded = model.load()
            self.logger.info(f"{'📂' if loaded else '🆕'} {symbol}")
            self.models[symbol] = model
        
        self.running = False
        self.last_retrain = None
    
    def _setup_commands(self):
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            trained = sum(1 for m in self.models.values() if m.is_trained)
            stats = self.db.get_stats()
            text = f"🦅 **Falcon Pro v4**\n✅ نماذج: {trained}/{len(self.models)}\n📊 صفقات: {stats['total']}\n📈 نجاح: {stats['win_rate']:.1%}"
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['stats'])
        def stats_cmd(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            s = self.db.get_stats()
            self.tb.reply_to(msg, f"📊 {s['total']} | ✅ {s['wins']} | 📈 {s['win_rate']:.1%}")
        
        @self.tb.message_handler(commands=['regime'])
        def regime_cmd(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            text = "📊 **حالة الأسواق:**\n"
            for symbol in self.config.SYMBOLS[:4]:
                df = self.fetch_data(symbol, '15m', '5d')
                if df is not None:
                    regime = MarketRegime.detect(df)
                    text += f"\n• {symbol}: {regime['regime']} (ADX: {regime['adx']})"
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def fetch_data(self, symbol: str, interval: str = '5m', period: str = '5d') -> Optional[pd.DataFrame]:
        for attempt in range(self.config.MAX_RETRIES):
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                if not df.empty:
                    df.columns = [c.capitalize() for c in df.columns]
                    return df
            except Exception as e:
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY)
        return None
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        try:
            model = self.models.get(symbol)
            if not model or not model.is_trained:
                return None
            
            if self.db.check_active_signal(symbol):
                return None
            
            if self.db.check_recent_signal(symbol, self.config.SIGNAL_COOLDOWN_MINUTES):
                return None
            
            df_5m = self.fetch_data(symbol, '5m', '3d')
            df_15m = self.fetch_data(symbol, '15m', '5d')
            
            if df_5m is None or df_15m is None:
                return None
            
            # ✅ Dynamic threshold based on performance
            threshold = self.db.get_dynamic_threshold(symbol)
            
            dir_5m, conf_5m, info_5m = model.predict(df_5m, threshold)
            dir_15m, conf_15m, info_15m = model.predict(df_15m, threshold)
            
            if dir_5m != dir_15m or dir_5m == "NEUTRAL":
                return None
            
            # ✅ Market Regime
            regime = MarketRegime.detect(df_15m)
            
            # Don't trade ranges with low confidence
            if not regime['is_trend'] and conf_5m < 0.70:
                return None
            
            confidence = (conf_5m + conf_15m) / 2
            
            if confidence < threshold:
                return None
            
            self.logger.info(f"🎯 {symbol}: {dir_5m} | Meta={info_5m.get('meta_proba', 0):.2%} | {regime['regime']}")
            
            return {
                'symbol': symbol,
                'direction': dir_5m,
                'entry_price': float(df_5m['Close'].iloc[-1]),
                'confidence': confidence,
                'regime': regime['regime'],
                'meta_proba': info_5m.get('meta_proba', 0),
                'expiry_time': (datetime.now() + timedelta(minutes=self.config.TRADE_DURATION_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Analyze {symbol}: {e}")
            return None
    
    def send_signal(self, signal: Dict):
        try:
            emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
            direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
            
            msg = f"{emoji} **{signal['symbol']}** - {direction}\n\n💰 {signal['entry_price']:.5f}\n⏳ {self.config.TRADE_DURATION_MINUTES} د\n💪 {signal['confidence']:.1%}\n📊 {signal['regime']}\n\n🤖 Falcon Pro v4"
            
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
            self.logger.info(f"✅ إشارة: {signal['symbol']} {signal['direction']}")
        except:
            pass
    
    def check_trades(self):
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
                
                self.db.update_result(trade['id'], current, result, pnl, trade['symbol'])
            except:
                pass
    
    def scan_markets(self):
        futures = {self.executor.submit(self.analyze_symbol, s): s for s in self.config.SYMBOLS}
        signals = 0
        for future in as_completed(futures, timeout=60):
            try:
                signal = future.result(timeout=20)
                if signal and self.db.save_signal(signal):
                    self.send_signal(signal)
                    signals += 1
            except:
                pass
        return signals
    
    def train_all_models(self):
        self.logger.info("🎓 بدء التدريب المتقدم...")
        
        for symbol in self.config.SYMBOLS:
            try:
                # ✅ 12-24 month training data
                df = None
                for interval, period in [('1h', self.config.TRAINING_PERIOD_1H), 
                                          ('15m', self.config.TRAINING_PERIOD_15M)]:
                    df = self.fetch_data(symbol, interval, period)
                    if df is not None and len(df) >= self.config.MIN_TRAINING_SAMPLES:
                        self.logger.info(f"{symbol}: {len(df)} صف بفريم {interval} ({period})")
                        break
                    time.sleep(3)
                
                if df is not None:
                    model = ProEnsembleModel(symbol, self.config, self.logger)
                    if model.train(df):
                        model.save()
                        self.models[symbol] = model
                
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Train {symbol}: {e}")
        
        self.last_retrain = datetime.now()
        
        trained = sum(1 for m in self.models.values() if m.is_trained)
        try:
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID,
                f"🎓 **تدريب مكتمل**\n✅ {trained}/{len(self.config.SYMBOLS)}\n📊 Walk-Forward + Meta Model",
                parse_mode='Markdown')
        except:
            pass
    
    def start_telegram(self):
        def poll():
            while True:
                try:
                    self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except:
                    time.sleep(10)
        threading.Thread(target=poll, daemon=True).start()
    
    def run(self):
        self.running = True
        
        self.logger.info("=" * 50)
        self.logger.info("🦅 Falcon AI Pro v4.0")
        self.logger.info(f"📅 تدريب: {self.config.TRAINING_PERIOD_1H}")
        self.logger.info(f"🔄 Walk-Forward: {self.config.WALK_FORWARD_WINDOWS} windows")
        self.logger.info(f"🧠 Meta Model: Logistic Regression")
        self.logger.info(f"📊 Calibration: Isotonic")
        self.logger.info(f"🎯 Dynamic Threshold: {self.config.CONFIDENCE_THRESHOLD_MIN}-{self.config.CONFIDENCE_THRESHOLD_MAX}")
        self.logger.info("=" * 50)
        
        self.start_telegram()
        time.sleep(2)
        
        if not any(m.is_trained for m in self.models.values()):
            self.train_all_models()
        
        self.last_retrain = datetime.now()
        
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID,
                f"🦅 **Falcon Pro v4**\n✅ {trained}/{len(self.config.SYMBOLS)}\n🧠 Meta Model Active\n⚡️ Scanning...",
                parse_mode='Markdown')
        except:
            pass
        
        while self.running:
            try:
                self.check_trades()
                self.scan_markets()
                
                if (datetime.now() - self.last_retrain).total_seconds() > 86400:
                    self.train_all_models()
                
                time.sleep(self.config.SCAN_INTERVAL_MINUTES * 60)
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Loop: {e}")
                time.sleep(30)
        
        self.executor.shutdown(wait=True)

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    config = Config()
    bot = FalconProBot(config)
    bot.run()
