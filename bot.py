#!/usr/bin/env python3
"""
Falcon AI Ultimate v2.5 - Forex Only | 6-Month Training
========================================================
Optimized for Forex pairs with 6-month training data.
Better target definition for quality signals.
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['figure.dpi'] = 72

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score
import xgboost as xgb

import telebot
from telebot import types
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
    
    # ✅ فوركس فقط - 8 أزواج رئيسية
    SYMBOLS: List[str] = field(default_factory=lambda: [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
        'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X'
    ])
    
    # ✅ تدريب 6 شهور
    CONFIDENCE_THRESHOLD: float = 0.65  # 65% ثقة عشان جودة الإشارات
    RETRAINING_INTERVAL_HOURS: int = 24
    MIN_TRAINING_SAMPLES: int = 5000  # زيادة عشان 6 شهور
    TRAINING_PERIOD: str = '6mo'  # ✅ 6 أشهر تدريب
    
    # ✅ هدف أفضل: تحرك السعر بنسبة معينة
    TARGET_RETURN_THRESHOLD: float = 0.0008  # 0.08% حركة (8 نقاط لليورو دولار)
    FORECAST_PERIODS: int = 5  # نتوقع بعد 5 شمعات (بدل 3)
    
    MAX_FEATURES: int = 40
    
    DB_PATH: str = 'falcon_trading.db'
    MODELS_DIR: str = 'models'
    
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5
    MAX_WORKERS: int = 4
    SIGNAL_COOLDOWN_MINUTES: int = 15  # 15 دقيقة عشان مننساش
    
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
    return logging.getLogger('FalconAI')

# ============================================================================
# DATABASE
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
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    confidence REAL,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME,
                    exit_time DATETIME,
                    result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL,
                    pnl_pips REAL,
                    signal_hash TEXT UNIQUE
                );
                
                CREATE TABLE IF NOT EXISTS daily_performance (
                    date TEXT PRIMARY KEY,
                    total_signals INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    total_pips REAL DEFAULT 0,
                    best_symbol TEXT,
                    worst_symbol TEXT
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
                    (symbol, direction, entry_price, confidence, expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data['expiry_time'], signal_hash))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except:
            return None
    
    def check_active_signal(self, symbol: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute('''
                SELECT COUNT(*) FROM signals 
                WHERE symbol = ? AND result = 'PENDING' 
                AND expiry_time > datetime('now', 'localtime')
            ''', (symbol,)).fetchone()[0]
            return count > 0
    
    def check_recent_signal(self, symbol: str, minutes: int) -> bool:
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute('''
                SELECT COUNT(*) FROM signals 
                WHERE symbol = ? AND entry_time > ?
            ''', (symbol, cutoff)).fetchone()[0]
            return count > 0
    
    def update_result(self, signal_id: int, exit_price: float, result: str, pnl: float, pips: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE signals SET 
                exit_price = ?, result = ?, pnl_percent = ?, pnl_pips = ?,
                exit_time = datetime('now', 'localtime')
                WHERE id = ?
            ''', (exit_price, result, pnl, pips, signal_id))
            
            today = datetime.now().strftime('%Y-%m-%d')
            conn.execute('''
                INSERT INTO daily_performance (date, total_signals, wins, losses, total_pnl, total_pips)
                VALUES (?, 1, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                total_signals = total_signals + 1,
                wins = wins + ?,
                losses = losses + ?,
                total_pnl = total_pnl + ?,
                total_pips = total_pips + ?
            ''', (today,
                  1 if result == 'WIN' else 0,
                  1 if result == 'LOSS' else 0,
                  pnl, pips,
                  1 if result == 'WIN' else 0,
                  1 if result == 'LOSS' else 0,
                  pnl, pips))
            conn.commit()
    
    def get_pending_trades(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM signals 
                WHERE result = 'PENDING' AND expiry_time <= datetime('now', 'localtime')
            ''').fetchall()
            return [dict(r) for r in rows]
    
    def get_full_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM signals WHERE result != 'PENDING'").fetchone()[0]
            wins = conn.execute("SELECT COUNT(*) FROM signals WHERE result = 'WIN'").fetchone()[0]
            losses = conn.execute("SELECT COUNT(*) FROM signals WHERE result = 'LOSS'").fetchone()[0]
            
            win_rate = (wins / total * 100) if total > 0 else 0
            
            avg_pnl = conn.execute(
                "SELECT AVG(pnl_percent) FROM signals WHERE result != 'PENDING'"
            ).fetchone()[0] or 0
            
            total_pnl = conn.execute(
                "SELECT SUM(pnl_percent) FROM signals WHERE result != 'PENDING'"
            ).fetchone()[0] or 0
            
            total_pips = conn.execute(
                "SELECT SUM(pnl_pips) FROM signals WHERE result != 'PENDING'"
            ).fetchone()[0] or 0
            
            today = datetime.now().strftime('%Y-%m-%d')
            today_stats = conn.execute(
                "SELECT * FROM daily_performance WHERE date = ?", (today,)
            ).fetchone()
            
            today_signals = today_stats[1] if today_stats else 0
            today_wins = today_stats[2] if today_stats else 0
            today_pnl = today_stats[4] if today_stats else 0
            
            # Best/Worst symbols
            symbols = conn.execute('''
                SELECT symbol,
                       COUNT(*) as total,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins,
                       AVG(pnl_percent) as avg_pnl
                FROM signals WHERE result != 'PENDING'
                GROUP BY symbol HAVING total >= 3
                ORDER BY wins*1.0/total DESC
            ''').fetchall()
            
            best = symbols[0] if symbols else None
            worst = symbols[-1] if symbols else None
            
            return {
                'total': total,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': total_pnl,
                'total_pips': total_pips,
                'today_signals': today_signals,
                'today_wins': today_wins,
                'today_pnl': today_pnl,
                'best_symbol': f"{best[0]}" if best else 'N/A',
                'worst_symbol': f"{worst[0]}" if worst else 'N/A'
            }

# ============================================================================
# ADVANCED TECHNICAL INDICATORS
# ============================================================================

def calculate_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive features for Forex."""
    f = pd.DataFrame(index=df.index)
    c = df['Close']
    h = df['High']
    l = df['Low']
    o = df['Open']
    
    # ========== PRICE FEATURES ==========
    for p in [1, 3, 5, 10, 20, 50]:
        f[f'ret_{p}'] = c.pct_change(p)
    
    f['log_ret'] = np.log(c / c.shift(1))
    f['hl_ratio'] = (h - l) / (c + 1e-8)
    f['close_pos'] = (c - l) / (h - l + 1e-8)
    f['gap'] = (o - c.shift(1)) / c.shift(1)
    
    # ========== MOVING AVERAGES ==========
    for p in [5, 10, 20, 50, 100, 200]:
        if len(df) >= p:
            f[f'sma_{p}'] = c.rolling(p).mean()
            f[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
            f[f'dist_sma_{p}'] = (c - f[f'sma_{p}']) / f[f'sma_{p}']
    
    # ========== RSI ==========
    for p in [7, 14, 21]:
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).rolling(p).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(p).mean()
        f[f'rsi_{p}'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    # ========== MACD ==========
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    f['macd'] = ema12 - ema26
    f['macd_signal'] = f['macd'].ewm(span=9).mean()
    f['macd_hist'] = f['macd'] - f['macd_signal']
    f['macd_cross'] = ((f['macd'] > f['macd_signal']) & (f['macd'].shift(1) <= f['macd_signal'].shift(1))).astype(int)
    
    # ========== BOLLINGER BANDS ==========
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f['bb_upper'] = sma20 + 2 * std20
    f['bb_lower'] = sma20 - 2 * std20
    f['bb_pos'] = (c - f['bb_lower']) / (f['bb_upper'] - f['bb_lower'] + 1e-8)
    f['bb_width'] = (f['bb_upper'] - f['bb_lower']) / (sma20 + 1e-8)
    f['bb_squeeze'] = f['bb_width'] / f['bb_width'].rolling(20).mean()
    
    # ========== ATR ==========
    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    f['atr_14'] = tr.ewm(span=14).mean()
    f['atr_pct'] = f['atr_14'] / (c + 1e-8)
    
    # ========== STOCHASTIC ==========
    for p in [14, 21]:
        low_p = l.rolling(p).min()
        high_p = h.rolling(p).max()
        f[f'stoch_k_{p}'] = 100 * (c - low_p) / (high_p - low_p + 1e-8)
        f[f'stoch_d_{p}'] = f[f'stoch_k_{p}'].rolling(3).mean()
    
    # ========== CCI ==========
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    f['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-8)
    
    # ========== WILLIAMS %R ==========
    hh14 = h.rolling(14).max()
    ll14 = l.rolling(14).min()
    f['williams_r'] = -100 * (hh14 - c) / (hh14 - ll14 + 1e-8)
    
    # ========== ADX (Trend Strength) ==========
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    atr14 = tr.ewm(span=14).mean()
    plus_di = 100 * (plus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    minus_di = 100 * (minus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    f['adx'] = dx.ewm(span=14).mean()
    f['di_plus'] = plus_di
    f['di_minus'] = minus_di
    
    # ========== ROC & MOMENTUM ==========
    for p in [5, 10, 20, 50]:
        f[f'roc_{p}'] = (c - c.shift(p)) / (c.shift(p) + 1e-8) * 100
        f[f'mom_{p}'] = c - c.shift(p)
    
    # ========== VOLATILITY ==========
    for p in [5, 10, 20, 50]:
        f[f'vol_{p}'] = c.pct_change().rolling(p).std()
    
    # ========== VOLUME ==========
    if 'Volume' in df.columns:
        v = df['Volume']
        f['vol_ratio'] = v / (v.rolling(20).mean() + 1e-8)
        f['vol_trend'] = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-8)
    
    # ========== TREND STRENGTH ==========
    f['trend_str'] = c.rolling(20).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
    )
    
    # ========== SUPPORT/RESISTANCE PROXIMITY ==========
    for p in [20, 50]:
        f[f'high_{p}d'] = c / (h.rolling(p).max() + 1e-8)
        f[f'low_{p}d'] = c / (l.rolling(p).min() + 1e-8)
    
    # ========== PRICE PATTERNS ==========
    # Higher highs / Lower lows
    f['hh_20'] = (h == h.rolling(20).max()).astype(int)
    f['ll_20'] = (l == l.rolling(20).min()).astype(int)
    
    return f.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# ============================================================================
# BETTER TARGET DEFINITION
# ============================================================================

def create_target(df: pd.DataFrame, threshold: float, periods: int) -> Tuple[pd.Series, pd.Series]:
    """
    ✅ هدف محسن:
    - BUY: السعر بيزيد بنسبة threshold خلال periods شمعات
    - SELL: السعر بينقص بنسبة threshold خلال periods شمعات
    - NEUTRAL: غير كده (ما بيتدربش عليه)
    """
    future_price = df['Close'].shift(-periods)
    future_return = (future_price - df['Close']) / df['Close']
    
    # BUY = 1, SELL = 0, NEUTRAL = نستبعده من التدريب
    buy_signal = (future_return > threshold).astype(int)
    sell_signal = (future_return < -threshold).astype(int)
    
    # دمج الإشارات
    target = buy_signal.copy()
    target[sell_signal == 1] = 0
    
    # تحديد الصفوف المحايدة (ما بينشملش في التدريب)
    neutral = (~buy_signal.astype(bool)) & (~sell_signal.astype(bool))
    
    return target, neutral

# ============================================================================
# IMPROVED MODEL
# ============================================================================

class ForexModel:
    """Optimized model for Forex with 6-month training."""
    
    def __init__(self, symbol: str, config: Config, logger: logging.Logger):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        self.model = None
        self.scaler = RobustScaler()
        self.selected_features = []
        self.is_trained = False
        self.version = None
        self.train_accuracy = 0
        self.feature_count = 0
    
    def train(self, df: pd.DataFrame) -> bool:
        try:
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                self.logger.warning(f"{self.symbol}: بيانات غير كافية ({len(df)} صف)")
                return False
            
            self.logger.info(f"🎓 تدريب {self.symbol} - {len(df)} صف بيانات...")
            
            # Calculate features
            features = calculate_advanced_features(df)
            
            # Create better target
            target, neutral = create_target(
                df,
                self.config.TARGET_RETURN_THRESHOLD,
                self.config.FORECAST_PERIODS
            )
            
            # Remove NaN and neutral periods
            valid = ~(features.isna().any(axis=1) | target.isna() | neutral)
            X = features[valid]
            y = target[valid]
            
            self.logger.info(f"{self.symbol}: {len(X)} عينة تدريب صالحة "
                           f"(تم استبعاد {neutral.sum()} فترة محايدة)")
            
            if len(X) < 1000:
                return False
            
            # Feature selection
            mi = mutual_info_classif(X, y, random_state=42)
            scores = sorted(zip(X.columns, mi), key=lambda x: x[1], reverse=True)
            self.selected_features = [s[0] for s in scores[:self.config.MAX_FEATURES]]
            self.feature_count = len(self.selected_features)
            
            X = X[self.selected_features]
            
            # Time-series split (80/20)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale
            X_train_s = self.scaler.fit_transform(X_train)
            X_val_s = self.scaler.transform(X_val)
            
            # Train XGBoost
            self.model = xgb.XGBClassifier(
                n_estimators=300,  # زيادة عدد الأشجار
                learning_rate=0.02,  # تعلم أبطأ عشان تعميم أفضل
                max_depth=5,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=2,
                verbosity=0,
                tree_method='hist',
                early_stopping_rounds=20  # إيقاف مبكر
            )
            
            self.model.fit(
                X_train_s, y_train,
                eval_set=[(X_val_s, y_val)],
                verbose=False
            )
            
            # Validate
            val_pred = self.model.predict(X_val_s)
            self.train_accuracy = accuracy_score(y_val, val_pred)
            train_f1 = f1_score(y_val, val_pred, zero_division=0)
            
            self.is_trained = True
            self.version = datetime.now().strftime('v%Y%m%d_%H%M%S')
            
            self.logger.info(f"✅ {self.symbol}: دقة={self.train_accuracy:.2%}, "
                           f"F1={train_f1:.3f}, ميزات={self.feature_count}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ تدريب {self.symbol}: {e}")
            return False
    
    def predict(self, df: pd.DataFrame) -> Tuple[str, float, float]:
        """
        Returns: (direction, confidence, expected_return)
        """
        if not self.is_trained:
            return "NEUTRAL", 0.0, 0.0
        
        try:
            features = calculate_advanced_features(df).iloc[[-1]]
            available = [f for f in self.selected_features if f in features.columns]
            
            if len(available) < 15:
                return "NEUTRAL", 0.0, 0.0
            
            X = features[available].fillna(0)
            X_s = self.scaler.transform(X)
            
            proba = float(self.model.predict_proba(X_s)[0, 1])
            
            # Calculate expected return
            expected_return = abs(proba - 0.5) * 2 * self.config.TARGET_RETURN_THRESHOLD * 100
            
            if proba > self.config.CONFIDENCE_THRESHOLD:
                return "BUY", proba, expected_return
            elif proba < (1 - self.config.CONFIDENCE_THRESHOLD):
                return "SELL", 1 - proba, -expected_return
            return "NEUTRAL", max(proba, 1 - proba), 0.0
            
        except Exception as e:
            return "NEUTRAL", 0.0, 0.0
    
    def save(self):
        path = os.path.join(self.config.MODELS_DIR, self.symbol)
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.selected_features,
            'version': self.version,
            'accuracy': self.train_accuracy,
            'feature_count': self.feature_count
        }, os.path.join(path, 'model.pkl'))
    
    def load(self) -> bool:
        path = os.path.join(self.config.MODELS_DIR, self.symbol, 'model.pkl')
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.selected_features = data['features']
        self.version = data['version']
        self.train_accuracy = data.get('accuracy', 0)
        self.feature_count = data.get('feature_count', 0)
        self.is_trained = True
        return True

# ============================================================================
# TREND FILTER FOR FOREX
# ============================================================================

def forex_trend_filter(df: pd.DataFrame) -> Dict:
    """
    ✅ فلتر اتجاه محسن للفوركس
    """
    if len(df) < 50:
        return {'trend': 'NEUTRAL', 'strength': 0, 'valid': True}
    
    c = df['Close']
    h = df['High']
    l = df['Low']
    
    # EMAs
    ema20 = c.ewm(span=20).mean().iloc[-1]
    ema50 = c.ewm(span=50).mean().iloc[-1]
    ema200 = c.ewm(span=200).mean().iloc[-1] if len(df) >= 200 else ema50
    current = c.iloc[-1]
    
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
    adx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    adx_val = float(adx.ewm(span=14).mean().iloc[-1])
    
    # Trend score
    score = 0
    
    if current > ema20: score += 1
    else: score -= 1
    
    if current > ema50: score += 1
    else: score -= 1
    
    if current > ema200: score += 1
    else: score -= 1
    
    if ema20 > ema50: score += 1
    else: score -= 1
    
    if adx_val > 25:
        if score > 0: score += 1
        elif score < 0: score -= 1
    
    if score >= 3:
        trend = 'STRONG_UP'
    elif score >= 1:
        trend = 'UP'
    elif score <= -3:
        trend = 'STRONG_DOWN'
    elif score <= -1:
        trend = 'DOWN'
    else:
        trend = 'SIDEWAYS'
    
    strength = min(abs(score) / 5, 1.0)
    
    # ✅ تحسين: لو السوق sideways، مندخلش
    valid = trend != 'SIDEWAYS' or adx_val > 20
    
    return {
        'trend': trend,
        'strength': strength,
        'adx': round(adx_val, 1),
        'valid': valid
    }

# ============================================================================
# PIP CALCULATOR
# ============================================================================

def calculate_pips(symbol: str, entry: float, exit: float, direction: str) -> float:
    """Calculate profit/loss in pips."""
    pip_value = 0.0001  # Default for most pairs
    
    if 'JPY' in symbol:
        pip_value = 0.01  # JPY pairs
    
    if direction == 'BUY':
        pips = (exit - entry) / pip_value
    else:
        pips = (entry - exit) / pip_value
    
    return round(pips, 1)

# ============================================================================
# MAIN BOT
# ============================================================================

class FalconForexBot:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.db = Database(config.DB_PATH, self.logger)
        self.models: Dict[str, ForexModel] = {}
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
        self.tb = telebot.TeleBot(config.TELEGRAM_TOKEN)
        self._setup_commands()
        
        # Load models
        for symbol in config.SYMBOLS:
            model = ForexModel(symbol, config, self.logger)
            if model.load():
                self.logger.info(f"📂 {symbol}: دقة={model.train_accuracy:.1%}, "
                               f"ميزات={model.feature_count}")
            else:
                self.logger.info(f"🆕 {symbol}: يحتاج تدريب")
            self.models[symbol] = model
        
        self.running = False
        self.last_retrain = None
        self.last_report = None
    
    def _setup_commands(self):
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            trained = sum(1 for m in self.models.values() if m.is_trained)
            stats = self.db.get_full_stats()
            
            text = f"""
🦅 **Falcon AI - فوركس فقط**

✅ الحالة: يعمل
🤖 النماذج: {trained}/{len(self.models)}
⚙️ عتبة الثقة: {self.config.CONFIDENCE_THRESHOLD:.0%}
📅 تدريب: {self.config.TRAINING_PERIOD}

📊 **الأداء:**
• إجمالي الصفقات: {stats['total']}
• رابحة: {stats['wins']} | خاسرة: {stats['losses']}
• 📈 نسبة النجاح: {stats['win_rate']:.1f}%
• 💰 الربح: {stats['total_pnl']:.2f}%
• 📊 النقاط: {stats['total_pips']:.1f}

📅 **اليوم:**
• صفقات: {stats['today_signals']}
• رابحة: {stats['today_wins']}
• ربح: {stats['today_pnl']:.2f}%

⭐ الأفضل: {stats['best_symbol']}
👎 الأسوأ: {stats['worst_symbol']}

⚡️ جاري التحليل...
"""
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['stats'])
        def stats_cmd(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            self.send_performance_report()
        
        @self.tb.message_handler(commands=['models'])
        def models_cmd(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            text = "🤖 **حالة النماذج:**\n\n"
            for symbol, model in self.models.items():
                if model.is_trained:
                    text += f"✅ {symbol}: دقة={model.train_accuracy:.1%}, ميزات={model.feature_count}\n"
                else:
                    text += f"❌ {symbol}: غير مدرب\n"
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def send_performance_report(self):
        stats = self.db.get_full_stats()
        
        if stats['total'] == 0:
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, "📊 لا توجد صفقات مكتملة")
            return
        
        emoji = "🟢" if stats['win_rate'] >= 60 else ("🟡" if stats['win_rate'] >= 45 else "🔴")
        
        text = f"""
{emoji} **تقرير أداء الفوركس**

📊 صفقات: **{stats['total']}**
✅ ربح: **{stats['wins']}** | ❌ خسارة: **{stats['losses']}**

📈 **نسبة النجاح: {stats['win_rate']:.1f}%**
💰 إجمالي الربح: **{stats['total_pnl']:.2f}%**
📊 إجمالي النقاط: **{stats['total_pips']:.1f}**

📅 اليوم: {stats['today_signals']} صفقة | ربح: {stats['today_pnl']:.2f}%

⭐ الأفضل: {stats['best_symbol']}
👎 الأسوأ: {stats['worst_symbol']}

🦅 Falcon AI Forex
"""
        self.tb.send_message(self.config.TELEGRAM_CHAT_ID, text, parse_mode='Markdown')
    
    def fetch_data(self, symbol: str, interval: str = '5m', period: str = '5d') -> Optional[pd.DataFrame]:
        for attempt in range(self.config.MAX_RETRIES):
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                if not df.empty:
                    df.columns = [c.capitalize() for c in df.columns]
                    return df
            except Exception as e:
                self.logger.warning(f"Fetch {symbol} attempt {attempt+1}: {e}")
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
            
            dir_5m, conf_5m, exp_ret_5m = model.predict(df_5m)
            dir_15m, conf_15m, exp_ret_15m = model.predict(df_15m)
            
            if dir_5m != dir_15m or dir_5m == "NEUTRAL":
                return None
            
            # ✅ فلتر اتجاه محسن
            trend_info = forex_trend_filter(df_15m)
            
            if not trend_info['valid']:
                return None
            
            if dir_5m == "BUY" and trend_info['trend'] in ['STRONG_DOWN', 'DOWN']:
                return None
            if dir_5m == "SELL" and trend_info['trend'] in ['STRONG_UP', 'UP']:
                return None
            
            confidence = (conf_5m * 0.6 + conf_15m * 0.4)
            
            if trend_info['adx'] > 25:
                confidence = min(confidence * 1.1, 0.95)
            
            if confidence < self.config.CONFIDENCE_THRESHOLD:
                return None
            
            expected_return = (exp_ret_5m + exp_ret_15m) / 2
            
            self.logger.info(f"🎯 {symbol}: {dir_5m} | ثقة={confidence:.1%} | "
                           f"العائد المتوقع={expected_return:+.3f}% | ADX={trend_info['adx']}")
            
            return {
                'symbol': symbol,
                'direction': dir_5m,
                'entry_price': float(df_5m['Close'].iloc[-1]),
                'confidence': confidence,
                'expected_return': expected_return,
                'trend': trend_info['trend'],
                'adx': trend_info['adx'],
                'expiry_time': (datetime.now() + timedelta(minutes=self.config.TRADE_DURATION_MINUTES)).strftime('%Y-%m-%d %H:%M:%S'),
                'model_version': model.version
            }
            
        except Exception as e:
            self.logger.error(f"Analyze {symbol}: {e}")
            return None
    
    def send_signal(self, signal: Dict):
        try:
            emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
            direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
            
            msg = f"""
{emoji} **{signal['symbol']}** - {direction}

💰 الدخول: {signal['entry_price']:.5f}
⏳ المدة: {self.config.TRADE_DURATION_MINUTES} دقائق
💪 الثقة: {signal['confidence']:.1%}
📊 العائد المتوقع: {signal['expected_return']:+.3f}%

📈 الاتجاه: {signal['trend']} (ADX: {signal['adx']})

🦅 Falcon AI Forex
"""
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
            self.logger.info(f"✅ إشارة: {signal['symbol']} {signal['direction']}")
            
        except Exception as e:
            self.logger.error(f"Send error: {e}")
    
    def check_trades(self):
        for trade in self.db.get_pending_trades():
            try:
                df = self.fetch_data(trade['symbol'], '5m', '1d')
                if df is None:
                    continue
                
                current = float(df['Close'].iloc[-1])
                entry = trade['entry_price']
                direction = trade['direction']
                
                if direction == 'BUY':
                    pnl = (current - entry) / entry * 100
                    result = 'WIN' if current > entry else 'LOSS'
                else:
                    pnl = (entry - current) / entry * 100
                    result = 'WIN' if current < entry else 'LOSS'
                
                pips = calculate_pips(trade['symbol'], entry, current, direction)
                
                self.db.update_result(trade['id'], current, result, pnl, pips)
                
                emoji = "✅" if result == 'WIN' else "❌"
                self.logger.info(f"{emoji} {trade['symbol']}: {result} | "
                               f"{pnl:+.2f}% | {pips:+} نقاط")
                
                try:
                    self.tb.send_message(
                        self.config.TELEGRAM_CHAT_ID,
                        f"{emoji} **{trade['symbol']}**\n"
                        f"النتيجة: {result}\n"
                        f"الربح: {pnl:+.2f}%\n"
                        f"النقاط: {pips:+}\n"
                        f"{entry:.5f} → {current:.5f}",
                        parse_mode='Markdown'
                    )
                except:
                    pass
                
            except Exception as e:
                self.logger.error(f"Check trade error: {e}")
    
    def scan_markets(self):
        futures = {}
        for s in self.config.SYMBOLS:
            futures[self.executor.submit(self.analyze_symbol, s)] = s
        
        signals_found = 0
        for future in as_completed(futures, timeout=60):
            symbol = futures[future]
            try:
                signal = future.result(timeout=20)
                if signal:
                    sig_id = self.db.save_signal(signal)
                    if sig_id:
                        self.send_signal(signal)
                        signals_found += 1
            except TimeoutError:
                pass
            except Exception as e:
                self.logger.error(f"Error: {symbol}: {e}")
        
        return signals_found
    
    def train_all_models(self):
        self.logger.info("🎓 بدء تدريب النماذج (6 أشهر بيانات)...")
        
        for symbol in self.config.SYMBOLS:
            try:
                df = self.fetch_data(symbol, '1h', self.config.TRAINING_PERIOD)
                if df is not None and len(df) >= self.config.MIN_TRAINING_SAMPLES:
                    model = ForexModel(symbol, self.config, self.logger)
                    if model.train(df):
                        model.save()
                        self.models[symbol] = model
                else:
                    self.logger.warning(f"{symbol}: بيانات غير كافية للتدريب")
            except Exception as e:
                self.logger.error(f"Train {symbol}: {e}")
        
        self.last_retrain = datetime.now()
        
        try:
            report = "🎓 **تقرير التدريب**\n\n"
            for symbol, model in self.models.items():
                if model.is_trained:
                    report += f"✅ {symbol}: دقة={model.train_accuracy:.1%}\n"
                else:
                    report += f"❌ {symbol}: فشل\n"
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, report, parse_mode='Markdown')
        except:
            pass
    
    def start_telegram(self):
        def poll():
            while True:
                try:
                    self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except Exception as e:
                    self.logger.error(f"Polling: {e}")
                    time.sleep(10)
        
        threading.Thread(target=poll, daemon=True).start()
    
    def run(self):
        self.running = True
        
        self.logger.info("=" * 50)
        self.logger.info("🦅 Falcon AI Forex - بدء التشغيل")
        self.logger.info(f"📊 الأزواج: {len(self.config.SYMBOLS)} أزواج فوركس")
        self.logger.info(f"📅 تدريب: {self.config.TRAINING_PERIOD}")
        self.logger.info(f"⚙️ ثقة: {self.config.CONFIDENCE_THRESHOLD:.0%}")
        self.logger.info(f"🎯 هدف: {self.config.TARGET_RETURN_THRESHOLD:.2%}")
        self.logger.info("=" * 50)
        
        self.start_telegram()
        time.sleep(2)
        
        if not any(m.is_trained for m in self.models.values()):
            self.train_all_models()
        
        self.last_retrain = datetime.now()
        self.last_report = datetime.now()
        
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(
                self.config.TELEGRAM_CHAT_ID,
                f"🦅 **Falcon AI Forex**\n\n"
                f"✅ جاهز | نماذج: {trained}/{len(self.models)}\n"
                f"📅 تدريب: {self.config.TRAINING_PERIOD}\n"
                f"⚙️ ثقة: {self.config.CONFIDENCE_THRESHOLD:.0%}\n\n"
                f"⚡️ جاري تحليل الأسواق...",
                parse_mode='Markdown'
            )
        except:
            pass
        
        while self.running:
            try:
                self.check_trades()
                signals = self.scan_markets()
                
                if (datetime.now() - self.last_retrain).total_seconds() > 86400:
                    self.train_all_models()
                
                if (datetime.now() - self.last_report).total_seconds() > 14400:
                    self.send_performance_report()
                    self.last_report = datetime.now()
                
                self.logger.info(f"😴 انتظار {self.config.SCAN_INTERVAL_MINUTES} دقائق...")
                time.sleep(self.config.SCAN_INTERVAL_MINUTES * 60)
                
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"خطأ: {e}")
                time.sleep(30)
        
        self.executor.shutdown(wait=True)
        self.logger.info("🛑 تم الإيقاف")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    config = Config()
    bot = FalconForexBot(config)
    bot.run()
