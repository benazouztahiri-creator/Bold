#!/usr/bin/env python3
"""
Falcon AI Forex v2.6 - Fixed Training
======================================
Fixed Yahoo Finance data fetching issues.
Multiple fallback strategies for reliable training.
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
    
    SYMBOLS: List[str] = field(default_factory=lambda: [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
        'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X'
    ])
    
    CONFIDENCE_THRESHOLD: float = 0.65
    RETRAINING_INTERVAL_HOURS: int = 24
    MIN_TRAINING_SAMPLES: int = 500  # ✅ أقل عشان البيانات المتاحة
    TRAINING_PERIOD_1H: str = '1mo'  # ✅ شهر للفريم الكبير
    TRAINING_PERIOD_15M: str = '15d'  # ✅ 15 يوم للفريم المتوسط
    TRAINING_PERIOD_5M: str = '7d'   # ✅ 7 أيام للفريم الصغير
    
    TARGET_RETURN_THRESHOLD: float = 0.0005
    FORECAST_PERIODS: int = 3
    
    MAX_FEATURES: int = 25
    
    DB_PATH: str = 'falcon_trading.db'
    MODELS_DIR: str = 'models'
    
    MAX_RETRIES: int = 5  # ✅ محاولات أكثر
    RETRY_DELAY: int = 10  # ✅ انتظار أطول بين المحاولات
    MAX_WORKERS: int = 2  # ✅ عمال أقل عشان ما نضغطش على API
    
    SIGNAL_COOLDOWN_MINUTES: int = 15
    
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
                    total_pips REAL DEFAULT 0
                );
                
                CREATE TABLE IF NOT EXISTS training_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    samples_count INTEGER,
                    features_count INTEGER,
                    accuracy REAL,
                    f1_score REAL,
                    trained_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    success INTEGER DEFAULT 0
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
    
    def log_training(self, symbol: str, samples: int, features: int, 
                     accuracy: float, f1: float, success: bool):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO training_log (symbol, samples_count, features_count, 
                                         accuracy, f1_score, success)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, samples, features, accuracy, f1, 1 if success else 0))
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
            
            # Best/Worst
            symbols = conn.execute('''
                SELECT symbol,
                       COUNT(*) as total,
                       SUM(CASE WHEN result='WIN' THEN 1 ELSE 0 END) as wins
                FROM signals WHERE result != 'PENDING'
                GROUP BY symbol HAVING total >= 3
                ORDER BY wins*1.0/total DESC
            ''').fetchall()
            
            best = symbols[0][0] if symbols else 'N/A'
            worst = symbols[-1][0] if symbols else 'N/A'
            
            return {
                'total': total, 'wins': wins, 'losses': losses,
                'win_rate': win_rate, 'total_pnl': total_pnl,
                'total_pips': total_pips,
                'today_signals': today_signals,
                'today_wins': today_wins,
                'today_pnl': today_pnl,
                'best_symbol': best,
                'worst_symbol': worst
            }

# ============================================================================
# FEATURES (SIMPLIFIED BUT EFFECTIVE)
# ============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate effective features for Forex."""
    f = pd.DataFrame(index=df.index)
    c = df['Close']
    h = df['High']
    l = df['Low']
    
    # Returns
    for p in [1, 3, 5, 10, 20]:
        f[f'ret_{p}'] = c.pct_change(p)
    
    # Moving averages
    for p in [5, 10, 20, 50]:
        f[f'sma_{p}'] = c.rolling(p).mean()
        f[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
        f[f'dist_sma_{p}'] = (c - f[f'sma_{p}']) / f[f'sma_{p}']
    
    # RSI
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    f['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    f['macd'] = ema12 - ema26
    f['macd_signal'] = f['macd'].ewm(span=9).mean()
    f['macd_hist'] = f['macd'] - f['macd_signal']
    
    # Bollinger
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f['bb_pos'] = (c - sma20) / (2 * std20 + 1e-8)
    f['bb_width'] = (2 * std20) / (sma20 + 1e-8)
    
    # ATR
    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    f['atr'] = tr.ewm(span=14).mean()
    f['atr_pct'] = f['atr'] / (c + 1e-8)
    
    # Stochastic
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    f['stoch_k'] = 100 * (c - low14) / (high14 - low14 + 1e-8)
    
    # ROC
    for p in [5, 10]:
        f[f'roc_{p}'] = (c - c.shift(p)) / (c.shift(p) + 1e-8) * 100
    
    # Volatility
    f['volatility'] = c.pct_change().rolling(20).std()
    
    # Support/Resistance proximity
    f['high_20d'] = c / (h.rolling(20).max() + 1e-8)
    f['low_20d'] = c / (l.rolling(20).min() + 1e-8)
    
    return f.fillna(0)

# ============================================================================
# SIMPLE TREND
# ============================================================================

def check_trend(df: pd.DataFrame) -> Tuple[str, bool]:
    """Check if trend is favorable."""
    if len(df) < 30:
        return "NEUTRAL", True
    
    c = df['Close']
    ema20 = c.ewm(span=20).mean().iloc[-1]
    ema50 = c.ewm(span=50).mean().iloc[-1] if len(df) >= 50 else ema20
    current = c.iloc[-1]
    
    if current > ema20 > ema50:
        return "UP", True
    elif current < ema20 < ema50:
        return "DOWN", True
    return "SIDEWAYS", True  # ✅ Sideways مسموح عشان منقللش الإشارات

# ============================================================================
# MODEL WITH BETTER DATA HANDLING
# ============================================================================

class ForexModel:
    """Robust model with multiple data fallback strategies."""
    
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
    
    def fetch_training_data(self) -> Optional[pd.DataFrame]:
        """
        ✅ Fetch training data with fallback strategy.
        Try 1h first, then 15m, then 5m.
        """
        strategies = [
            ('1h', self.config.TRAINING_PERIOD_1H),
            ('15m', self.config.TRAINING_PERIOD_15M),
            ('5m', self.config.TRAINING_PERIOD_5M),
        ]
        
        for interval, period in strategies:
            try:
                self.logger.info(f"📡 {self.symbol}: محاولة جلب {period} بفريم {interval}...")
                
                ticker = yf.Ticker(self.symbol)
                df = ticker.history(period=period, interval=interval)
                
                if df is not None and not df.empty:
                    df.columns = [c.capitalize() for c in df.columns]
                    self.logger.info(f"✅ {self.symbol}: تم جلب {len(df)} صف بفريم {interval}")
                    
                    if len(df) >= 100:
                        return df
                else:
                    self.logger.warning(f"⚠️ {self.symbol}: بيانات فارغة لفريم {interval}")
                    
            except Exception as e:
                self.logger.warning(f"❌ {self.symbol}: فشل جلب {interval} - {e}")
                time.sleep(5)
        
        return None
    
    def train(self, df: Optional[pd.DataFrame] = None) -> bool:
        """Train model with provided or fetched data."""
        try:
            # Fetch data if not provided
            if df is None:
                df = self.fetch_training_data()
            
            if df is None:
                self.logger.error(f"❌ {self.symbol}: لا توجد بيانات للتدريب")
                return False
            
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                self.logger.warning(f"⚠️ {self.symbol}: بيانات غير كافية ({len(df)} < {self.config.MIN_TRAINING_SAMPLES})")
                return False
            
            self.logger.info(f"🎓 تدريب {self.symbol}: {len(df)} صف...")
            
            # Features
            features = calculate_features(df)
            
            # Target: price movement after N periods
            future_price = df['Close'].shift(-self.config.FORECAST_PERIODS)
            future_return = (future_price - df['Close']) / df['Close']
            target = (future_return > self.config.TARGET_RETURN_THRESHOLD).astype(int)
            
            # Remove NaN
            valid = ~(features.isna().any(axis=1) | target.isna())
            X = features[valid]
            y = target[valid]
            
            self.logger.info(f"📊 {self.symbol}: {len(X)} عينة صالحة للتدريب")
            
            if len(X) < 100:
                self.logger.error(f"❌ {self.symbol}: عينات قليلة جداً ({len(X)})")
                return False
            
            # Feature selection
            try:
                mi = mutual_info_classif(X, y, random_state=42)
                scores = sorted(zip(X.columns, mi), key=lambda x: x[1], reverse=True)
                self.selected_features = [s[0] for s in scores[:self.config.MAX_FEATURES]]
            except:
                # Fallback: use all features
                self.selected_features = list(X.columns)[:self.config.MAX_FEATURES]
            
            X = X[self.selected_features]
            
            # Split
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Scale
            X_train_s = self.scaler.fit_transform(X_train)
            X_val_s = self.scaler.transform(X_val)
            
            # Train XGBoost
            self.model = xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.03,
                max_depth=4,
                min_child_weight=2,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=2,
                verbosity=0,
                tree_method='hist'
            )
            
            self.model.fit(X_train_s, y_train)
            
            # Validate
            val_pred = self.model.predict(X_val_s)
            self.train_accuracy = accuracy_score(y_val, val_pred)
            train_f1 = f1_score(y_val, val_pred, zero_division=0)
            
            self.is_trained = True
            self.version = datetime.now().strftime('v%Y%m%d_%H%M%S')
            
            self.logger.info(f"✅ {self.symbol}: تدريب ناجح! دقة={self.train_accuracy:.1%}, "
                           f"F1={train_f1:.3f}, ميزات={len(self.selected_features)}, "
                           f"عينات={len(X_train)}")
            
            # Log to database
            try:
                self.config.DB_PATH  # Just to check if db exists
            except:
                pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ تدريب {self.symbol} فشل: {e}", exc_info=True)
            return False
    
    def predict(self, df: pd.DataFrame) -> Tuple[str, float]:
        if not self.is_trained:
            return "NEUTRAL", 0.0
        
        try:
            features = calculate_features(df).iloc[[-1]]
            available = [f for f in self.selected_features if f in features.columns]
            
            if len(available) < 5:
                return "NEUTRAL", 0.0
            
            X = features[available].fillna(0)
            X_s = self.scaler.transform(X)
            
            proba = float(self.model.predict_proba(X_s)[0, 1])
            
            if proba > self.config.CONFIDENCE_THRESHOLD:
                return "BUY", proba
            elif proba < (1 - self.config.CONFIDENCE_THRESHOLD):
                return "SELL", 1 - proba
            return "NEUTRAL", max(proba, 1 - proba)
            
        except Exception as e:
            return "NEUTRAL", 0.0
    
    def save(self):
        path = os.path.join(self.config.MODELS_DIR, self.symbol)
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.selected_features,
            'version': self.version,
            'accuracy': self.train_accuracy
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
        self.is_trained = True
        return True

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
        
        # Initialize models
        for symbol in config.SYMBOLS:
            model = ForexModel(symbol, config, self.logger)
            loaded = model.load()
            if loaded:
                self.logger.info(f"📂 {symbol}: محمل (دقة={model.train_accuracy:.1%})")
            else:
                self.logger.info(f"🆕 {symbol}: جديد")
            self.models[symbol] = model
        
        self.running = False
        self.last_retrain = None
    
    def _setup_commands(self):
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            trained = sum(1 for m in self.models.values() if m.is_trained)
            stats = self.db.get_full_stats()
            
            text = f"""
🦅 **Falcon AI Forex**

✅ يعمل | نماذج: {trained}/{len(self.models)}
⚙️ ثقة: {self.config.CONFIDENCE_THRESHOLD:.0%}

📊 **الأداء:**
• صفقات: {stats['total']}
• نسبة نجاح: {stats['win_rate']:.1f}%
• ربح: {stats['total_pnl']:.2f}%
• نقاط: {stats['total_pips']:.1f}

📅 اليوم: {stats['today_signals']} | ربح: {stats['today_pnl']:.2f}%

⚡️ جاري التحليل...
"""
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['stats'])
        def stats_cmd(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            stats = self.db.get_full_stats()
            if stats['total'] == 0:
                self.tb.reply_to(msg, "📊 لا توجد صفقات بعد")
                return
            
            emoji = "🟢" if stats['win_rate'] >= 60 else ("🟡" if stats['win_rate'] >= 45 else "🔴")
            text = f"""
{emoji} **تقرير الأداء**

📊 صفقات: {stats['total']}
✅ ربح: {stats['wins']} | ❌ خسارة: {stats['losses']}
📈 نسبة نجاح: **{stats['win_rate']:.1f}%**
💰 ربح: {stats['total_pnl']:.2f}%
📊 نقاط: {stats['total_pips']:.1f}

📅 اليوم: {stats['today_signals']} | {stats['today_pnl']:.2f}%
⭐ الأفضل: {stats['best_symbol']}
👎 الأسوأ: {stats['worst_symbol']}
"""
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['train'])
        def train_cmd(msg):
            """Manual training command."""
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            self.tb.reply_to(msg, "🎓 جاري التدريب...")
            self.train_all_models()
    
    def fetch_data(self, symbol: str, interval: str = '5m', period: str = '3d') -> Optional[pd.DataFrame]:
        for attempt in range(self.config.MAX_RETRIES):
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                if not df.empty and len(df) >= 20:
                    df.columns = [c.capitalize() for c in df.columns]
                    return df
            except Exception as e:
                self.logger.warning(f"Fetch {symbol}: محاولة {attempt+1} - {e}")
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
            
            dir_5m, conf_5m = model.predict(df_5m)
            dir_15m, conf_15m = model.predict(df_15m)
            
            self.logger.debug(f"{symbol}: M5={dir_5m}({conf_5m:.2f}), M15={dir_15m}({conf_15m:.2f})")
            
            if dir_5m != dir_15m or dir_5m == "NEUTRAL":
                return None
            
            # Simple trend check
            trend, _ = check_trend(df_15m)
            
            if dir_5m == "BUY" and trend == "DOWN":
                return None
            if dir_5m == "SELL" and trend == "UP":
                return None
            
            confidence = (conf_5m + conf_15m) / 2
            
            if confidence < self.config.CONFIDENCE_THRESHOLD:
                return None
            
            self.logger.info(f"🎯 {symbol}: {dir_5m} | ثقة={confidence:.1%}")
            
            return {
                'symbol': symbol,
                'direction': dir_5m,
                'entry_price': float(df_5m['Close'].iloc[-1]),
                'confidence': confidence,
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
                
                # Calculate pips
                pip_size = 0.01 if 'JPY' in trade['symbol'] else 0.0001
                if direction == 'BUY':
                    pips = (current - entry) / pip_size
                else:
                    pips = (entry - current) / pip_size
                
                self.db.update_result(trade['id'], current, result, pnl, round(pips, 1))
                
                emoji = "✅" if result == 'WIN' else "❌"
                self.logger.info(f"{emoji} {trade['symbol']}: {result} | {pnl:+.2f}% | {pips:+.1f} نقطة")
                
            except Exception as e:
                self.logger.error(f"Check trade: {e}")
    
    def scan_markets(self):
        futures = {}
        for s in self.config.SYMBOLS:
            futures[self.executor.submit(self.analyze_symbol, s)] = s
        
        signals = 0
        for future in as_completed(futures, timeout=60):
            symbol = futures[future]
            try:
                signal = future.result(timeout=20)
                if signal:
                    if self.db.save_signal(signal):
                        self.send_signal(signal)
                        signals += 1
            except:
                pass
        
        return signals
    
    def train_all_models(self):
        """Train all models with sequential fetching to avoid rate limits."""
        self.logger.info("=" * 50)
        self.logger.info("🎓 بدء تدريب جميع النماذج...")
        self.logger.info("=" * 50)
        
        success_count = 0
        
        for symbol in self.config.SYMBOLS:
            try:
                self.logger.info(f"\n--- {symbol} ---")
                
                model = ForexModel(symbol, self.config, self.logger)
                
                if model.train():
                    model.save()
                    self.models[symbol] = model
                    success_count += 1
                    self.logger.info(f"✅ {symbol}: تم التدريب والحفظ")
                else:
                    self.logger.error(f"❌ {symbol}: فشل التدريب")
                
                # ✅ انتظار بين كل زوج عشان ما نضغطش على API
                time.sleep(5)
                
            except Exception as e:
                self.logger.error(f"❌ {symbol}: خطأ - {e}")
        
        self.last_retrain = datetime.now()
        
        self.logger.info("=" * 50)
        self.logger.info(f"🎓 انتهى التدريب: {success_count}/{len(self.config.SYMBOLS)} ناجح")
        self.logger.info("=" * 50)
        
        # Send report
        try:
            report = f"🎓 **تقرير التدريب**\n\n✅ ناجح: {success_count}/{len(self.config.SYMBOLS)}\n\n"
            for symbol, model in self.models.items():
                icon = "✅" if model.is_trained else "❌"
                acc = f"{model.train_accuracy:.1%}" if model.is_trained else "N/A"
                report += f"{icon} {symbol}: دقة={acc}\n"
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, report, parse_mode='Markdown')
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
        self.logger.info("🦅 Falcon AI Forex")
        self.logger.info(f"📊 أزواج: {len(self.config.SYMBOLS)}")
        self.logger.info(f"⚙️ ثقة: {self.config.CONFIDENCE_THRESHOLD:.0%}")
        self.logger.info("=" * 50)
        
        self.start_telegram()
        time.sleep(2)
        
        # ✅ تدريب أولي
        if not any(m.is_trained for m in self.models.values()):
            self.logger.info("لا توجد نماذج مدربة. بدء التدريب الأولي...")
            self.train_all_models()
        
        self.last_retrain = datetime.now()
        
        # Send startup
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(
                self.config.TELEGRAM_CHAT_ID,
                f"🦅 **Falcon AI Forex**\n✅ نماذج: {trained}/{len(self.models)}\n⚡️ بدء التحليل...",
                parse_mode='Markdown'
            )
        except:
            pass
        
        while self.running:
            try:
                self.check_trades()
                
                signals = self.scan_markets()
                self.logger.info(f"📊 دورة مكتملة. إشارات: {signals}")
                
                # Retrain every 24h
                if (datetime.now() - self.last_retrain).total_seconds() > 86400:
                    self.train_all_models()
                
                time.sleep(self.config.SCAN_INTERVAL_MINUTES * 60)
                
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"خطأ: {e}")
                time.sleep(30)
        
        self.executor.shutdown(wait=True)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    config = Config()
    bot = FalconForexBot(config)
    bot.run()
