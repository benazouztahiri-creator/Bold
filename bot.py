#!/usr/bin/env python3
"""
Falcon AI Pro v5.0 - Maximum Profit, Minimum Loss
====================================================
- Strict Stop Loss
- News Filter
- Liquidity Filter
- Triple Confirmation
- Fixed Risk Management
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier

import telebot
import joblib

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '2'

# ============================================================================
# CONFIG - RISK MANAGEMENT
# ============================================================================

@dataclass
class Config:
    TELEGRAM_TOKEN: str = os.environ.get('TELEGRAM_TOKEN', '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '7553333305')
    
    # ✅ Risk Management
    MAX_PIPS_STOP_LOSS: int = 15  # أقصى خسارة بالنقاط
    MAX_LOSS_PERCENT: float = 0.15  # أقصى خسارة 0.15%
    RISK_PER_TRADE: float = 0.01  # 1% مخاطرة
    MIN_RISK_REWARD: float = 1.5  # نسبة الربح للخسارة 1:1.5
    
    TRADE_DURATION_MINUTES: int = 10
    SCAN_INTERVAL_MINUTES: int = 3
    
    SYMBOLS: List[str] = field(default_factory=lambda: [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
        'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X'
    ])
    
    TRAINING_PERIOD_1H: str = '6mo'
    TRAINING_PERIOD_15M: str = '1mo'
    
    # ✅ Strict thresholds
    CONFIDENCE_THRESHOLD: float = 0.70  # عتبة عالية
    ENSEMBLE_AGREEMENT: int = 3  # 3 من 4 لازم يتفقوا
    
    WALK_FORWARD_WINDOWS: int = 3
    MIN_TRAINING_SAMPLES: int = 500
    MAX_FEATURES: int = 60
    FORECAST_PERIODS: int = 5
    
    DB_PATH: str = 'falcon_trading.db'
    MODELS_DIR: str = 'models'
    
    MAX_RETRIES: int = 5
    RETRY_DELAY: int = 10
    MAX_WORKERS: int = 2
    SIGNAL_COOLDOWN_MINUTES: int = 15  # تهدئة أطول
    
    LOG_FILE: str = 'falcon_bot.log'

# ============================================================================
# NEWS & LIQUIDITY FILTER
# ============================================================================

class MarketFilter:
    """Filter bad trading times."""
    
    # ✅ أوقات الأخبار القوية (بتوقيت UTC)
    HIGH_IMPACT_NEWS_TIMES = [
        (12, 30),  # USD News 8:30 AM ET
        (14, 0),   # FOMC 2:00 PM ET
        (8, 30),   # UK News 8:30 AM GMT
        (9, 30),   # UK GDP 9:30 AM GMT
        (10, 0),   # EU News 10:00 AM GMT
    ]
    
    @staticmethod
    def is_news_time() -> bool:
        """✅ لو في خبر قوي خلال 15 دقيقة، ما نتاجرش"""
        now = datetime.utcnow()
        
        for hour, minute in MarketFilter.HIGH_IMPACT_NEWS_TIMES:
            news_time = now.replace(hour=hour, minute=minute, second=0)
            
            # قبل الخبر بـ 15 دقيقة أو بعده بـ 15 دقيقة
            if abs((now - news_time).total_seconds()) < 900:  # 15 minutes
                return True
        
        return False
    
    @staticmethod
    def is_low_liquidity() -> bool:
        """✅ أوقات سيولة ضعيفة"""
        now = datetime.utcnow()
        day = now.weekday()  # 0=Monday, 4=Friday, 5=Saturday, 6=Sunday
        hour = now.hour
        
        # نهاية الأسبوع
        if day >= 5:  # Saturday-Sunday
            return True
        
        # إغلاق الجمعة
        if day == 4 and hour >= 20:
            return True
        
        # افتتاح الأسبوع (أول ساعة)
        if day == 0 and hour < 1:
            return True
        
        # نهاية جلسة آسيا - بداية أوروبا (سيولة منخفضة)
        if 6 <= hour <= 7:
            return True
        
        return False
    
    @staticmethod
    def can_trade() -> Tuple[bool, str]:
        """✅ هل مسموح بالتداول دلوقتي؟"""
        if MarketFilter.is_low_liquidity():
            return False, "سيولة منخفضة"
        
        if MarketFilter.is_news_time():
            return False, "قرب صدور أخبار"
        
        return True, "مسموح"

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
                    symbol TEXT, direction TEXT, entry_price REAL,
                    stop_loss REAL, take_profit REAL,
                    confidence REAL, proba_buy REAL, proba_sell REAL,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, exit_time DATETIME,
                    result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, pnl_pips REAL,
                    signal_hash TEXT UNIQUE
                );
                CREATE TABLE IF NOT EXISTS performance (
                    symbol TEXT PRIMARY KEY,
                    total_signals INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    total_pips REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
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
                    (symbol, direction, entry_price, stop_loss, take_profit,
                     confidence, proba_buy, proba_sell, expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data.get('stop_loss'), data.get('take_profit'),
                      data['confidence'], data.get('proba_buy', 0), 
                      data.get('proba_sell', 0), data['expiry_time'], signal_hash))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except:
            return None
    
    def update_result(self, signal_id: int, exit_price: float, result: str, 
                      pnl: float, pips: float, symbol: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE signals SET exit_price=?, result=?, pnl_percent=?, pnl_pips=?,
                exit_time=datetime('now', 'localtime') WHERE id=?
            ''', (exit_price, result, pnl, pips, signal_id))
            
            conn.execute('''
                INSERT INTO performance (symbol, total_signals, wins, losses, total_pnl, total_pips)
                VALUES (?, 1, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                total_signals = total_signals + 1,
                wins = wins + ?,
                losses = losses + ?,
                total_pnl = total_pnl + ?,
                total_pips = total_pips + ?
            ''', (symbol, 
                  1 if result == 'WIN' else 0, 1 if result == 'LOSS' else 0, pnl, pips,
                  1 if result == 'WIN' else 0, 1 if result == 'LOSS' else 0, pnl, pips))
            conn.commit()
    
    def check_active_signal(self, symbol: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' 
                AND expiry_time > datetime('now', 'localtime')
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
            total_pnl = conn.execute("SELECT SUM(pnl_percent) FROM signals WHERE result!='PENDING'").fetchone()[0] or 0
            total_pips = conn.execute("SELECT SUM(pnl_pips) FROM signals WHERE result!='PENDING'").fetchone()[0] or 0
            buys = conn.execute("SELECT COUNT(*) FROM signals WHERE result!='PENDING' AND direction='BUY'").fetchone()[0]
            sells = conn.execute("SELECT COUNT(*) FROM signals WHERE result!='PENDING' AND direction='SELL'").fetchone()[0]
            
            return {
                'total': total, 'wins': wins, 'losses': total-wins,
                'win_rate': wins/total if total > 0 else 0,
                'total_pnl': total_pnl, 'total_pips': total_pips,
                'buy_count': buys, 'sell_count': sells
            }

# ============================================================================
# FEATURES
# ============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l, o = df['Close'], df['High'], df['Low'], df['Open']
    
    for p in [1, 3, 5, 10, 20]:
        f[f'ret_{p}'] = c.pct_change(p)
    
    for p in [5, 10, 20, 50]:
        f[f'sma_{p}'] = c.rolling(p).mean()
        f[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
        f[f'dist_sma_{p}'] = (c - f[f'sma_{p}']) / (f[f'sma_{p}'] + 1e-8)
    
    for p in [7, 14]:
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).rolling(p).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(p).mean()
        f[f'rsi_{p}'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    f['macd'] = ema12 - ema26
    f['macd_signal'] = f['macd'].ewm(span=9).mean()
    f['macd_hist'] = f['macd'] - f['macd_signal']
    
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f['bb_pos'] = (c - sma20) / (2 * std20 + 1e-8)
    
    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    f['atr'] = tr.ewm(span=14).mean()
    
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    f['stoch_k'] = 100 * (c - low14) / (high14 - low14 + 1e-8)
    
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    f['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-8)
    
    for p in [5, 10]:
        f[f'roc_{p}'] = (c - c.shift(p)) / (c.shift(p) + 1e-8) * 100
    
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    atr14 = tr.ewm(span=14).mean()
    plus_di = 100 * (plus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    minus_di = 100 * (minus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    f['adx'] = dx.ewm(span=14).mean()
    
    f['body_size'] = abs(c - o) / (h - l + 1e-8)
    f['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-8)
    f['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-8)
    
    return f.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# ============================================================================
# BALANCED TARGET
# ============================================================================

def create_balanced_target(df: pd.DataFrame, periods: int) -> pd.Series:
    atr = (df['High'] - df['Low']).rolling(14).mean()
    future = df['Close'].shift(-periods)
    change = future - df['Close']
    threshold = atr * 0.5
    
    buy_count = (change > threshold).sum()
    sell_count = (change < -threshold).sum()
    
    target = pd.Series(np.nan, index=df.index)
    
    if buy_count > sell_count * 1.5:
        target[change > threshold * 1.3] = 1
        target[change < -threshold] = 0
    elif sell_count > buy_count * 1.5:
        target[change > threshold] = 1
        target[change < -threshold * 1.3] = 0
    else:
        target[change > threshold] = 1
        target[change < -threshold] = 0
    
    return target

# ============================================================================
# ENSEMBLE MODEL
# ============================================================================

class EnsembleModel:
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
        self.buy_sell_ratio = 1.0
    
    def _init_models(self):
        self.base_models = {
            'xgboost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=5,
                                          random_state=42, n_jobs=2, verbosity=0, tree_method='hist'),
            'catboost': CatBoostClassifier(iterations=200, learning_rate=0.03, depth=5,
                                            random_seed=42, verbose=False, thread_count=2,
                                            allow_writing_files=False, auto_class_weights='Balanced'),
            'randomforest': RandomForestClassifier(n_estimators=200, max_depth=10,
                                                    random_state=42, n_jobs=2, class_weight='balanced'),
            'gradient_boost': GradientBoostingClassifier(n_estimators=200, learning_rate=0.03,
                                                          max_depth=5, random_state=42)
        }
    
    def train(self, df: pd.DataFrame) -> bool:
        try:
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                return False
            
            self.logger.info(f"🎓 {self.symbol}: {len(df)} عينة...")
            
            features = calculate_features(df)
            target = create_balanced_target(df, self.config.FORECAST_PERIODS)
            
            valid = ~(features.isna().any(axis=1) | target.isna())
            X = features[valid]
            y = target[valid]
            
            buy_pct = y.mean()
            self.buy_sell_ratio = buy_pct / (1 - buy_pct) if buy_pct > 0 and buy_pct < 1 else 1.0
            
            if len(X) < 200:
                return False
            
            mi = mutual_info_classif(X, y, random_state=42)
            scores = sorted(zip(X.columns, mi), key=lambda x: x[1], reverse=True)
            self.selected_features = [s[0] for s in scores[:self.config.MAX_FEATURES]]
            X = X[self.selected_features]
            
            self._init_models()
            X_s = self.scaler.fit_transform(X)
            
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X_s[:split_idx], X_s[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            base_preds = np.zeros((len(X_val), len(self.base_models)))
            
            for i, (name, model) in enumerate(self.base_models.items()):
                try:
                    if name == 'catboost':
                        model.fit(X_train, y_train, verbose=False)
                    else:
                        model.fit(X_train, y_train)
                    base_preds[:, i] = model.predict_proba(X_val)[:, 1]
                    
                    self.calibrators[name] = CalibratedClassifierCV(model, cv=3, method='isotonic')
                    self.calibrators[name].fit(X_train, y_train)
                except:
                    base_preds[:, i] = 0.5
            
            self.meta_model = LogisticRegression(class_weight='balanced')
            self.meta_model.fit(base_preds, y_val)
            
            self.is_trained = True
            self.version = datetime.now().strftime('v%Y%m%d_%H%M%S')
            
            meta_pred = self.meta_model.predict(base_preds)
            acc = accuracy_score(y_val, meta_pred)
            
            self.logger.info(f"✅ {self.symbol}: دقة={acc:.1%}, BUY/SELL={buy_pct:.1%}/{1-buy_pct:.1%}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.symbol}: {e}")
            return False
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.70) -> Tuple[str, float, float, float, int]:
        """
        ✅ يرجع: (الاتجاه, الثقة, proba_buy, proba_sell, عدد النماذج الموافقة)
        """
        if not self.is_trained:
            return "NEUTRAL", 0.0, 0.0, 0.0, 0
        
        try:
            features = calculate_features(df).iloc[[-1]]
            available = [f for f in self.selected_features if f in features.columns]
            
            if len(available) < 10:
                return "NEUTRAL", 0.0, 0.0, 0.0, 0
            
            X = features[available].fillna(0)
            X_s = self.scaler.transform(X)
            
            base_probas = []
            model_votes = []
            
            for name, cal in self.calibrators.items():
                try:
                    proba = float(cal.predict_proba(X_s)[0, 1])
                    base_probas.append(proba)
                    model_votes.append(1 if proba > 0.5 else 0)
                except:
                    base_probas.append(0.5)
                    model_votes.append(0)
            
            proba_buy = float(self.meta_model.predict_proba(np.array([base_probas]))[0, 1])
            
            # ✅ تعديل حسب نسبة البيانات
            if self.buy_sell_ratio > 1.3:
                proba_buy *= 0.95
            elif self.buy_sell_ratio < 0.7:
                proba_buy *= 1.05
            
            proba_buy = np.clip(proba_buy, 0.05, 0.95)
            proba_sell = 1 - proba_buy
            
            # ✅ عدد النماذج الموافقة
            buy_votes = sum(model_votes)
            sell_votes = len(model_votes) - buy_votes
            
            if proba_buy > threshold and buy_votes >= self.config.ENSEMBLE_AGREEMENT:
                return "BUY", proba_buy, proba_buy, proba_sell, buy_votes
            elif proba_sell > threshold and sell_votes >= self.config.ENSEMBLE_AGREEMENT:
                return "SELL", proba_sell, proba_buy, proba_sell, sell_votes
            
            return "NEUTRAL", max(proba_buy, proba_sell), proba_buy, proba_sell, max(buy_votes, sell_votes)
            
        except:
            return "NEUTRAL", 0.0, 0.0, 0.0, 0
    
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
            'buy_sell_ratio': self.buy_sell_ratio
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
        self.buy_sell_ratio = data.get('buy_sell_ratio', 1.0)
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
        self._remove_webhook()
        self._setup_commands()
        
        for symbol in config.SYMBOLS:
            model = EnsembleModel(symbol, config, self.logger)
            loaded = model.load()
            self.logger.info(f"{'📂' if loaded else '🆕'} {symbol}")
            self.models[symbol] = model
        
        self.running = False
        self.last_retrain = None
    
    def _remove_webhook(self):
        try:
            self.tb.remove_webhook()
            time.sleep(1)
        except:
            pass
    
    def _setup_commands(self):
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            trained = sum(1 for m in self.models.values() if m.is_trained)
            can_trade, reason = MarketFilter.can_trade()
            stats = self.db.get_stats()
            text = (f"🦅 **Falcon Pro v5**\n"
                   f"✅ نماذج: {trained}/{len(self.models)}\n"
                   f"📊 صفقات: {stats['total']}\n"
                   f"📈 نجاح: {stats['win_rate']:.1%}\n"
                   f"💰 ربح: {stats['total_pnl']:.2f}% | {stats['total_pips']:.0f} نقطة\n"
                   f"🟢 شراء: {stats['buy_count']} | 🔴 بيع: {stats['sell_count']}\n"
                   f"🚦 التداول: {'✅ مسموح' if can_trade else '⛔ ' + reason}")
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['stats'])
        def stats_cmd(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            s = self.db.get_stats()
            self.tb.reply_to(msg, 
                f"📊 {s['total']} | ✅ {s['wins']} | 📈 {s['win_rate']:.1%}\n"
                f"💰 {s['total_pnl']:.2f}% | 📊 {s['total_pips']:.0f} نقطة\n"
                f"🟢 {s['buy_count']} | 🔴 {s['sell_count']}")
    
    def fetch_data(self, symbol: str, interval: str = '5m', period: str = '5d') -> Optional[pd.DataFrame]:
        for attempt in range(self.config.MAX_RETRIES):
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                if not df.empty:
                    df.columns = [c.capitalize() for c in df.columns]
                    return df
            except:
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY)
        return None
    
    def calculate_sl_tp(self, symbol: str, entry: float, direction: str, atr: float) -> Tuple[float, float]:
        """✅ حساب Stop Loss و Take Profit"""
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        
        if direction == 'BUY':
            stop_loss = entry - (self.config.MAX_PIPS_STOP_LOSS * pip_value)
            take_profit = entry + (self.config.MAX_PIPS_STOP_LOSS * self.config.MIN_RISK_REWARD * pip_value)
        else:
            stop_loss = entry + (self.config.MAX_PIPS_STOP_LOSS * pip_value)
            take_profit = entry - (self.config.MAX_PIPS_STOP_LOSS * self.config.MIN_RISK_REWARD * pip_value)
        
        return round(stop_loss, 5), round(take_profit, 5)
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        try:
            # ✅ فلتر السوق أول حاجة
            can_trade, reason = MarketFilter.can_trade()
            if not can_trade:
                return None
            
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
            
            # ✅ تنبؤات مع عدد النماذج الموافقة
            dir_5m, conf_5m, pb_5m, ps_5m, votes_5m = model.predict(df_5m, self.config.CONFIDENCE_THRESHOLD)
            dir_15m, conf_15m, pb_15m, ps_15m, votes_15m = model.predict(df_15m, self.config.CONFIDENCE_THRESHOLD)
            
            # ✅ Triple confirmation: M5 + M15 + Models
            if dir_5m != dir_15m or dir_5m == "NEUTRAL":
                return None
            
            if votes_5m < self.config.ENSEMBLE_AGREEMENT or votes_15m < self.config.ENSEMBLE_AGREEMENT:
                return None
            
            confidence = (conf_5m + conf_15m) / 2
            proba_buy = (pb_5m + pb_15m) / 2
            proba_sell = (ps_5m + ps_15m) / 2
            
            if confidence < self.config.CONFIDENCE_THRESHOLD:
                return None
            
            entry_price = float(df_5m['Close'].iloc[-1])
            atr = float(df_15m['High'].iloc[-1] - df_15m['Low'].iloc[-1])
            
            # ✅ حساب SL/TP
            stop_loss, take_profit = self.calculate_sl_tp(symbol, entry_price, dir_5m, atr)
            
            self.logger.info(f"🎯 {symbol}: {dir_5m} | ثقة={confidence:.1%} | "
                           f"نماذج={votes_5m}+{votes_15m} | SL={stop_loss:.5f} | TP={take_profit:.5f}")
            
            return {
                'symbol': symbol,
                'direction': dir_5m,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence,
                'proba_buy': proba_buy,
                'proba_sell': proba_sell,
                'votes': f"{votes_5m}+{votes_15m}",
                'expiry_time': (datetime.now() + timedelta(minutes=self.config.TRADE_DURATION_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Analyze {symbol}: {e}")
            return None
    
    def send_signal(self, signal: Dict):
        try:
            emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
            direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
            
            msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
                   f"💰 الدخول: {signal['entry_price']:.5f}\n"
                   f"🛑 إيقاف خسارة: {signal['stop_loss']:.5f}\n"
                   f"🎯 هدف ربح: {signal['take_profit']:.5f}\n"
                   f"⏳ المدة: {self.config.TRADE_DURATION_MINUTES} د\n"
                   f"💪 الثقة: {signal['confidence']:.1%}\n"
                   f"🗳 النماذج: {signal['votes']}/8\n"
                   f"📊 P(BUY): {signal['proba_buy']:.1%} | P(SELL): {signal['proba_sell']:.1%}\n\n"
                   f"🤖 Falcon Pro v5")
            
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
            self.logger.info(f"✅ إشارة: {signal['symbol']} {signal['direction']} | SL={signal['stop_loss']:.5f}")
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
                direction = trade['direction']
                
                # ✅ حساب النقاط
                pip_value = 0.01 if 'JPY' in trade['symbol'] else 0.0001
                
                if direction == 'BUY':
                    pnl = (current - entry) / entry * 100
                    pips = (current - entry) / pip_value
                    result = 'WIN' if current > entry else 'LOSS'
                else:
                    pnl = (entry - current) / entry * 100
                    pips = (entry - current) / pip_value
                    result = 'WIN' if current < entry else 'LOSS'
                
                self.db.update_result(trade['id'], current, result, pnl, round(pips, 1), trade['symbol'])
                
                emoji = "✅" if result == 'WIN' else "❌"
                self.logger.info(f"{emoji} {trade['symbol']}: {result} | {pnl:+.2f}% | {pips:+.1f} نقطة")
                
                # ✅ إرسال إشعار بالنتيجة
                try:
                    self.tb.send_message(
                        self.config.TELEGRAM_CHAT_ID,
                        f"{emoji} **{trade['symbol']}** - {trade['direction']}\n"
                        f"النتيجة: {result}\n"
                        f"الربح/الخسارة: {pnl:+.2f}% | {pips:+.1f} نقطة\n"
                        f"{entry:.5f} → {current:.5f}",
                        parse_mode='Markdown'
                    )
                except:
                    pass
                
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
        self.logger.info("🎓 تدريب...")
        
        for symbol in self.config.SYMBOLS:
            try:
                df = None
                for interval, period in [('1h', self.config.TRAINING_PERIOD_1H), 
                                          ('15m', self.config.TRAINING_PERIOD_15M)]:
                    df = self.fetch_data(symbol, interval, period)
                    if df is not None and len(df) >= self.config.MIN_TRAINING_SAMPLES:
                        break
                    time.sleep(3)
                
                if df is not None:
                    model = EnsembleModel(symbol, self.config, self.logger)
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
                f"🎓 **تدريب مكتمل**\n✅ {trained}/{len(self.config.SYMBOLS)}",
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
        self.logger.info("🦅 Falcon AI Pro v5.0 - Risk Managed")
        self.logger.info(f"🛑 Stop Loss: {self.config.MAX_PIPS_STOP_LOSS} pips")
        self.logger.info(f"📊 Risk/Reward: 1:{self.config.MIN_RISK_REWARD}")
        self.logger.info(f"🗳 Ensemble Agreement: {self.config.ENSEMBLE_AGREEMENT}/4")
        self.logger.info(f"🚦 News & Liquidity Filters: ACTIVE")
        self.logger.info("=" * 50)
        
        self.start_telegram()
        time.sleep(2)
        
        if not any(m.is_trained for m in self.models.values()):
            self.train_all_models()
        
        self.last_retrain = datetime.now()
        
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID,
                f"🦅 **Falcon Pro v5**\n✅ {trained}/{len(self.config.SYMBOLS)}\n"
                f"🛑 SL: {self.config.MAX_PIPS_STOP_LOSS}pips\n"
                f"🚦 Filters Active\n⚡️ Scanning...",
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
# RUN
# ============================================================================

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    config = Config()
    bot = FalconProBot(config)
    bot.run()
