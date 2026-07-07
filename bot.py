#!/usr/bin/env python3
"""
Falcon AI Pro v5.3 - Continuous Monitoring
============================================
No sleep. Always watching. Instant signals.
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
import signal
import traceback
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import requests

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
# CONFIG - NO SLEEP MODE
# ============================================================================

@dataclass
class Config:
    TELEGRAM_TOKEN: str = os.environ.get('TELEGRAM_TOKEN', '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '7553333305')
    
    MAX_PIPS_STOP_LOSS: int = 15
    MIN_RISK_REWARD: float = 1.5
    TRADE_DURATION_MINUTES: int = 10
    
    # ✅ بدل ما ينام 3 دقائق، يفحص كل 30 ثانية
    SCAN_INTERVAL_SECONDS: int = 30  # فحص كل 30 ثانية
    CANDLE_CLOSE_BUFFER: int = 10  # انتظار 10 ثواني بعد إغلاق الشمعة
    
    SYMBOLS: List[str] = field(default_factory=lambda: [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
        'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X'
    ])
    
    TRAINING_PERIOD_1H: str = '6mo'
    TRAINING_PERIOD_15M: str = '1mo'
    
    CONFIDENCE_THRESHOLD: float = 0.65  # أقل شوية عشان إشارات أكثر
    ENSEMBLE_AGREEMENT: int = 2  # 2 من 4 بدل 3
    MIN_TRAINING_SAMPLES: int = 500
    MAX_FEATURES: int = 50  # أقل عشان أسرع
    FORECAST_PERIODS: int = 5
    
    DB_PATH: str = 'falcon_trading.db'
    MODELS_DIR: str = 'models'
    
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 3
    MAX_WORKERS: int = 4  # عمال أكتر عشان فحص أسرع
    SIGNAL_COOLDOWN_MINUTES: int = 8  # تهدئة أقل
    
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
# TELEGRAM MANAGER
# ============================================================================

class TelegramManager:
    def __init__(self, token: str):
        self.token = token
        self.bot = None
        self._cleanup()
    
    def _cleanup(self):
        try:
            requests.get(f'https://api.telegram.org/bot{self.token}/deleteWebhook')
            time.sleep(0.5)
        except:
            pass
    
    def get_bot(self) -> telebot.TeleBot:
        if self.bot is None:
            self.bot = telebot.TeleBot(self.token)
        return self.bot
    
    def start_polling(self):
        bot = self.get_bot()
        
        def poll_worker():
            while True:
                try:
                    bot.infinity_polling(timeout=10, long_polling_timeout=5)
                except:
                    time.sleep(5)
        
        threading.Thread(target=poll_worker, daemon=True).start()

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
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    stop_loss REAL, take_profit REAL,
                    confidence REAL, entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, pnl_pips REAL, signal_hash TEXT UNIQUE
                )
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
                     confidence, expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data.get('stop_loss'), data.get('take_profit'),
                      data['confidence'], data['expiry_time'], signal_hash))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except:
            return None
    
    def update_result(self, signal_id: int, exit_price: float, result: str, pnl: float, pips: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE signals SET exit_price=?, result=?, pnl_percent=?, pnl_pips=?,
                exit_time=datetime('now', 'localtime') WHERE id=?
            ''', (exit_price, result, pnl, pips, signal_id))
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
            return {
                'total': total, 'wins': wins, 'losses': total-wins,
                'win_rate': wins/total if total > 0 else 0,
                'total_pnl': total_pnl, 'total_pips': total_pips
            }

# ============================================================================
# MARKET FILTER
# ============================================================================

class MarketFilter:
    @staticmethod
    def can_trade() -> Tuple[bool, str]:
        now = datetime.utcnow()
        day = now.weekday()
        hour = now.hour
        
        if day >= 5:
            return False, "ويكند"
        if day == 4 and hour >= 20:
            return False, "إغلاق جمعة"
        
        return True, "مسموح"

# ============================================================================
# CANDLE TIMER - أهم حاجة!
# ============================================================================

class CandleTimer:
    """
    ✅ يتأكد إن الإشارة بتطلع فور إغلاق الشمعة
    مش اثناء النوم ولا بعد فوات الأوان
    """
    
    @staticmethod
    def seconds_to_next_candle(interval_minutes: int = 5) -> int:
        """كم ثانية باقية على إغلاق الشمعة القادمة"""
        now = datetime.utcnow()
        current_minute = now.minute
        current_second = now.second
        
        # الشمعة القادمة
        next_candle_minute = ((current_minute // interval_minutes) + 1) * interval_minutes
        
        if next_candle_minute >= 60:
            next_candle_minute = 0
        
        # الوقت المتبقي
        if next_candle_minute > current_minute:
            minutes_left = next_candle_minute - current_minute - 1
            seconds_left = 60 - current_second
        else:
            minutes_left = (60 - current_minute) + next_candle_minute - 1
            seconds_left = 60 - current_second
        
        total_seconds = minutes_left * 60 + seconds_left
        
        return max(0, total_seconds)
    
    @staticmethod
    def is_candle_just_closed(interval_minutes: int = 5, buffer_seconds: int = 15) -> bool:
        """
        ✅ هل الشمعة لسه مقفلة حالاً؟
        لو أيوه، ده أفضل وقت للإشارة
        """
        now = datetime.utcnow()
        current_minute = now.minute
        current_second = now.second
        
        # لو في أول buffer_seconds بعد إغلاق الشمعة
        if current_minute % interval_minutes == 0 and current_second <= buffer_seconds:
            return True
        
        return False
    
    @staticmethod
    def wait_for_candle_close(interval_minutes: int = 5):
        """
        ✅ يستنى لحد ما الشمعة تقفل بالضبط
        عشان الإشارة تكون في التوقيت المثالي
        """
        seconds = CandleTimer.seconds_to_next_candle(interval_minutes)
        
        if seconds > 10:
            # لسه في وقت - استنى لحد ما يبقى 5 ثواني على الإغلاق
            time.sleep(max(0, seconds - 5))
        
        # انتظر لحد ما الثواني تبقى 0-10 (وقت الإغلاق)
        while True:
            now = datetime.utcnow()
            if now.second <= 10 and now.minute % interval_minutes == 0:
                break
            time.sleep(0.5)

# ============================================================================
# FEATURES (FAST VERSION)
# ============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l, o = df['Close'], df['High'], df['Low'], df['Open']
    
    for p in [1, 3, 5, 10]:
        f[f'ret_{p}'] = c.pct_change(p)
    
    for p in [5, 10, 20, 50]:
        f[f'sma_{p}'] = c.rolling(p).mean()
        f[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
    
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    f['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    f['macd'] = ema12 - ema26
    f['macd_signal'] = f['macd'].ewm(span=9).mean()
    
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
    
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    atr14 = tr.ewm(span=14).mean()
    plus_di = 100 * (plus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    minus_di = 100 * (minus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    f['adx'] = dx.ewm(span=14).mean()
    
    for p in [5, 10]:
        f[f'roc_{p}'] = (c - c.shift(p)) / (c.shift(p) + 1e-8) * 100
    
    return f.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# ============================================================================
# TARGET
# ============================================================================

def create_balanced_target(df: pd.DataFrame, periods: int) -> pd.Series:
    atr = (df['High'] - df['Low']).rolling(14).mean()
    future = df['Close'].shift(-periods)
    change = future - df['Close']
    threshold = atr * 0.5
    
    target = pd.Series(np.nan, index=df.index)
    
    buy_count = (change > threshold).sum()
    sell_count = (change < -threshold).sum()
    
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
# MODEL
# ============================================================================

class EnsembleModel:
    def __init__(self, symbol: str, config: Config, logger: logging.Logger):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        self.models = {}
        self.meta_model = None
        self.calibrators = {}
        self.scaler = RobustScaler()
        self.selected_features = []
        self.is_trained = False
        self.version = None
    
    def _init_models(self):
        self.models = {
            'xgb': xgb.XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=4,
                                      random_state=42, n_jobs=2, verbosity=0, tree_method='hist'),
            'cat': CatBoostClassifier(iterations=150, learning_rate=0.03, depth=4,
                                       random_seed=42, verbose=False, thread_count=2,
                                       allow_writing_files=False),
            'rf': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=2),
            'gb': GradientBoostingClassifier(n_estimators=150, learning_rate=0.03, max_depth=4, random_state=42)
        }
    
    def train(self, df: pd.DataFrame) -> bool:
        try:
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                return False
            
            features = calculate_features(df)
            target = create_balanced_target(df, self.config.FORECAST_PERIODS)
            
            valid = ~(features.isna().any(axis=1) | target.isna())
            X = features[valid]
            y = target[valid]
            
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
            
            base_preds = np.zeros((len(X_val), len(self.models)))
            
            for i, (name, model) in enumerate(self.models.items()):
                try:
                    if name == 'cat':
                        model.fit(X_train, y_train, verbose=False)
                    else:
                        model.fit(X_train, y_train)
                    base_preds[:, i] = model.predict_proba(X_val)[:, 1]
                    
                    self.calibrators[name] = CalibratedClassifierCV(model, cv=3, method='isotonic')
                    self.calibrators[name].fit(X_train, y_train)
                except:
                    base_preds[:, i] = 0.5
            
            self.meta_model = LogisticRegression()
            self.meta_model.fit(base_preds, y_val)
            
            self.is_trained = True
            self.version = datetime.now().strftime('v%Y%m%d_%H%M%S')
            return True
            
        except:
            return False
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.65) -> Tuple[str, float, int]:
        if not self.is_trained:
            return "NEUTRAL", 0.0, 0
        
        try:
            features = calculate_features(df).iloc[[-1]]
            available = [f for f in self.selected_features if f in features.columns]
            
            if len(available) < 10:
                return "NEUTRAL", 0.0, 0
            
            X = features[available].fillna(0)
            X_s = self.scaler.transform(X)
            
            base_probas = []
            votes = 0
            
            for name, cal in self.calibrators.items():
                try:
                    proba = float(cal.predict_proba(X_s)[0, 1])
                    base_probas.append(proba)
                    if proba > 0.5:
                        votes += 1
                except:
                    base_probas.append(0.5)
            
            proba_buy = float(self.meta_model.predict_proba(np.array([base_probas]))[0, 1])
            
            if proba_buy > threshold and votes >= self.config.ENSEMBLE_AGREEMENT:
                return "BUY", proba_buy, votes
            elif (1 - proba_buy) > threshold and (len(self.calibrators) - votes) >= self.config.ENSEMBLE_AGREEMENT:
                return "SELL", 1 - proba_buy, len(self.calibrators) - votes
            
            return "NEUTRAL", max(proba_buy, 1 - proba_buy), max(votes, len(self.calibrators) - votes)
            
        except:
            return "NEUTRAL", 0.0, 0
    
    def save(self):
        path = os.path.join(self.config.MODELS_DIR, self.symbol)
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'models': self.models,
            'meta_model': self.meta_model,
            'calibrators': self.calibrators,
            'scaler': self.scaler,
            'features': self.selected_features,
            'version': self.version
        }, os.path.join(path, 'model.pkl'))
    
    def load(self) -> bool:
        path = os.path.join(self.config.MODELS_DIR, self.symbol, 'model.pkl')
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.models = data['models']
        self.meta_model = data['meta_model']
        self.calibrators = data['calibrators']
        self.scaler = data['scaler']
        self.selected_features = data['features']
        self.version = data['version']
        self.is_trained = True
        return True

# ============================================================================
# MAIN BOT - NO SLEEP
# ============================================================================

class FalconProBot:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.db = Database(config.DB_PATH, self.logger)
        self.models = {}
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
        self.tg = TelegramManager(config.TELEGRAM_TOKEN)
        self.tb = self.tg.get_bot()
        self._setup_commands()
        
        for symbol in config.SYMBOLS:
            model = EnsembleModel(symbol, config, self.logger)
            if model.load():
                self.logger.info(f"📂 {symbol}")
            else:
                self.logger.info(f"🆕 {symbol}")
            self.models[symbol] = model
        
        self.running = False
        self.last_retrain = None
        self.last_scan_time = {}  # ✅ متى آخر مرة فحصنا كل زوج
    
    def _setup_commands(self):
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            trained = sum(1 for m in self.models.values() if m.is_trained)
            stats = self.db.get_stats()
            can_trade, reason = MarketFilter.can_trade()
            text = (f"🦅 **Falcon Pro v5.3**\n"
                   f"✅ نماذج: {trained}/{len(self.models)}\n"
                   f"📊 صفقات: {stats['total']}\n"
                   f"📈 نجاح: {stats['win_rate']:.1%}\n"
                   f"💰 ربح: {stats['total_pnl']:.2f}% | {stats['total_pips']:.0f}p\n"
                   f"🔄 فحص: كل {self.config.SCAN_INTERVAL_SECONDS}ث\n"
                   f"🚦 {'✅' if can_trade else '⛔ '+reason}")
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
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
    
    def calculate_sl_tp(self, symbol: str, entry: float, direction: str) -> Tuple[float, float]:
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
            can_trade, _ = MarketFilter.can_trade()
            if not can_trade:
                return None
            
            model = self.models.get(symbol)
            if not model or not model.is_trained:
                return None
            
            # ✅ لو لسه فحصنا الزوج ده من قريب، نتخطاه
            last_scan = self.last_scan_time.get(symbol)
            if last_scan and (datetime.now() - last_scan).total_seconds() < 25:
                return None
            
            self.last_scan_time[symbol] = datetime.now()
            
            if self.db.check_active_signal(symbol):
                return None
            
            if self.db.check_recent_signal(symbol, self.config.SIGNAL_COOLDOWN_MINUTES):
                return None
            
            df_5m = self.fetch_data(symbol, '5m', '3d')
            df_15m = self.fetch_data(symbol, '15m', '5d')
            
            if df_5m is None or df_15m is None:
                return None
            
            dir_5m, conf_5m, votes_5m = model.predict(df_5m, self.config.CONFIDENCE_THRESHOLD)
            dir_15m, conf_15m, votes_15m = model.predict(df_15m, self.config.CONFIDENCE_THRESHOLD)
            
            if dir_5m != dir_15m or dir_5m == "NEUTRAL":
                return None
            
            confidence = (conf_5m + conf_15m) / 2
            
            if confidence < self.config.CONFIDENCE_THRESHOLD:
                return None
            
            entry_price = float(df_5m['Close'].iloc[-1])
            stop_loss, take_profit = self.calculate_sl_tp(symbol, entry_price, dir_5m)
            
            self.logger.info(f"🎯 {symbol}: {dir_5m} | ثقة={confidence:.1%} | صوت={votes_5m}+{votes_15m}")
            
            return {
                'symbol': symbol,
                'direction': dir_5m,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence,
                'expiry_time': (datetime.now() + timedelta(minutes=self.config.TRADE_DURATION_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except:
            return None
    
    def send_signal(self, signal: Dict):
        try:
            emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
            direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
            
            msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
                   f"💰 دخول: {signal['entry_price']:.5f}\n"
                   f"🛑 SL: {signal['stop_loss']:.5f}\n"
                   f"🎯 TP: {signal['take_profit']:.5f}\n"
                   f"💪 ثقة: {signal['confidence']:.1%}\n\n"
                   f"🤖 Falcon Pro v5.3")
            
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
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
                
                pip_value = 0.01 if 'JPY' in trade['symbol'] else 0.0001
                
                if direction == 'BUY':
                    pnl = (current - entry) / entry * 100
                    pips = (current - entry) / pip_value
                    result = 'WIN' if current > entry else 'LOSS'
                else:
                    pnl = (entry - current) / entry * 100
                    pips = (entry - current) / pip_value
                    result = 'WIN' if current < entry else 'LOSS'
                
                self.db.update_result(trade['id'], current, result, pnl, round(pips, 1))
            except:
                pass
    
    def scan_all_symbols(self):
        """✅ فحص كل الأزواج مرة واحدة"""
        futures = {self.executor.submit(self.analyze_symbol, s): s for s in self.config.SYMBOLS}
        signals = 0
        for future in as_completed(futures, timeout=30):
            try:
                signal = future.result(timeout=15)
                if signal and self.db.save_signal(signal):
                    self.send_signal(signal)
                    signals += 1
            except:
                pass
        return signals
    
    def train_all_models(self):
        for symbol in self.config.SYMBOLS:
            try:
                df = None
                for interval, period in [('1h', self.config.TRAINING_PERIOD_1H), 
                                          ('15m', self.config.TRAINING_PERIOD_15M)]:
                    df = self.fetch_data(symbol, interval, period)
                    if df is not None and len(df) >= self.config.MIN_TRAINING_SAMPLES:
                        break
                    time.sleep(2)
                
                if df is not None:
                    model = EnsembleModel(symbol, self.config, self.logger)
                    if model.train(df):
                        model.save()
                        self.models[symbol] = model
                
                time.sleep(2)
            except:
                pass
        
        self.last_retrain = datetime.now()
    
    def run(self):
        self.running = True
        
        self.logger.info("=" * 50)
        self.logger.info("🦅 Falcon Pro v5.3 - No Sleep")
        self.logger.info(f"🔄 فحص كل {self.config.SCAN_INTERVAL_SECONDS} ثانية")
        self.logger.info(f"🕯️ انتظار إغلاق الشمعة")
        self.logger.info("=" * 50)
        
        self.tg.start_polling()
        time.sleep(1)
        
        if not any(m.is_trained for m in self.models.values()):
            self.train_all_models()
        
        self.last_retrain = datetime.now()
        
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID,
                f"🦅 **Falcon Pro v5.3**\n✅ {trained}/{len(self.config.SYMBOLS)}\n🔄 فحص مستمر\n⚡️ يعمل...",
                parse_mode='Markdown')
        except:
            pass
        
        # ============================================
        # ✅ الحلقة الرئيسية - بدون نوم طويل
        # ============================================
        while self.running:
            try:
                # ✅ فحص الصفقات المنتهية
                self.check_trades()
                
                # ✅ فحص كل الأزواج
                signals = self.scan_all_symbols()
                
                # ✅ إعادة تدريب يومي
                if (datetime.now() - self.last_retrain).total_seconds() > 86400:
                    self.train_all_models()
                
                # ✅ لو في إشارات، سجل. لو مفيش، نام شوية صغيرين
                if signals > 0:
                    self.logger.info(f"✅ إشارات: {signals}")
                else:
                    # نام 5 ثواني بس بدل 3 دقائق!
                    time.sleep(5)
                    continue
                
                # ✅ انتظار قصير جداً بين الدورات
                time.sleep(self.config.SCAN_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"خطأ: {e}")
                time.sleep(5)  # لو فيه خطأ، نام 5 ثواني وجرب تاني
        
        self.executor.shutdown(wait=True)
        self.logger.info("🛑 توقف")

# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    while True:
        try:
            bot = FalconProBot(config)
            bot.run()
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ خطأ: {e}")
            traceback.print_exc()
            print("🔄 إعادة التشغيل...")
            time.sleep(5)

if __name__ == "__main__":
    main()
