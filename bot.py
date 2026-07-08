#!/usr/bin/env python3
"""
Falcon AI Pro v8.0 - Fixed Buy/Sell Logic
============================================
Corrected target and prediction logic.
BUY = price goes UP. SELL = price goes DOWN.
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
import requests

# River
from river import compose, preprocessing, tree

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
# CONFIG
# ============================================================================

@dataclass
class Config:
    TELEGRAM_TOKEN: str = os.environ.get('TELEGRAM_TOKEN', '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '7553333305')
    
    SCAN_INTERVAL_SECONDS: int = 60
    MIN_TRADE_DURATION: int = 3
    MAX_TRADE_DURATION: int = 15
    
    SYMBOLS: List[str] = field(default_factory=lambda: [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
        'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X'
    ])
    
    TRAINING_PERIOD_1H: str = '3mo'
    TRAINING_PERIOD_15M: str = '1mo'
    
    CONFIDENCE_THRESHOLD: float = 0.60
    ENSEMBLE_AGREEMENT: int = 2
    MIN_TRAINING_SAMPLES: int = 300
    MAX_FEATURES: int = 20
    FORECAST_PERIODS: int = 5
    
    DB_PATH: str = 'falcon_trading.db'
    MODELS_DIR: str = 'models'
    
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5
    MAX_WORKERS: int = min(2, os.cpu_count() or 2)
    SIGNAL_COOLDOWN_MINUTES: int = 10
    RIVER_UPDATE_INTERVAL: int = 200
    RETRAINING_INTERVAL_SECONDS: int = 604800
    
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
            requests.get(f'https://api.telegram.org/bot{self.token}/deleteWebhook', timeout=3)
            time.sleep(0.3)
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
    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    confidence REAL, proba_buy REAL, proba_sell REAL,
                    trade_duration INTEGER,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, signal_hash TEXT UNIQUE
                )
            ''')
            conn.commit()
    
    def save(self, data: Dict) -> bool:
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO signals 
                    (symbol, direction, entry_price, confidence, proba_buy, proba_sell,
                     trade_duration, expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data.get('proba_buy', 0), data.get('proba_sell', 0),
                      data['trade_duration'], data['expiry_time'], h))
                conn.commit()
            return True
        except:
            return False
    
    def update(self, signal_id: int, exit_price: float, result: str, pnl: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE signals SET exit_price=?, result=?, pnl_percent=?,
                exit_time=datetime('now', 'localtime') WHERE id=?
            ''', (exit_price, result, pnl, signal_id))
            conn.commit()
    
    def has_active(self, symbol: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' 
                AND expiry_time > datetime('now', 'localtime')
            ''', (symbol,)).fetchone()[0]
            return c > 0
    
    def was_recent(self, symbol: str, minutes: int) -> bool:
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?
            ''', (symbol, cutoff)).fetchone()[0]
            return c > 0
    
    def get_expired(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM signals WHERE result='PENDING' 
                AND expiry_time <= datetime('now', 'localtime')
            ''').fetchall()
            return [dict(r) for r in rows]
    
    def stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM signals WHERE result!='PENDING'").fetchone()[0]
            wins = conn.execute("SELECT COUNT(*) FROM signals WHERE result='WIN'").fetchone()[0]
            return {'total': total, 'wins': wins, 'losses': total-wins,
                    'win_rate': wins/total if total > 0 else 0}

# ============================================================================
# FEATURES
# ============================================================================

_features_cache = {}
_features_cache_ttl = 30

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > 0:
        cache_key = hash(str(df.index[-1]) + str(len(df)))
        if cache_key in _features_cache:
            cached_time, cached_data = _features_cache[cache_key]
            if time.time() - cached_time < _features_cache_ttl:
                return cached_data.copy()
    
    f = pd.DataFrame(index=df.index)
    c, h, l = df['Close'], df['High'], df['Low']
    
    for p in [1, 3, 5]:
        f[f'ret_{p}'] = c.pct_change(p)
    for p in [5, 10, 20]:
        f[f'sma_{p}'] = c.rolling(p).mean()
        f[f'dist_{p}'] = (c - f[f'sma_{p}']) / (f[f'sma_{p}'] + 1e-8)
    
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    f['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    f['macd'] = ema12 - ema26
    
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f['bb'] = (c - sma20) / (2 * std20 + 1e-8)
    
    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    f['atr'] = tr.ewm(span=14).mean()
    
    # ADX
    atr14 = tr.ewm(span=14, adjust=False).mean()
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    plus_di = 100 * (plus_dm.ewm(span=14, adjust=False).mean()) / (atr14 + 1e-8)
    minus_di = 100 * (minus_dm.ewm(span=14, adjust=False).mean()) / (atr14 + 1e-8)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    f['adx'] = dx.ewm(span=14, adjust=False).mean()
    
    result = f.fillna(0)
    
    if len(df) > 0:
        _features_cache[cache_key] = (time.time(), result)
        if len(_features_cache) > 50:
            oldest = min(_features_cache, key=lambda k: _features_cache[k][0])
            del _features_cache[oldest]
    
    return result

# ============================================================================
# TARGET - CORRECTED
# ============================================================================

def create_target(df: pd.DataFrame, periods: int) -> pd.Series:
    """
    ✅ Target واضح:
    1 = BUY (السعر ارتفع)
    0 = SELL (السعر انخفض)
    NaN = محايد
    """
    future = df['Close'].shift(-periods)
    current = df['Close']
    change_pct = (future - current) / current * 100
    
    target = pd.Series(np.nan, index=df.index)
    target[change_pct > 0.05] = 1   # BUY = Class 1
    target[change_pct < -0.05] = 0  # SELL = Class 0
    
    return target

# ============================================================================
# MODEL - CORRECTED
# ============================================================================

class OptimizedModel:
    def __init__(self, symbol: str, config: Config, logger: logging.Logger):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        
        self.xgb_model = None
        self.cat_model = None
        self.meta_model = None
        self.scaler = RobustScaler()
        self.selected_features = []
        
        self.river_model = None
        self.river_samples = 0
        
        self.is_trained = False
        self.version = None
    
    def _init_river(self):
        self.river_model = compose.Pipeline(
            preprocessing.StandardScaler(),
            tree.HoeffdingAdaptiveTreeClassifier(
                grace_period=200, delta=1e-4, leaf_prediction='mc'
            )
        )
    
    def train(self, df: pd.DataFrame) -> bool:
        try:
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                return False
            
            self.logger.info(f"🎓 {self.symbol}: {len(df)} عينة...")
            
            features = calculate_features(df)
            target = create_target(df, self.config.FORECAST_PERIODS)
            
            # ✅ إحصائيات
            buy_count = target.sum()
            total = len(target.dropna())
            self.logger.info(f"📊 Target: BUY={buy_count:.0f}, SELL={total-buy_count:.0f}")
            
            valid = ~(features.isna().any(axis=1) | target.isna())
            X = features[valid]
            y = target[valid]
            
            if len(X) < 200:
                return False
            
            mi = mutual_info_classif(X, y, random_state=42)
            scores = sorted(zip(X.columns, mi), key=lambda x: x[1], reverse=True)
            self.selected_features = [s[0] for s in scores[:self.config.MAX_FEATURES]]
            X = X[self.selected_features]
            
            self._init_river()
            for i in range(len(X)):
                row = X.iloc[i].to_dict()
                try:
                    self.river_model.learn_one(row, int(y.iloc[i]))
                    self.river_samples += 1
                except:
                    pass
            
            self.logger.info(f"🌊 {self.symbol}: River={self.river_samples}")
            
            X_s = self.scaler.fit_transform(X)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X_s[:split_idx], X_s[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            self.xgb_model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                random_state=42, n_jobs=1, verbosity=0, tree_method='hist'
            )
            self.xgb_model.fit(X_train, y_train)
            
            self.cat_model = CatBoostClassifier(
                iterations=100, depth=4, learning_rate=0.05,
                random_seed=42, verbose=False, thread_count=1,
                allow_writing_files=False
            )
            self.cat_model.fit(X_train, y_train)
            
            xgb_pred = self.xgb_model.predict_proba(X_val)[:, 1]
            cat_pred = self.cat_model.predict_proba(X_val)[:, 1]
            
            meta_X = np.column_stack([xgb_pred, cat_pred])
            self.meta_model = LogisticRegression()
            self.meta_model.fit(meta_X, y_val)
            
            self.is_trained = True
            self.version = datetime.now().strftime('v%Y%m%d_%H%M%S')
            
            acc = accuracy_score(y_val, self.meta_model.predict(meta_X))
            self.logger.info(f"✅ {self.symbol}: دقة={acc:.1%}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.symbol}: {e}")
            return False
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.60) -> Tuple[str, float, float, float]:
        """
        ✅ يرجع: (الاتجاه, الثقة, proba_buy, proba_sell)
        """
        if not self.is_trained:
            return "NEUTRAL", 0.0, 0.5, 0.5
        
        try:
            features = calculate_features(df).iloc[[-1]]
            available = [f for f in self.selected_features if f in features.columns]
            
            if len(available) < 5:
                return "NEUTRAL", 0.0, 0.5, 0.5
            
            X = features[available].fillna(0)
            X_s = self.scaler.transform(X)
            
            # ✅ احتمالية BUY = predict_proba[1]
            xgb_proba_buy = float(self.xgb_model.predict_proba(X_s)[0, 1])
            cat_proba_buy = float(self.cat_model.predict_proba(X_s)[0, 1])
            
            meta_proba_buy = float(self.meta_model.predict_proba(
                np.array([[xgb_proba_buy, cat_proba_buy]]))[0, 1])
            
            X_dict = features[available].fillna(0).iloc[0].to_dict()
            try:
                river_proba = self.river_model.predict_proba_one(X_dict)
                river_proba_buy = river_proba.get(True, 0.5) if river_proba else 0.5
            except:
                river_proba_buy = 0.5
            
            # ✅ احتمالية BUY النهائية
            proba_buy = meta_proba_buy * 0.7 + river_proba_buy * 0.3
            proba_sell = 1 - proba_buy
            
            # ✅ قرار صريح
            if proba_buy > threshold:
                return "BUY", proba_buy, proba_buy, proba_sell
            elif proba_sell > threshold:
                return "SELL", proba_sell, proba_buy, proba_sell
            
            return "NEUTRAL", max(proba_buy, proba_sell), proba_buy, proba_sell
            
        except:
            return "NEUTRAL", 0.0, 0.5, 0.5
    
    def online_learn(self, df: pd.DataFrame, result: int):
        try:
            features = calculate_features(df).iloc[[-1]]
            available = [f for f in self.selected_features if f in features.columns]
            if len(available) >= 5:
                X_dict = features[available].fillna(0).iloc[0].to_dict()
                self.river_model.learn_one(X_dict, result)
                self.river_samples += 1
        except:
            pass
    
    def save(self):
        path = os.path.join(self.config.MODELS_DIR, self.symbol)
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'xgb': self.xgb_model, 'cat': self.cat_model,
            'meta': self.meta_model, 'scaler': self.scaler,
            'features': self.selected_features, 'version': self.version,
            'river': self.river_model, 'river_samples': self.river_samples
        }, os.path.join(path, 'optimized_model.pkl'))
    
    def load(self) -> bool:
        path = os.path.join(self.config.MODELS_DIR, self.symbol, 'optimized_model.pkl')
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.xgb_model = data['xgb']
        self.cat_model = data['cat']
        self.meta_model = data['meta']
        self.scaler = data['scaler']
        self.selected_features = data['features']
        self.version = data['version']
        self.river_model = data.get('river')
        self.river_samples = data.get('river_samples', 0)
        self.is_trained = True
        return True

# ============================================================================
# MAIN BOT
# ============================================================================

class FalconProBot:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.db = Database(config.DB_PATH)
        self.models = {}
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
        self.tg = TelegramManager(config.TELEGRAM_TOKEN)
        self.tb = self.tg.get_bot()
        self._setup_commands()
        
        for symbol in config.SYMBOLS:
            model = OptimizedModel(symbol, config, self.logger)
            if model.load():
                self.logger.info(f"📂 {symbol}")
            else:
                self.logger.info(f"🆕 {symbol}")
            self.models[symbol] = model
        
        self.running = False
        self.last_retrain = None
    
    def _setup_commands(self):
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            trained = sum(1 for m in self.models.values() if m.is_trained)
            stats = self.db.stats()
            text = f"🦅 Falcon Pro\n✅ نماذج: {trained}/{len(self.models)}\n📊 صفقات: {stats['total']}\n📈 نجاح: {stats['win_rate']:.1%}"
            try:
                self.tb.reply_to(msg, text)
            except:
                pass
    
    def fetch_data(self, symbol: str, interval: str = '5m', period: str = '3d') -> Optional[pd.DataFrame]:
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
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        try:
            model = self.models.get(symbol)
            if not model or not model.is_trained:
                return None
            
            if self.db.has_active(symbol):
                return None
            
            if self.db.was_recent(symbol, self.config.SIGNAL_COOLDOWN_MINUTES):
                return None
            
            df_5m = self.fetch_data(symbol, '5m', '3d')
            df_15m = self.fetch_data(symbol, '15m', '5d')
            
            if df_5m is None or df_15m is None:
                return None
            
            dir_5m, conf_5m, pb_5m, ps_5m = model.predict(df_5m, self.config.CONFIDENCE_THRESHOLD)
            dir_15m, conf_15m, pb_15m, ps_15m = model.predict(df_15m, self.config.CONFIDENCE_THRESHOLD)
            
            self.logger.info(f"{symbol}: M5={dir_5m}(B:{pb_5m:.1%} S:{ps_5m:.1%}) "
                           f"M15={dir_15m}(B:{pb_15m:.1%} S:{ps_15m:.1%})")
            
            if dir_5m != dir_15m or dir_5m == "NEUTRAL":
                return None
            
            confidence = (conf_5m + conf_15m) / 2
            
            if confidence < self.config.CONFIDENCE_THRESHOLD:
                return None
            
            entry_price = float(df_5m['Close'].iloc[-1])
            atr = float(df_5m['High'].iloc[-1] - df_5m['Low'].iloc[-1])
            duration = 5 if atr > 0.0008 else (7 if atr > 0.0004 else 10)
            
            return {
                'symbol': symbol,
                'direction': dir_5m,
                'entry_price': entry_price,
                'confidence': confidence,
                'proba_buy': (pb_5m + pb_15m) / 2,
                'proba_sell': (ps_5m + ps_15m) / 2,
                'trade_duration': duration,
                'expiry_time': (datetime.now() + timedelta(minutes=duration)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except:
            return None
    
    def send_signal(self, signal: Dict):
        try:
            emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
            direction = "شراء" if signal['direction'] == 'BUY' else "بيع"
            
            msg = (f"{emoji} {signal['symbol']} - {direction}\n"
                   f"السعر: {signal['entry_price']:.5f}\n"
                   f"المدة: {signal['trade_duration']} د\n"
                   f"الثقة: {signal['confidence']:.1%}")
            
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, msg)
            self.logger.info(f"✅ {signal['symbol']} {signal['direction']}")
        except:
            pass
    
    def check_trades(self):
        for trade in self.db.get_expired():
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
                
                self.db.update(trade['id'], current, result, pnl)
                
                model = self.models.get(trade['symbol'])
                if model and df is not None:
                    if direction == 'BUY':
                        learn = 1 if result == 'WIN' else 0
                    else:
                        learn = 0 if result == 'WIN' else 1
                    model.online_learn(df, learn)
                
            except:
                pass
    
    def scan_all_symbols(self):
        signals = 0
        for symbol in self.config.SYMBOLS:
            try:
                signal = self.analyze_symbol(symbol)
                if signal and self.db.save(signal):
                    self.send_signal(signal)
                    signals += 1
                time.sleep(0.5)
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
                    time.sleep(2)
                
                if df is not None:
                    model = OptimizedModel(symbol, self.config, self.logger)
                    if model.train(df):
                        model.save()
                        self.models[symbol] = model
                
                time.sleep(2)
                gc.collect()
            except:
                pass
        
        self.last_retrain = datetime.now()
    
    def run(self):
        self.running = True
        self.logger.info("🦅 Falcon Pro - Fixed Logic")
        
        self.tg.start_polling()
        time.sleep(1)
        
        if not any(m.is_trained for m in self.models.values()):
            self.train_all_models()
        
        self.last_retrain = datetime.now()
        
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID,
                f"🦅 Falcon Pro\n✅ {trained}/{len(self.config.SYMBOLS)}\n⚡️ يعمل...")
        except:
            pass
        
        while self.running:
            try:
                self.check_trades()
                self.scan_all_symbols()
                
                if (datetime.now() - self.last_retrain).total_seconds() > self.config.RETRAINING_INTERVAL_SECONDS:
                    self.train_all_models()
                
                time.sleep(self.config.SCAN_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Loop: {e}")
                time.sleep(10)
        
        self.executor.shutdown(wait=True)

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
            logging.error(f"Fatal: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
