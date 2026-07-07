#!/usr/bin/env python3
"""
Falcon AI Ultimate v2.3 - Forex Only
========================================
8 Forex pairs. XGBoost. Ready to run.
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['figure.dpi'] = 72

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
from catboost import CatBoostClassifier

import telebot
from telebot import types
import joblib
import shutil

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
    
    # ✅ فوركس فقط
    SYMBOLS: List[str] = field(default_factory=lambda: [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
        'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X'
    ])
    
    CONFIDENCE_THRESHOLD: float = 0.55
    RETRAINING_INTERVAL_HOURS: int = 24
    MIN_TRAINING_SAMPLES: int = 200
    TRAINING_PERIOD: str = '1mo'
    MAX_FEATURES: int = 30
    
    DB_PATH: str = 'falcon_trading.db'
    MODELS_DIR: str = 'models'
    
    CHART_CANDLES: int = 30
    CHART_DPI: int = 72
    
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 5
    MAX_WORKERS: int = 3
    SIGNAL_COOLDOWN_MINUTES: int = 5
    
    LOG_FILE: str = 'falcon_bot.log'

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    logging.basicConfig(
        level=logging.DEBUG,
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
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT,
                    direction TEXT,
                    entry_price REAL,
                    confidence REAL,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME,
                    result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL,
                    signal_hash TEXT UNIQUE
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
                    (symbol, direction, entry_price, confidence, expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data['expiry_time'], signal_hash))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except Exception as e:
            self.logger.error(f"Save signal error: {e}")
            return None
    
    def check_active_signal(self, symbol: str) -> bool:
        try:
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute('''
                    SELECT COUNT(*) FROM signals 
                    WHERE symbol = ? AND result = 'PENDING' 
                    AND expiry_time > datetime('now', 'localtime')
                ''', (symbol,)).fetchone()[0]
                return count > 0
        except:
            return True
    
    def check_recent_signal(self, symbol: str, minutes: int) -> bool:
        try:
            cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute('''
                    SELECT COUNT(*) FROM signals 
                    WHERE symbol = ? AND entry_time > ?
                ''', (symbol, cutoff)).fetchone()[0]
                return count > 0
        except:
            return True
    
    def update_result(self, signal_id: int, exit_price: float, result: str, pnl: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE signals SET exit_price = ?, result = ?, pnl_percent = ?
                WHERE id = ?
            ''', (exit_price, result, pnl, signal_id))
            conn.commit()
    
    def get_pending_trades(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM signals 
                WHERE result = 'PENDING' AND expiry_time <= datetime('now', 'localtime')
            ''').fetchall()
            return [dict(r) for r in rows]
    
    def get_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM signals WHERE result != 'PENDING'").fetchone()[0]
            wins = conn.execute("SELECT COUNT(*) FROM signals WHERE result = 'WIN'").fetchone()[0]
            return {
                'total': total,
                'wins': wins,
                'losses': total - wins,
                'win_rate': wins / total if total > 0 else 0
            }

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c = df['Close']
    h = df['High']
    l = df['Low']
    
    for p in [1, 3, 5, 10]:
        f[f'ret_{p}'] = c.pct_change(p)
    
    for p in [5, 10, 20, 50]:
        f[f'sma_{p}'] = c.rolling(p).mean()
        f[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
        f[f'price_sma_{p}'] = c / f[f'sma_{p}'] - 1
    
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    f['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
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
    f['atr_pct'] = f['atr'] / (c + 1e-8)
    
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    f['stoch_k'] = 100 * (c - low14) / (high14 - low14 + 1e-8)
    
    for p in [5, 10]:
        f[f'roc_{p}'] = (c - c.shift(p)) / (c.shift(p) + 1e-8) * 100
    
    f['volatility'] = c.pct_change().rolling(20).std()
    
    if 'Volume' in df.columns:
        v = df['Volume']
        f['vol_ratio'] = v / (v.rolling(20).mean() + 1e-8)
    
    return f.fillna(0)

def calculate_trend(df: pd.DataFrame) -> Tuple[str, float]:
    if len(df) < 30:
        return "NEUTRAL", 0
    
    c = df['Close']
    ema20 = c.ewm(span=20).mean().iloc[-1]
    ema50 = c.ewm(span=50).mean().iloc[-1] if len(df) >= 50 else ema20
    current = c.iloc[-1]
    
    sma20 = c.rolling(20).mean()
    strength = abs((sma20.iloc[-1] - sma20.iloc[-20]) / sma20.iloc[-20]) if len(sma20) >= 20 else 0
    
    if current > ema20 > ema50:
        return "UP", strength
    elif current < ema20 < ema50:
        return "DOWN", strength
    return "NEUTRAL", 0

# ============================================================================
# FAST MODEL
# ============================================================================

class FastModel:
    def __init__(self, symbol: str, config: Config, logger: logging.Logger):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        
        self.model = None
        self.scaler = RobustScaler()
        self.selected_features = []
        self.is_trained = False
        self.version = None
    
    def train(self, df: pd.DataFrame) -> bool:
        try:
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                self.logger.warning(f"{self.symbol}: Not enough data ({len(df)})")
                return False
            
            self.logger.info(f"Training {self.symbol} with {len(df)} samples...")
            
            features = calculate_features(df)
            target = (df['Close'].shift(-3) > df['Close']).astype(int)
            
            valid = ~(features.isna().any(axis=1) | target.isna())
            X = features[valid]
            y = target[valid]
            
            if len(X) < 100:
                return False
            
            mi = mutual_info_classif(X, y, random_state=42)
            scores = sorted(zip(X.columns, mi), key=lambda x: x[1], reverse=True)
            self.selected_features = [s[0] for s in scores[:25]]
            
            X = X[self.selected_features]
            
            split_idx = int(len(X) * 0.8)
            X_train = X[:split_idx]
            y_train = y[:split_idx]
            
            X_train_s = self.scaler.fit_transform(X_train)
            
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
                n_jobs=2,
                verbosity=0,
                tree_method='hist'
            )
            self.model.fit(X_train_s, y_train)
            
            self.is_trained = True
            self.version = datetime.now().strftime('v%Y%m%d_%H%M%S')
            
            self.logger.info(f"Trained: {self.symbol} | Features: {len(self.selected_features)}")
            return True
            
        except Exception as e:
            self.logger.error(f"Train {self.symbol}: {e}")
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
            self.logger.error(f"Predict {self.symbol}: {e}")
            return "NEUTRAL", 0.0
    
    def save(self):
        path = os.path.join(self.config.MODELS_DIR, self.symbol)
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.selected_features,
            'version': self.version
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
        self.is_trained = True
        return True

# ============================================================================
# MAIN BOT
# ============================================================================

class FalconBot:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.db = Database(config.DB_PATH, self.logger)
        self.models = {}
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
        self.tb = telebot.TeleBot(config.TELEGRAM_TOKEN)
        self._setup_commands()
        
        for symbol in config.SYMBOLS:
            model = FastModel(symbol, config, self.logger)
            if model.load():
                self.logger.info(f"Loaded: {symbol}")
            else:
                self.logger.info(f"New: {symbol}")
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
            text = f"🦅 Falcon AI Forex\n✅ Models: {trained}/{len(self.models)}\n📊 Signals: {stats['total']}\n📈 Win Rate: {stats['win_rate']:.1%}"
            self.tb.reply_to(msg, text)
        
        @self.tb.message_handler(commands=['stats'])
        def stats_cmd(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            s = self.db.get_stats()
            text = f"📊 Signals: {s['total']}\n✅ Wins: {s['wins']}\n❌ Losses: {s['losses']}\n📈 Rate: {s['win_rate']:.1%}"
            self.tb.reply_to(msg, text)
        
        @self.tb.message_handler(commands=['force'])
        def force_scan(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            self.tb.reply_to(msg, "🔍 Scanning...")
            self.scan_markets()
    
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
            
            dir_5m, conf_5m = model.predict(df_5m)
            dir_15m, conf_15m = model.predict(df_15m)
            
            self.logger.debug(f"{symbol}: M5={dir_5m}({conf_5m:.2f}), M15={dir_15m}({conf_15m:.2f})")
            
            if dir_5m != dir_15m or dir_5m == "NEUTRAL":
                return None
            
            trend, trend_strength = calculate_trend(df_15m)
            
            if dir_5m == "BUY" and trend == "DOWN":
                return None
            if dir_5m == "SELL" and trend == "UP":
                return None
            
            confidence = (conf_5m + conf_15m) / 2
            
            if confidence < self.config.CONFIDENCE_THRESHOLD:
                return None
            
            self.logger.info(f"SIGNAL: {symbol} {dir_5m} | Confidence: {confidence:.2f}")
            
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
            direction = "شراء" if signal['direction'] == 'BUY' else "بيع"
            
            msg = f"{emoji} **{signal['symbol']}** - {direction}\n\n💰 السعر: {signal['entry_price']:.5f}\n⏳ المدة: {self.config.TRADE_DURATION_MINUTES} دقائق\n💪 الثقة: {signal['confidence']:.1%}\n\n🤖 Falcon AI Forex"
            
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
            self.logger.info(f"SENT: {signal['symbol']} {signal['direction']}")
            
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
                
                if trade['direction'] == 'BUY':
                    pnl = (current - entry) / entry * 100
                    result = 'WIN' if current > entry else 'LOSS'
                else:
                    pnl = (entry - current) / entry * 100
                    result = 'WIN' if current < entry else 'LOSS'
                
                self.db.update_result(trade['id'], current, result, pnl)
                self.logger.info(f"Trade {trade['id']}: {result} ({pnl:.2f}%)")
                
            except Exception as e:
                self.logger.error(f"Check trade error: {e}")
    
    def scan_markets(self):
        self.logger.info("Scanning markets...")
        
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
                self.logger.warning(f"Timeout: {symbol}")
            except Exception as e:
                self.logger.error(f"Error: {symbol}: {e}")
        
        self.logger.info(f"Scan done. Signals: {signals_found}")
    
    def train_all_models(self):
        self.logger.info("Training models...")
        
        for symbol in self.config.SYMBOLS:
            try:
                df = self.fetch_data(symbol, '1h', self.config.TRAINING_PERIOD)
                if df is not None:
                    model = FastModel(symbol, self.config, self.logger)
                    if model.train(df):
                        model.save()
                        self.models[symbol] = model
            except Exception as e:
                self.logger.error(f"Train {symbol}: {e}")
        
        self.last_retrain = datetime.now()
        self.logger.info("Training complete!")
    
    def start_telegram(self):
        def poll():
            self.logger.info("Telegram polling started")
            while True:
                try:
                    self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except Exception as e:
                    self.logger.error(f"Polling: {e}")
                    time.sleep(10)
        
        threading.Thread(target=poll, daemon=True).start()
    
    def run(self):
        self.running = True
        
        self.logger.info("=" * 40)
        self.logger.info("Falcon AI Forex Starting...")
        self.logger.info("=" * 40)
        
        self.start_telegram()
        time.sleep(2)
        
        if not any(m.is_trained for m in self.models.values()):
            self.logger.info("Training models...")
            self.train_all_models()
        
        self.last_retrain = datetime.now()
        
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(
                self.config.TELEGRAM_CHAT_ID,
                f"🦅 Falcon AI Forex Started\n✅ Models: {trained}/{len(self.models)}\n⚡️ Scanning...",
                parse_mode='Markdown'
            )
        except:
            pass
        
        while self.running:
            try:
                self.check_trades()
                self.scan_markets()
                
                if (datetime.now() - self.last_retrain).total_seconds() > 86400:
                    self.train_all_models()
                
                self.logger.info(f"Sleeping {self.config.SCAN_INTERVAL_MINUTES}min...")
                time.sleep(self.config.SCAN_INTERVAL_MINUTES * 60)
                
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                self.logger.error(f"Loop error: {e}")
                time.sleep(30)
        
        self.executor.shutdown(wait=True)
        self.logger.info("Shutdown complete")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    config = Config()
    bot = FalconBot(config)
    bot.run()
