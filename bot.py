#!/usr/bin/env python3
"""
Falcon AI v9.0 - Self-Evolving Trading Strategy
=================================================
The AI discovers its own winning strategy.
No hardcoded rules. Pure reinforcement learning.
12 Forex pairs. Automatic training.
"""

import os
import sys
import time
import logging
import sqlite3
import hashlib
import threading
import shutil
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score
import xgboost as xgb

import telebot
import joblib

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    TELEGRAM_TOKEN: str = os.environ.get('TELEGRAM_TOKEN', '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '7553333305')
    
    SCAN_INTERVAL: int = 60
    
    # ✅ 12 زوج فوركس
    SYMBOLS: List[str] = field(default_factory=lambda: [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
        'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X',
        'GBPJPY=X', 'EURCHF=X', 'USDCHF=X', 'AUDJPY=X'
    ])
    
    TRAINING_PERIOD: str = '2mo'
    MIN_TRAINING_SAMPLES: int = 500
    
    DB_PATH: str = 'falcon_v9.db'
    MODELS_DIR: str = 'models_v9'
    STRATEGY_FILE: str = 'evolved_strategy.json'
    
    LEARNING_RATE: float = 0.1
    MEMORY_SIZE: int = 100
    MIN_WIN_RATE: float = 0.45
    SIGNAL_COOLDOWN_MINUTES: int = 5

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FalconV9')

# ============================================================================
# SELF-EVOLVING STRATEGY
# ============================================================================

class EvolvingStrategy:
    def __init__(self, config: Config):
        self.config = config
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        
        self.confidence_threshold = 0.52  # ✅ أقل عشان إشارات أكثر
        self.base_duration = 7
        self.symbol_weights = {s: 1.0 for s in config.SYMBOLS}
        self.best_hours = list(range(24))
        self.avoid_hours = []
        
        self.total_trades = 0
        self.total_wins = 0
        self.current_win_rate = 0.5
        self.evolution_generation = 1
        
        self.load()
    
    def add_result(self, trade_data: Dict):
        self.memory.append({
            'symbol': trade_data['symbol'],
            'direction': trade_data['direction'],
            'hour': trade_data.get('hour', datetime.now().hour),
            'confidence': trade_data.get('confidence', 0),
            'duration': trade_data.get('trade_duration', 7),
            'pnl': trade_data.get('pnl_percent', 0),
            'result': trade_data.get('result', 'LOSS'),
        })
        
        self.total_trades += 1
        if trade_data.get('result') == 'WIN':
            self.total_wins += 1
        
        self.current_win_rate = self.total_wins / max(self.total_trades, 1)
        
        if self.total_trades >= 10 and self.total_trades % 10 == 0:
            self.evolve()
    
    def evolve(self):
        logger.info(f"🧬 تطور {self.evolution_generation}...")
        
        recent = list(self.memory)[-30:]
        wins = [t for t in recent if t['result'] == 'WIN']
        losses = [t for t in recent if t['result'] == 'LOSS']
        
        if wins and losses:
            avg_win_conf = np.mean([t['confidence'] for t in wins])
            avg_loss_conf = np.mean([t['confidence'] for t in losses])
            
            if avg_win_conf > avg_loss_conf + 0.05:
                self.confidence_threshold = min(0.70, self.confidence_threshold + 0.02)
            elif avg_loss_conf > avg_win_conf + 0.05:
                self.confidence_threshold = max(0.48, self.confidence_threshold - 0.02)
        
        if wins:
            self.base_duration = int(np.mean([t['duration'] for t in wins]))
        
        hour_perf = {}
        for t in recent:
            h = t['hour']
            if h not in hour_perf:
                hour_perf[h] = {'w': 0, 't': 0}
            hour_perf[h]['t'] += 1
            if t['result'] == 'WIN':
                hour_perf[h]['w'] += 1
        
        sorted_h = sorted(hour_perf.items(), key=lambda x: x[1]['w']/max(x[1]['t'],1), reverse=True)
        self.best_hours = [h for h, _ in sorted_h[:10]]
        
        self.evolution_generation += 1
        self.save()
        
        logger.info(f"  🎯 عتبة={self.confidence_threshold:.0%} | ⏱️ مدة={self.base_duration}د")
    
    def should_trade_now(self, symbol: str) -> Tuple[bool, str]:
        now = datetime.utcnow()
        if now.weekday() >= 5:
            return False, "ويكند"
        if now.weekday() == 4 and now.hour >= 20:
            return False, "إغلاق جمعة"
        return True, "مسموح"
    
    def get_dynamic_threshold(self, symbol: str) -> float:
        return self.confidence_threshold
    
    def get_dynamic_duration(self, symbol: str) -> int:
        return self.base_duration
    
    def save(self):
        with open(self.config.STRATEGY_FILE, 'w') as f:
            json.dump({
                'confidence_threshold': self.confidence_threshold,
                'base_duration': self.base_duration,
                'symbol_weights': self.symbol_weights,
                'best_hours': self.best_hours,
                'total_trades': self.total_trades,
                'total_wins': self.total_wins,
                'evolution_generation': self.evolution_generation
            }, f, indent=2)
    
    def load(self):
        if os.path.exists(self.config.STRATEGY_FILE):
            with open(self.config.STRATEGY_FILE, 'r') as f:
                data = json.load(f)
                self.confidence_threshold = data.get('confidence_threshold', 0.52)
                self.base_duration = data.get('base_duration', 7)
                self.total_trades = data.get('total_trades', 0)
                self.total_wins = data.get('total_wins', 0)
                self.evolution_generation = data.get('evolution_generation', 1)

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
                    exit_price REAL, confidence REAL,
                    proba_buy REAL, proba_sell REAL,
                    trade_duration INTEGER, trade_hour INTEGER,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, exit_time DATETIME,
                    result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, pnl_pips REAL,
                    strategy_generation INTEGER,
                    signal_hash TEXT UNIQUE
                )
            ''')
            conn.commit()
    
    def save_signal(self, data: Dict) -> Optional[int]:
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO signals 
                    (symbol, direction, entry_price, confidence, proba_buy, proba_sell,
                     trade_duration, trade_hour, expiry_time, strategy_generation, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data.get('proba_buy', 0), data.get('proba_sell', 0),
                      data['trade_duration'], data.get('hour', datetime.now().hour),
                      data['expiry_time'], data.get('strategy_gen', 1), h))
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
    
    def has_active_signal(self, symbol: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' 
                AND expiry_time > datetime('now', 'localtime')
            ''', (symbol,)).fetchone()[0]
            return c > 0
    
    def was_recent(self, symbol: str, minutes: int = 5) -> bool:
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?
            ''', (symbol, cutoff)).fetchone()[0]
            return c > 0
    
    def get_expired_trades(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM signals WHERE result='PENDING' 
                AND expiry_time <= datetime('now', 'localtime')
            ''').fetchall()
            return [dict(r) for r in rows]

# ============================================================================
# FEATURES
# ============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l = df['Close'], df['High'], df['Low']
    
    for p in [1, 3, 5, 10]:
        f[f'ret_{p}'] = c.pct_change(p)
    for p in [5, 10, 20, 50]:
        f[f'sma_{p}'] = c.rolling(p).mean()
        f[f'dist_{p}'] = (c - f[f'sma_{p}']) / (f[f'sma_{p}'] + 1e-8)
    
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    f['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    f['macd'] = ema12 - ema26
    f['macd_s'] = f['macd'].ewm(span=9).mean()
    f['macd_h'] = f['macd'] - f['macd_s']
    
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f['bb'] = (c - sma20) / (2 * std20 + 1e-8)
    
    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    f['atr'] = tr.ewm(span=14).mean()
    f['atr_pct'] = f['atr'] / (c + 1e-8)
    
    atr14 = tr.ewm(span=14).mean()
    pdm = h.diff().clip(lower=0)
    ndm = (-l.diff()).clip(lower=0)
    pdi = 100 * (pdm.ewm(span=14).mean()) / (atr14 + 1e-8)
    ndi = 100 * (ndm.ewm(span=14).mean()) / (atr14 + 1e-8)
    dx = 100 * abs(pdi - ndi) / (pdi + ndi + 1e-8)
    f['adx'] = dx.ewm(span=14).mean()
    
    l14 = l.rolling(14).min()
    h14 = h.rolling(14).max()
    f['stoch'] = 100 * (c - l14) / (h14 - l14 + 1e-8)
    
    for p in [5, 10]:
        f[f'mom_{p}'] = c - c.shift(p)
    
    return f.fillna(0)

# ============================================================================
# TARGET
# ============================================================================

def create_target(df: pd.DataFrame, periods: int = 5) -> pd.Series:
    future = df['Close'].shift(-periods)
    current = df['Close']
    change = (future - current) / current * 100
    target = pd.Series(np.nan, index=df.index)
    target[change > 0.03] = 1
    target[change < -0.03] = 0
    return target

# ============================================================================
# MODEL
# ============================================================================

class TradingModel:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = None
        self.scaler = RobustScaler()
        self.features = []
        self.is_trained = False
    
    def train(self, df: pd.DataFrame) -> bool:
        try:
            X = calculate_features(df)
            y = create_target(df)
            valid = ~(X.isna().any(axis=1) | y.isna())
            X, y = X[valid], y[valid]
            
            if len(X) < 200:
                return False
            
            mi = mutual_info_classif(X, y, random_state=42)
            scores = sorted(zip(X.columns, mi), key=lambda x: x[1], reverse=True)
            self.features = [s[0] for s in scores[:18]]
            X = X[self.features]
            
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            X_train_s = self.scaler.fit_transform(X_train)
            
            self.model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.03,
                random_state=42, n_jobs=1, verbosity=0, tree_method='hist'
            )
            self.model.fit(X_train_s, y_train)
            
            X_val_s = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_s)
            acc = accuracy_score(y_val, val_pred)
            
            logger.info(f"✅ {self.symbol}: دقة={acc:.1%}")
            self.is_trained = True
            return True
        except Exception as e:
            logger.error(f"❌ {self.symbol}: {e}")
            return False
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.52) -> Tuple[str, float, float, float]:
        if not self.is_trained:
            return "NEUTRAL", 0.0, 0.5, 0.5
        
        try:
            X = calculate_features(df).iloc[[-1]]
            available = [f for f in self.features if f in X.columns]
            if len(available) < 8:
                return "NEUTRAL", 0.0, 0.5, 0.5
            
            X = X[available].fillna(0)
            X_s = self.scaler.transform(X)
            
            probas = self.model.predict_proba(X_s)[0]
            proba_sell = float(probas[0])
            proba_buy = float(probas[1])
            
            if proba_buy > threshold:
                return "BUY", proba_buy, proba_buy, proba_sell
            elif proba_sell > threshold:
                return "SELL", proba_sell, proba_buy, proba_sell
            else:
                return "NEUTRAL", max(proba_buy, proba_sell), proba_buy, proba_sell
        except:
            return "NEUTRAL", 0.0, 0.5, 0.5
    
    def save(self):
        os.makedirs(f"models_v9/{self.symbol}", exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'features': self.features},
                   f"models_v9/{self.symbol}/model.pkl")
    
    def load(self) -> bool:
        path = f"models_v9/{self.symbol}/model.pkl"
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.features = data['features']
        self.is_trained = True
        return True

# ============================================================================
# MAIN BOT
# ============================================================================

class FalconV9:
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config.DB_PATH)
        self.strategy = EvolvingStrategy(config)
        self.models = {}
        
        self.tb = telebot.TeleBot(config.TELEGRAM_TOKEN)
        self._setup_bot()
        
        # ✅ حذف النماذج القديمة وتدريب جديد
        if os.path.exists(config.MODELS_DIR):
            shutil.rmtree(config.MODELS_DIR)
        
        self._train_all_models()
    
    def _train_all_models(self):
        """✅ تدريب كل الأزواج"""
        logger.info(f"🎓 بدء تدريب {len(self.config.SYMBOLS)} زوج...")
        
        for symbol in self.config.SYMBOLS:
            try:
                logger.info(f"📥 {symbol}: جلب البيانات...")
                df = self.fetch_data(symbol, '15m', self.config.TRAINING_PERIOD)
                
                if df is not None and len(df) >= self.config.MIN_TRAINING_SAMPLES:
                    model = TradingModel(symbol)
                    if model.train(df):
                        model.save()
                        self.models[symbol] = model
                        logger.info(f"✅ {symbol}: تم")
                else:
                    logger.warning(f"⚠️ {symbol}: بيانات غير كافية")
                
                time.sleep(3)  # ✅ استراحة بين كل زوج
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
        
        trained = sum(1 for m in self.models.values() if m.is_trained)
        logger.info(f"🎓 اكتمل التدريب: {trained}/{len(self.config.SYMBOLS)}")
    
    def _setup_bot(self):
        try:
            requests.get(f'https://api.telegram.org/bot{self.config.TELEGRAM_TOKEN}/deleteWebhook', timeout=3)
        except:
            pass
        
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            trained = sum(1 for m in self.models.values() if m.is_trained)
            text = (f"🦅 **Falcon V9.0**\n"
                   f"✅ نماذج: {trained}/{len(self.config.SYMBOLS)}\n"
                   f"🧬 الجيل: {self.strategy.evolution_generation}\n"
                   f"📊 صفقات: {self.strategy.total_trades}\n"
                   f"📈 نجاح: {self.strategy.current_win_rate:.1%}\n"
                   f"🎯 العتبة: {self.strategy.confidence_threshold:.0%}\n"
                   f"⏱️ المدة: {self.strategy.base_duration} د")
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['train'])
        def train_cmd(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            self.tb.reply_to(msg, "🎓 جاري إعادة التدريب...")
            self._train_all_models()
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.reply_to(msg, f"✅ جاهز: {trained}/{len(self.config.SYMBOLS)}")
        
        @self.tb.message_handler(func=lambda msg: True)
        def analyze_any(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            
            symbol = msg.text.strip().upper()
            if '=X' not in symbol and '/' not in symbol and '-' not in symbol:
                symbol = f"{symbol}=X"
            
            self.tb.reply_to(msg, f"🔍 تحليل {symbol}...")
            
            df_5m = self.fetch_data(symbol, '5m', '3d')
            df_15m = self.fetch_data(symbol, '15m', '5d')
            
            if df_5m is None or df_15m is None:
                self.tb.reply_to(msg, f"❌ لا بيانات لـ {symbol}")
                return
            
            model = self.models.get(symbol) or list(self.models.values())[0] if self.models else None
            
            if model is None or not model.is_trained:
                self.tb.reply_to(msg, "❌ النموذج غير جاهز")
                return
            
            dir_5m, conf_5m, pb_5m, ps_5m = model.predict(df_5m, 0.50)
            dir_15m, conf_15m, pb_15m, ps_15m = model.predict(df_15m, 0.50)
            price = float(df_5m['Close'].iloc[-1])
            
            text = (f"📊 **{symbol}**\n\n"
                   f"💰 {price:.5f}\n\n"
                   f"M5: {dir_5m} (B:{pb_5m:.0%} S:{ps_5m:.0%})\n"
                   f"M15: {dir_15m} (B:{pb_15m:.0%} S:{ps_15m:.0%})")
            
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def fetch_data(self, symbol: str, interval: str = '5m', period: str = '5d') -> Optional[pd.DataFrame]:
        for _ in range(3):
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                if not df.empty:
                    df.columns = [c.capitalize() for c in df.columns]
                    return df
            except:
                time.sleep(3)
        return None
    
    def analyze(self, symbol: str) -> Optional[Dict]:
        should_trade, reason = self.strategy.should_trade_now(symbol)
        if not should_trade:
            return None
        
        model = self.models.get(symbol)
        if not model or not model.is_trained:
            return None
        
        if self.db.has_active_signal(symbol):
            return None
        
        if self.db.was_recent(symbol, self.config.SIGNAL_COOLDOWN_MINUTES):
            return None
        
        df_5m = self.fetch_data(symbol, '5m', '3d')
        df_15m = self.fetch_data(symbol, '15m', '5d')
        
        if df_5m is None or df_15m is None:
            return None
        
        threshold = self.strategy.get_dynamic_threshold(symbol)
        
        dir_5m, conf_5m, pb_5m, ps_5m = model.predict(df_5m, threshold)
        dir_15m, conf_15m, pb_15m, ps_15m = model.predict(df_15m, threshold)
        
        if dir_5m != dir_15m or dir_5m == "NEUTRAL":
            return None
        
        confidence = (conf_5m + conf_15m) / 2
        
        if confidence < threshold:
            return None
        
        duration = self.strategy.get_dynamic_duration(symbol)
        entry = float(df_5m['Close'].iloc[-1])
        
        return {
            'symbol': symbol,
            'direction': dir_5m,
            'entry_price': entry,
            'confidence': confidence,
            'proba_buy': (pb_5m + pb_15m) / 2,
            'proba_sell': (ps_5m + ps_15m) / 2,
            'trade_duration': duration,
            'hour': datetime.now().hour,
            'strategy_gen': self.strategy.evolution_generation,
            'expiry_time': (datetime.now() + timedelta(minutes=duration)).strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def send_signal(self, signal: Dict):
        emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
        
        msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
               f"💰 {signal['entry_price']:.5f}\n"
               f"⏳ {signal['trade_duration']} د\n"
               f"💪 {signal['confidence']:.1%}\n\n"
               f"🤖 Falcon V9")
        
        try:
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
            logger.info(f"✅ {signal['symbol']} {signal['direction']}")
        except:
            pass
    
    def check_trades(self):
        for trade in self.db.get_expired_trades():
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
                
                trade['pnl_percent'] = pnl
                trade['result'] = result
                self.strategy.add_result(trade)
                
            except:
                pass
    
    def scan(self):
        for symbol in self.config.SYMBOLS:
            try:
                signal = self.analyze(symbol)
                if signal and self.db.save_signal(signal):
                    self.send_signal(signal)
                time.sleep(0.5)
            except:
                pass
    
    def run(self):
        logger.info(f"🦅 Falcon V9.0 - {len(self.config.SYMBOLS)} زوج")
        
        def poll():
            while True:
                try:
                    self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except:
                    time.sleep(5)
        threading.Thread(target=poll, daemon=True).start()
        time.sleep(1)
        
        trained = sum(1 for m in self.models.values() if m.is_trained)
        try:
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID,
                f"🦅 **Falcon V9.0**\n✅ {trained}/{len(self.config.SYMBOLS)}\n⚡️ يعمل...",
                parse_mode='Markdown')
        except:
            pass
        
        while True:
            try:
                self.check_trades()
                self.scan()
                time.sleep(self.config.SCAN_INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"خطأ: {e}")
                time.sleep(10)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    config = Config()
    bot = FalconV9(config)
    bot.run()
