#!/usr/bin/env python3
"""
Falcon AI v9.1 - Alpha Vantage Data
=====================================
Free data source. No Yahoo Finance needed.
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
    
    # ✅ Alpha Vantage API Key (مجاني من alphavantage.co)
    ALPHA_VANTAGE_KEY: str = os.environ.get('5TFFWK21CUNA3P25', 'demo')
    
    SCAN_INTERVAL: int = 60
    
    SYMBOLS: List[str] = field(default_factory=lambda: [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
        'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY',
        'GBPJPY', 'EURCHF', 'USDCHF', 'AUDJPY'
    ])
    
    TRAINING_PERIOD: str = '2mo'
    MIN_TRAINING_SAMPLES: int = 500
    
    DB_PATH: str = 'falcon_v9.db'
    MODELS_DIR: str = 'models_v9'
    STRATEGY_FILE: str = 'evolved_strategy.json'
    
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
# DATA FETCHER - Alpha Vantage (مجاني)
# ============================================================================

class DataFetcher:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 30  # ثواني
    
    def fetch_forex(self, symbol: str, interval: str = '5min', outputsize: str = 'compact') -> Optional[pd.DataFrame]:
        """جلب بيانات الفوركس"""
        
        # ✅ رمز الفوركس لـ Alpha Vantage
        from_currency = symbol[:3]
        to_currency = symbol[3:]
        
        # ✅ كاش
        cache_key = f"{symbol}_{interval}_{outputsize}"
        if cache_key in self.cache:
            if time.time() - self.cache_time.get(cache_key, 0) < self.cache_duration:
                return self.cache[cache_key].copy()
        
        # ✅ تحويل الفاصل الزمني
        interval_map = {
            '1m': '1min', '5m': '5min', '15m': '15min',
            '30m': '30min', '1h': '60min'
        }
        av_interval = interval_map.get(interval, '5min')
        
        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': from_currency,
            'to_symbol': to_currency,
            'interval': av_interval,
            'outputsize': outputsize,
            'apikey': self.api_key
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            data = response.json()
            
            # ✅ استخراج البيانات
            time_series_key = f"Time Series FX ({av_interval})"
            
            if time_series_key not in data:
                logger.warning(f"⚠️ {symbol}: لا بيانات من Alpha Vantage")
                # ✅ جرب Yahoo Finance كبديل
                return self._fetch_yahoo_fallback(symbol, interval)
            
            time_series = data[time_series_key]
            
            records = []
            for timestamp, values in time_series.items():
                records.append({
                    'Date': timestamp,
                    'Open': float(values['1. open']),
                    'High': float(values['2. high']),
                    'Low': float(values['3. low']),
                    'Close': float(values['4. close']),
                    'Volume': 0
                })
            
            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            
            # ✅ حفظ في الكاش
            self.cache[cache_key] = df
            self.cache_time[cache_key] = time.time()
            
            logger.info(f"✅ {symbol}: {len(df)} صف من Alpha Vantage")
            return df
            
        except Exception as e:
            logger.error(f"❌ {symbol}: Alpha Vantage فشل - {e}")
            return self._fetch_yahoo_fallback(symbol, interval)
    
    def _fetch_yahoo_fallback(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """✅ بديل احتياطي: Yahoo Finance"""
        try:
            import yfinance as yf
            
            # تحويل الرمز
            yahoo_symbol = f"{symbol}=X"
            
            # تحويل الفاصل
            interval_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h'}
            yf_interval = interval_map.get(interval, '5m')
            
            df = yf.download(yahoo_symbol, period='5d', interval=yf_interval, progress=False)
            
            if not df.empty:
                df.columns = [c.capitalize() for c in df.columns]
                logger.info(f"✅ {symbol}: {len(df)} صف من Yahoo (بديل)")
                return df
            
        except:
            pass
        
        return None

# ============================================================================
# STRATEGY
# ============================================================================

class EvolvingStrategy:
    def __init__(self, config: Config):
        self.config = config
        self.memory = deque(maxlen=100)
        self.confidence_threshold = 0.52
        self.base_duration = 7
        self.total_trades = 0
        self.total_wins = 0
        self.current_win_rate = 0.5
        self.evolution_generation = 1
        self.load()
    
    def add_result(self, trade_data: Dict):
        self.memory.append(trade_data)
        self.total_trades += 1
        if trade_data.get('result') == 'WIN':
            self.total_wins += 1
        self.current_win_rate = self.total_wins / max(self.total_trades, 1)
        
        if self.total_trades >= 10 and self.total_trades % 10 == 0:
            recent = list(self.memory)[-20:]
            wins = [t for t in recent if t['result'] == 'WIN']
            if wins:
                self.confidence_threshold = np.clip(
                    self.confidence_threshold + (0.01 if len(wins) > len(recent)/2 else -0.01),
                    0.48, 0.70
                )
                self.base_duration = int(np.mean([t.get('duration', 7) for t in wins]))
            self.evolution_generation += 1
            self.save()
    
    def should_trade_now(self) -> Tuple[bool, str]:
        now = datetime.utcnow()
        if now.weekday() >= 5:
            return False, "ويكند"
        return True, "مسموح"
    
    def save(self):
        with open(self.config.STRATEGY_FILE, 'w') as f:
            json.dump({
                'confidence_threshold': self.confidence_threshold,
                'base_duration': self.base_duration,
                'total_trades': self.total_trades,
                'evolution_generation': self.evolution_generation
            }, f)
    
    def load(self):
        if os.path.exists(self.config.STRATEGY_FILE):
            with open(self.config.STRATEGY_FILE, 'r') as f:
                data = json.load(f)
                self.confidence_threshold = data.get('confidence_threshold', 0.52)
                self.base_duration = data.get('base_duration', 7)
                self.total_trades = data.get('total_trades', 0)
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
                    trade_duration INTEGER,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, exit_time DATETIME,
                    result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, pnl_pips REAL,
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
                    (symbol, direction, entry_price, confidence, trade_duration,
                     expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data['trade_duration'],
                      data['expiry_time'], h))
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
    f['macd_s'] = f['macd'].ewm(span=9).mean()
    
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
            self.features = [s[0] for s in scores[:15]]
            X = X[self.features]
            
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            X_train_s = self.scaler.fit_transform(X_train)
            
            self.model = xgb.XGBClassifier(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                random_state=42, n_jobs=1, verbosity=0, tree_method='hist'
            )
            self.model.fit(X_train_s, y_train)
            
            self.is_trained = True
            logger.info(f"✅ {self.symbol}: مدرب")
            return True
        except:
            return False
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.52) -> Tuple[str, float, float, float]:
        if not self.is_trained:
            return "NEUTRAL", 0.0, 0.5, 0.5
        
        try:
            X = calculate_features(df).iloc[[-1]]
            available = [f for f in self.features if f in X.columns]
            if len(available) < 5:
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
        self.fetcher = DataFetcher(config.ALPHA_VANTAGE_KEY)
        self.db = Database(config.DB_PATH)
        self.strategy = EvolvingStrategy(config)
        self.models = {}
        
        self.tb = telebot.TeleBot(config.TELEGRAM_TOKEN)
        self._setup_bot()
        
        if os.path.exists(config.MODELS_DIR):
            shutil.rmtree(config.MODELS_DIR)
        
        self._train_all_models()
    
    def _train_all_models(self):
        logger.info(f"🎓 تدريب {len(self.config.SYMBOLS)} زوج...")
        
        for symbol in self.config.SYMBOLS:
            try:
                df = self.fetcher.fetch_forex(symbol, '15min', 'compact')
                
                if df is not None and len(df) >= 200:
                    model = TradingModel(symbol)
                    if model.train(df):
                        model.save()
                        self.models[symbol] = model
                        logger.info(f"✅ {symbol}")
                
                time.sleep(15)  # ✅ Alpha Vantage: 5 طلبات/دقيقة
            except Exception as e:
                logger.error(f"❌ {symbol}: {e}")
        
        trained = sum(1 for m in self.models.values() if m.is_trained)
        logger.info(f"🎓 جاهز: {trained}/{len(self.config.SYMBOLS)}")
    
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
            text = (f"🦅 **Falcon V9.1**\n"
                   f"✅ نماذج: {trained}/{len(self.config.SYMBOLS)}\n"
                   f"📊 صفقات: {self.strategy.total_trades}\n"
                   f"📈 نجاح: {self.strategy.current_win_rate:.1%}\n"
                   f"🎯 العتبة: {self.strategy.confidence_threshold:.0%}\n"
                   f"📡 Alpha Vantage")
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['train'])
        def train_cmd(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            self.tb.reply_to(msg, "🎓 جاري التدريب...")
            self._train_all_models()
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.reply_to(msg, f"✅ جاهز: {trained}/{len(self.config.SYMBOLS)}")
        
        @self.tb.message_handler(func=lambda msg: True)
        def analyze_any(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            
            symbol = msg.text.strip().upper()
            self.tb.reply_to(msg, f"🔍 تحليل {symbol}...")
            
            df_5m = self.fetcher.fetch_forex(symbol, '5min')
            
            if df_5m is None or df_5m.empty:
                self.tb.reply_to(msg, f"❌ لا بيانات لـ {symbol}")
                return
            
            model = self.models.get(symbol) or list(self.models.values())[0] if self.models else None
            
            if model is None or not model.is_trained:
                self.tb.reply_to(msg, "❌ النموذج غير جاهز")
                return
            
            dir_5m, conf_5m, pb_5m, ps_5m = model.predict(df_5m, 0.50)
            price = float(df_5m['Close'].iloc[-1])
            
            text = (f"📊 **{symbol}**\n\n"
                   f"💰 {price:.5f}\n\n"
                   f"M5: {dir_5m} (B:{pb_5m:.0%} S:{ps_5m:.0%})")
            
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def analyze(self, symbol: str) -> Optional[Dict]:
        if not self.strategy.should_trade_now()[0]:
            return None
        
        model = self.models.get(symbol)
        if not model or not model.is_trained:
            return None
        
        if self.db.has_active_signal(symbol):
            return None
        
        if self.db.was_recent(symbol, self.config.SIGNAL_COOLDOWN_MINUTES):
            return None
        
        df_5m = self.fetcher.fetch_forex(symbol, '5min')
        df_15m = self.fetcher.fetch_forex(symbol, '15min')
        
        if df_5m is None or df_15m is None:
            return None
        
        threshold = self.strategy.confidence_threshold
        
        dir_5m, conf_5m, pb_5m, ps_5m = model.predict(df_5m, threshold)
        dir_15m, conf_15m, pb_15m, ps_15m = model.predict(df_15m, threshold)
        
        if dir_5m != dir_15m or dir_5m == "NEUTRAL":
            return None
        
        confidence = (conf_5m + conf_15m) / 2
        
        if confidence < threshold:
            return None
        
        entry = float(df_5m['Close'].iloc[-1])
        
        return {
            'symbol': symbol,
            'direction': dir_5m,
            'entry_price': entry,
            'confidence': confidence,
            'trade_duration': self.strategy.base_duration,
            'expiry_time': (datetime.now() + timedelta(minutes=self.strategy.base_duration)).strftime('%Y-%m-%d %H:%M:%S')
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
        except:
            pass
    
    def check_trades(self):
        for trade in self.db.get_expired_trades():
            try:
                df = self.fetcher.fetch_forex(trade['symbol'], '5min')
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
                self.strategy.add_result({'symbol': trade['symbol'], 'result': result, 
                                          'duration': trade.get('trade_duration', 7)})
            except:
                pass
    
    def scan(self):
        for symbol in self.config.SYMBOLS:
            try:
                signal = self.analyze(symbol)
                if signal and self.db.save_signal(signal):
                    self.send_signal(signal)
            except:
                pass
    
    def run(self):
        logger.info(f"🦅 Falcon V9.1 - Alpha Vantage - {len(self.config.SYMBOLS)} زوج")
        
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
                f"🦅 **Falcon V9.1**\n✅ {trained}/{len(self.config.SYMBOLS)}\n📡 Alpha Vantage\n⚡️ يعمل...",
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
            except:
                time.sleep(10)

if __name__ == "__main__":
    config = Config()
    bot = FalconV9(config)
    bot.run()
