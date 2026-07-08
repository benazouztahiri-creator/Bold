#!/usr/bin/env python3
"""
Falcon AI v11 - Adaptive Strategy Switcher
============================================
Step 1: Read market (Trend / Range / Breakout)
Step 2: Choose strategy (1 / 2 / 3)
Step 3: Execute
No auto-train on startup. Use /train command.
"""

import os
import sys
import time
import json
import logging
import sqlite3
import hashlib
import threading
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests

from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb

import telebot
import joblib

# ============================================================================
# CONFIG
# ============================================================================

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '7553333305')
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', '5TFFWK21CUNA3P25')

SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'EURJPY', 'GBPJPY']

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FalconV11')

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    def __init__(self, db_path: str = 'falcon_v11.db'):
        self.db_path = db_path
        with sqlite3.connect(db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    exit_price REAL, confidence REAL,
                    strategy TEXT, market_type TEXT,
                    duration INTEGER,
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
                    (symbol, direction, entry_price, confidence, strategy, market_type,
                     duration, expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data.get('strategy', ''), data.get('market_type', ''),
                      data['duration'], data['expiry_time'], h))
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
    
    def get_strategy_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            stats = {}
            for strategy in ['trend', 'range', 'breakout']:
                total = conn.execute('''
                    SELECT COUNT(*) FROM signals WHERE strategy=? AND result!='PENDING'
                ''', (strategy,)).fetchone()[0]
                wins = conn.execute('''
                    SELECT COUNT(*) FROM signals WHERE strategy=? AND result='WIN'
                ''', (strategy,)).fetchone()[0]
                stats[strategy] = {
                    'total': total,
                    'wins': wins,
                    'win_rate': wins/total if total > 0 else 0
                }
            return stats

# ============================================================================
# MARKET READER
# ============================================================================

class MarketReader:
    @staticmethod
    def read(df: pd.DataFrame) -> Dict:
        if len(df) < 50:
            return {'type': 'unknown', 'confidence': 0}
        
        c = df['Close'].values
        h = df['High'].values
        l = df['Low'].values
        
        # ADX
        adx = MarketReader._calculate_adx(df)
        current_adx = adx[-1]
        
        # Bollinger Band Width
        sma20 = np.mean(c[-20:])
        std20 = np.std(c[-20:])
        bb_width = (4 * std20) / sma20 * 100
        
        historical_widths = []
        for i in range(50, len(c)):
            sma = np.mean(c[i-20:i])
            std = np.std(c[i-20:i])
            historical_widths.append((4 * std) / sma * 100)
        avg_width = np.mean(historical_widths) if historical_widths else bb_width
        
        # Breakout detection
        high_20 = np.max(h[-20:])
        low_20 = np.min(l[-20:])
        current_price = c[-1]
        near_breakout_high = current_price > high_20 * 0.998
        near_breakout_low = current_price < low_20 * 1.002
        
        # Scoring
        scores = {'trend': 0, 'range': 0, 'breakout': 0}
        
        if current_adx > 25: scores['trend'] += 3
        elif current_adx > 20: scores['trend'] += 1
        
        if bb_width < avg_width * 0.7: scores['range'] += 3
        elif bb_width < avg_width * 0.9: scores['range'] += 1
        
        if near_breakout_high or near_breakout_low: scores['breakout'] += 2
        if bb_width > avg_width * 1.3:
            scores['trend'] += 1
            scores['breakout'] += 1
        if current_adx < 20: scores['range'] += 2
        
        market_type = max(scores, key=scores.get)
        confidence = scores[market_type] / max(sum(scores.values()), 1)
        
        return {
            'type': market_type,
            'confidence': confidence,
            'adx': round(current_adx, 1),
            'bb_width': round(bb_width, 2),
            'scores': scores
        }
    
    @staticmethod
    def _calculate_adx(df: pd.DataFrame, period: int = 14) -> np.ndarray:
        h = df['High'].values
        l = df['Low'].values
        c = df['Close'].values
        
        tr1 = h[1:] - l[1:]
        tr2 = np.abs(h[1:] - c[:-1])
        tr3 = np.abs(l[1:] - c[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
        
        up_move = h[1:] - h[:-1]
        down_move = l[:-1] - l[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / (atr + 1e-8)
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / (atr + 1e-8)
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
        
        return adx

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
    f['macd_h'] = f['macd'] - f['macd_s']
    
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f['bb'] = (c - sma20) / (2 * std20 + 1e-8)
    f['bb_w'] = (4 * std20) / (sma20 + 1e-8)
    
    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    f['atr'] = tr.ewm(span=14).mean()
    f['atr_pct'] = f['atr'] / (c + 1e-8)
    
    return f.fillna(0)

# ============================================================================
# STRATEGY ENGINE
# ============================================================================

class StrategyEngine:
    def __init__(self):
        self.models = {'trend': None, 'range': None, 'breakout': None}
        self.scalers = {'trend': RobustScaler(), 'range': RobustScaler(), 'breakout': RobustScaler()}
        self.features = {}
        self.is_trained = False
    
    def train_all(self, df: pd.DataFrame):
        logger.info("🎓 تدريب الاستراتيجيات...")
        
        X = calculate_features(df)
        future = df['Close'].shift(-5)
        change = (future - df['Close']) / df['Close'] * 100
        
        ema20 = df['Close'].ewm(span=20).mean()
        ema50 = df['Close'].ewm(span=50).mean()
        uptrend = ema20 > ema50
        downtrend = ema20 < ema50
        
        rsi = X['rsi']
        high_20 = df['High'].rolling(20).max()
        low_20 = df['Low'].rolling(20).min()
        breakout_up = df['Close'] > high_20.shift(1)
        breakout_down = df['Close'] < low_20.shift(1)
        
        targets = {
            'trend': pd.Series(np.nan, index=df.index),
            'range': pd.Series(np.nan, index=df.index),
            'breakout': pd.Series(np.nan, index=df.index)
        }
        
        targets['trend'][(uptrend) & (change > 0.05)] = 1
        targets['trend'][(downtrend) & (change < -0.05)] = 0
        targets['range'][(rsi < 35) & (change > 0.03)] = 1
        targets['range'][(rsi > 65) & (change < -0.03)] = 0
        targets['breakout'][(breakout_up) & (change > 0.08)] = 1
        targets['breakout'][(breakout_down) & (change < -0.08)] = 0
        
        for name in ['trend', 'range', 'breakout']:
            y = targets[name]
            valid = ~(X.isna().any(axis=1) | y.isna())
            X_valid = X[valid]
            y_valid = y[valid]
            
            if len(X_valid) < 100:
                continue
            
            mi = mutual_info_classif(X_valid, y_valid, random_state=42)
            scores = sorted(zip(X_valid.columns, mi), key=lambda x: x[1], reverse=True)
            self.features[name] = [s[0] for s in scores[:12]]
            
            X_sel = X_valid[self.features[name]]
            split = int(len(X_sel) * 0.8)
            X_train_s = self.scalers[name].fit_transform(X_sel[:split])
            
            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                random_state=42, verbosity=0, tree_method='hist'
            )
            model.fit(X_train_s, y_valid[:split])
            self.models[name] = model
            logger.info(f"  ✅ {name}: {len(X_valid)} عينة")
        
        self.is_trained = True
    
    def predict(self, df: pd.DataFrame, strategy: str, threshold: float = 0.55) -> Tuple[Optional[str], float]:
        if not self.is_trained or strategy not in self.models or self.models[strategy] is None:
            return None, 0.0
        
        X = calculate_features(df).iloc[[-1]]
        features = self.features.get(strategy, [])
        available = [f for f in features if f in X.columns]
        
        if len(available) < 5:
            return None, 0.0
        
        X = X[available].fillna(0)
        X_s = self.scalers[strategy].transform(X)
        
        proba = self.models[strategy].predict_proba(X_s)[0]
        
        if proba[1] > threshold:
            return 'BUY', proba[1]
        elif proba[0] > threshold:
            return 'SELL', proba[0]
        
        return None, max(proba)
    
    def save(self):
        os.makedirs('models_v11', exist_ok=True)
        joblib.dump({
            'models': self.models,
            'scalers': self.scalers,
            'features': self.features
        }, 'models_v11/strategies.pkl')
    
    def load(self) -> bool:
        path = 'models_v11/strategies.pkl'
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.models = data['models']
        self.scalers = data['scalers']
        self.features = data['features']
        self.is_trained = True
        return True

# ============================================================================
# MAIN BOT
# ============================================================================

class FalconV11:
    def __init__(self):
        self.db = Database()
        self.reader = MarketReader()
        self.engine = StrategyEngine()
        
        self.tb = telebot.TeleBot(TELEGRAM_TOKEN)
        self._setup_bot()
        
        if self.engine.load():
            logger.info("📂 تم تحميل الاستراتيجيات")
        else:
            logger.info("⚠️ استخدم /train للتدريب")
    
    def _fetch_live(self, symbol: str) -> Optional[pd.DataFrame]:
        # Yahoo Finance
        try:
            import yfinance as yf
            df = yf.download(f'{symbol}=X', period='5d', interval='5m', progress=False)
            if not df.empty:
                df.columns = [c.capitalize() for c in df.columns]
                return df
        except:
            pass
        
        # Alpha Vantage
        try:
            from_curr = symbol[:3]
            to_curr = symbol[3:]
            params = {
                'function': 'FX_INTRADAY',
                'from_symbol': from_curr,
                'to_symbol': to_curr,
                'interval': '5min',
                'outputsize': 'compact',
                'apikey': ALPHA_VANTAGE_KEY
            }
            r = requests.get('https://www.alphavantage.co/query', params=params, timeout=10)
            data = r.json()
            key = 'Time Series FX (5min)'
            if key in data:
                records = []
                for ts, vals in data[key].items():
                    records.append({
                        'Date': ts, 'Open': float(vals['1. open']),
                        'High': float(vals['2. high']), 'Low': float(vals['3. low']),
                        'Close': float(vals['4. close']), 'Volume': 0
                    })
                df = pd.DataFrame(records)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                return df
        except:
            pass
        
        return None
    
    def _fetch_training_data(self) -> Optional[pd.DataFrame]:
        """جلب بيانات للتدريب"""
        # Alpha Vantage
        try:
            params = {
                'function': 'FX_INTRADAY',
                'from_symbol': 'EUR', 'to_symbol': 'USD',
                'interval': '15min', 'outputsize': 'full',
                'apikey': ALPHA_VANTAGE_KEY
            }
            r = requests.get('https://www.alphavantage.co/query', params=params, timeout=15)
            data = r.json()
            key = 'Time Series FX (15min)'
            if key in data:
                records = []
                for ts, vals in data[key].items():
                    records.append({
                        'Date': ts, 'Open': float(vals['1. open']),
                        'High': float(vals['2. high']), 'Low': float(vals['3. low']),
                        'Close': float(vals['4. close']), 'Volume': 0
                    })
                df = pd.DataFrame(records)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                df.columns = [c.capitalize() for c in df.columns]
                logger.info(f"✅ Alpha Vantage: {len(df)} صف")
                return df
        except:
            pass
        
        # Yahoo بديل
        try:
            import yfinance as yf
            df = yf.download('EURUSD=X', period='2mo', interval='15m', progress=False)
            if not df.empty:
                df.columns = [c.capitalize() for c in df.columns]
                logger.info(f"✅ Yahoo: {len(df)} صف")
                return df
        except:
            pass
        
        return None
    
    def _setup_bot(self):
        try:
            requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=3)
        except:
            pass
        
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != TELEGRAM_CHAT_ID:
                return
            
            if self.engine.is_trained:
                stats = self.db.get_strategy_stats()
                text = (f"🦅 **Falcon V11**\n\n"
                       f"🧠 3 استراتيجيات\n"
                       f"📈 ترند: {stats['trend']['win_rate']:.0%}\n"
                       f"📊 تذبذب: {stats['range']['win_rate']:.0%}\n"
                       f"🚀 اختراق: {stats['breakout']['win_rate']:.0%}")
            else:
                text = "🦅 **Falcon V11**\n\n⚠️ استخدم /train للتدريب"
            
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['train'])
        def train_cmd(msg):
            if str(msg.chat.id) != TELEGRAM_CHAT_ID:
                return
            
            self.tb.reply_to(msg, "📥 جلب بيانات للتدريب...")
            df = self._fetch_training_data()
            
            if df is None or len(df) < 100:
                self.tb.reply_to(msg, "❌ لا توجد بيانات كافية")
                return
            
            self.tb.reply_to(msg, f"🎓 تدريب على {len(df)} صف...")
            self.engine.train_all(df)
            self.engine.save()
            self.tb.reply_to(msg, "✅ التدريب اكتمل!")
        
        @self.tb.message_handler(func=lambda msg: True)
        def analyze_any(msg):
            if str(msg.chat.id) != TELEGRAM_CHAT_ID:
                return
            
            if not self.engine.is_trained:
                self.tb.reply_to(msg, "❌ استخدم /train الأول")
                return
            
            symbol = msg.text.strip().upper()
            self.tb.reply_to(msg, f"🔍 تحليل {symbol}...")
            
            df = self._fetch_live(symbol)
            if df is None:
                self.tb.reply_to(msg, "❌ لا بيانات")
                return
            
            market = self.reader.read(df)
            strategy = market['type']
            direction, confidence = self.engine.predict(df, strategy, 0.52)
            price = float(df['Close'].iloc[-1])
            
            if direction:
                emoji = "🟢" if direction == 'BUY' else "🔴"
                dir_ar = "شراء ▲" if direction == 'BUY' else "بيع ▼"
                text = (f"📊 **{symbol}**\n\n"
                       f"🏷 {strategy}\n{emoji} {dir_ar}\n"
                       f"💰 {price:.5f}\n💪 {confidence:.1%}")
            else:
                text = (f"📊 **{symbol}**\n\n🏷 {strategy}\n💰 {price:.5f}\n⏳ انتظار")
            
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def analyze(self, symbol: str) -> Optional[Dict]:
        if not self.engine.is_trained:
            return None
        
        if self.db.has_active_signal(symbol):
            return None
        
        if self.db.was_recent(symbol, 5):
            return None
        
        df = self._fetch_live(symbol)
        if df is None:
            return None
        
        market = self.reader.read(df)
        strategy = market['type']
        
        if market['confidence'] < 0.4:
            return None
        
        threshold = 0.55 if strategy == 'trend' else (0.52 if strategy == 'range' else 0.58)
        direction, confidence = self.engine.predict(df, strategy, threshold)
        
        if direction is None:
            return None
        
        duration = 5 if strategy == 'breakout' else (7 if strategy == 'trend' else 10)
        entry = float(df['Close'].iloc[-1])
        
        logger.info(f"🎯 {symbol}: {direction} | {strategy} | {confidence:.1%}")
        
        return {
            'symbol': symbol, 'direction': direction,
            'entry_price': entry, 'confidence': confidence,
            'strategy': strategy, 'market_type': strategy,
            'duration': duration,
            'expiry_time': (datetime.now() + timedelta(minutes=duration)).strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def send_signal(self, signal: Dict):
        emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
        strategy_emoji = {'trend': '📈', 'range': '📊', 'breakout': '🚀'}
        
        msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
               f"💰 {signal['entry_price']:.5f}\n"
               f"⏳ {signal['duration']} د\n"
               f"💪 {signal['confidence']:.1%}\n"
               f"{strategy_emoji.get(signal['strategy'], '')} {signal['strategy']}\n\n"
               f"🤖 Falcon V11")
        
        try:
            self.tb.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
        except:
            pass
    
    def check_trades(self):
        for trade in self.db.get_expired_trades():
            try:
                df = self._fetch_live(trade['symbol'])
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
    
    def scan(self):
        for symbol in SYMBOLS:
            try:
                signal = self.analyze(symbol)
                if signal and self.db.save_signal(signal):
                    self.send_signal(signal)
            except:
                pass
    
    def run(self):
        logger.info("🦅 Falcon V11 - Adaptive Strategy Switcher")
        
        def poll():
            while True:
                try:
                    self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except:
                    time.sleep(5)
        threading.Thread(target=poll, daemon=True).start()
        time.sleep(1)
        
        try:
            status_text = "🦅 **Falcon V11**\n\n🧠 3 استراتيجيات\n📈 📊 🚀\n⚡️ يعمل..."
            if not self.engine.is_trained:
                status_text += "\n⚠️ استخدم /train"
            self.tb.send_message(TELEGRAM_CHAT_ID, status_text, parse_mode='Markdown')
        except:
            pass
        
        while True:
            try:
                self.check_trades()
                self.scan()
                time.sleep(60)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"حلقة: {e}")
                time.sleep(30)

if __name__ == "__main__":
    os.makedirs('models_v11', exist_ok=True)
    bot = FalconV11()
    bot.run()
