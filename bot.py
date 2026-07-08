#!/usr/bin/env python3
"""
Falcon AI v10 - Complete System
=================================
Phase 1: Gather Data (Yahoo + Alpha Vantage)
Phase 2: Train Models (XGBoost + RandomForest + GradientBoost)
Phase 3: Backtest Strategies
Phase 4: Evolve Brain (Choose Best)
Phase 5: Trade Smart (Send Signals)
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
from collections import deque
import numpy as np
import pandas as pd
import requests

from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import xgboost as xgb

import telebot
import joblib

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FalconV10')

# ============================================================================
# CONFIG
# ============================================================================

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '7553333305')
ALPHA_VANTAGE_KEY = os.environ.get('ALPHA_VANTAGE_KEY', '5TFFWK21CUNA3P25')

SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
    'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY'
]

DB_PATH = 'falcon_v10.db'
MODELS_DIR = 'models_v10'
BRAIN_FILE = 'brain_v10.json'

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        with sqlite3.connect(db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, timestamp DATETIME,
                    open REAL, high REAL, low REAL, close REAL,
                    UNIQUE(symbol, timestamp)
                );
                
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    exit_price REAL, confidence REAL,
                    duration INTEGER, model_name TEXT,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, exit_time DATETIME,
                    result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, pnl_pips REAL,
                    signal_hash TEXT UNIQUE
                );
                
                CREATE INDEX IF NOT EXISTS idx_market ON market_data(symbol, timestamp);
            ''')
            conn.commit()
    
    def save_signal(self, data: Dict) -> Optional[int]:
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO signals 
                    (symbol, direction, entry_price, confidence, duration, model_name,
                     expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data.get('duration', 7), data.get('model_name', ''),
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
    
    def get_data_count(self, symbol: str) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute('SELECT COUNT(*) FROM market_data WHERE symbol=?', 
                              (symbol,)).fetchone()[0]

# ============================================================================
# PHASE 1: DATA GATHERER
# ============================================================================

class DataGatherer:
    def __init__(self, db: Database):
        self.db = db
    
    def fetch_all(self):
        logger.info("📥 Phase 1: جمع البيانات...")
        
        for symbol in SYMBOLS:
            self._fetch_yahoo(symbol)
            time.sleep(2)
        
        # إحصائيات
        for symbol in SYMBOLS:
            count = self.db.get_data_count(symbol)
            logger.info(f"  📊 {symbol}: {count} صف")
        
        logger.info("✅ Phase 1: اكتمل")
    
    def _fetch_yahoo(self, symbol: str) -> bool:
        try:
            import yfinance as yf
            
            yf_symbol = f"{symbol}=X"
            df = yf.download(yf_symbol, period='2mo', interval='15m', progress=False)
            
            if df is None or df.empty:
                return False
            
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            count = 0
            with sqlite3.connect(self.db.db_path) as conn:
                for idx, row in df.iterrows():
                    conn.execute('''
                        INSERT OR IGNORE INTO market_data (symbol, timestamp, open, high, low, close)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (symbol, str(idx), float(row['open']), float(row['high']),
                          float(row['low']), float(row['close'])))
                    count += 1
                conn.commit()
            
            logger.info(f"  ✅ {symbol}: {count} صف (Yahoo)")
            return count > 50
            
        except Exception as e:
            logger.error(f"  ❌ {symbol}: {e}")
            return False
    
    def fetch_live(self, symbol: str) -> Optional[pd.DataFrame]:
        """جلب بيانات حية للفحص"""
        try:
            import yfinance as yf
            
            yf_symbol = f"{symbol}=X"
            df = yf.download(yf_symbol, period='5d', interval='5m', progress=False)
            
            if df is not None and not df.empty:
                df.columns = [c.capitalize() for c in df.columns]
                return df
            
        except:
            pass
        
        # Alpha Vantage كبديل
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
                        'Date': ts,
                        'Open': float(vals['1. open']),
                        'High': float(vals['2. high']),
                        'Low': float(vals['3. low']),
                        'Close': float(vals['4. close']),
                        'Volume': 0
                    })
                
                df = pd.DataFrame(records)
                df['Date'] = pd.to_datetime(df['Date'])
                df = df.set_index('Date').sort_index()
                return df
        except:
            pass
        
        return None

# ============================================================================
# PHASE 2+3+4: TRAINER + BACKTESTER + BRAIN
# ============================================================================

class Brain:
    def __init__(self, db: Database):
        self.db = db
        self.best_model = None
        self.best_scaler = None
        self.best_features = None
        self.best_threshold = 0.55
        self.best_duration = 7
        self.best_name = ""
        self.is_trained = False
    
    def get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        with sqlite3.connect(self.db.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM market_data WHERE symbol=?
                ORDER BY timestamp ASC
            ''', conn, params=(symbol,))
        
        if df.empty:
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        c, h, l = df['close'], df['high'], df['low']
        
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
        
        tr1 = h - l
        tr2 = abs(h - c.shift())
        tr3 = abs(l - c.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        f['atr'] = tr.ewm(span=14).mean()
        f['atr_pct'] = f['atr'] / (c + 1e-8)
        
        return f.fillna(0)
    
    def create_target(self, df: pd.DataFrame, periods: int = 5) -> pd.Series:
        future = df['close'].shift(-periods)
        change = (future - df['close']) / df['close'] * 100
        target = pd.Series(np.nan, index=df.index)
        target[change > 0.03] = 1
        target[change < -0.03] = 0
        return target
    
    def evolve(self):
        """تدريب واختبار واختيار أفضل استراتيجية"""
        logger.info("🧠 Phase 2-4: تدريب واختبار وتعلم...")
        
        best_overall = {'win_rate': 0, 'total': 0}
        
        for symbol in SYMBOLS[:4]:  # أول 4 أزواج
            df = self.get_data(symbol)
            if df is None or len(df) < 500:
                logger.warning(f"  ⚠️ {symbol}: بيانات غير كافية")
                continue
            
            X = self.create_features(df)
            y = self.create_target(df)
            
            valid = ~(X.isna().any(axis=1) | y.isna())
            X, y = X[valid], y[valid]
            
            if len(X) < 200:
                continue
            
            features_list = list(X.columns)
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            scaler = RobustScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)
            
            # تدريب 3 نماذج
            models = []
            
            # XGBoost
            xgb_m = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                       random_state=42, verbosity=0, tree_method='hist')
            xgb_m.fit(X_train_s, y_train)
            models.append(('XGBoost', xgb_m))
            
            # RandomForest
            rf_m = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
            rf_m.fit(X_train_s, y_train)
            models.append(('RandomForest', rf_m))
            
            # GradientBoosting
            gb_m = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
            gb_m.fit(X_train_s, y_train)
            models.append(('GradientBoost', gb_m))
            
            # اختبار كل نموذج
            for name, model in models:
                for duration in [5, 7, 10]:
                    for threshold in [0.52, 0.55, 0.58, 0.62]:
                        result = self._backtest(model, scaler, features_list, 
                                                X_test, df.iloc[split:], duration, threshold)
                        
                        if result['total'] >= 10 and result['win_rate'] > best_overall['win_rate']:
                            best_overall = result
                            self.best_model = model
                            self.best_scaler = scaler
                            self.best_features = features_list
                            self.best_threshold = threshold
                            self.best_duration = duration
                            self.best_name = f"{symbol}_{name}"
                            
                            logger.info(f"  ⭐ {symbol} {name}: {result['win_rate']:.1%} "
                                      f"({result['wins']}/{result['total']}) | {duration}د | {threshold:.0%}")
        
        if self.best_model:
            self.is_trained = True
            self._save_brain()
            logger.info(f"🏆 أفضل استراتيجية: {self.best_name} "
                       f"نسبة={best_overall['win_rate']:.1%}")
            return True
        
        return False
    
    def _backtest(self, model, scaler, features, X_test, df_test, duration, threshold):
        """اختبار سريع"""
        results = []
        
        for i in range(0, len(X_test) - duration, 5):
            X_s = X_test.iloc[[i]]
            proba = model.predict_proba(X_s)[0]
            
            if proba[1] > threshold:
                direction = 'BUY'
            elif proba[0] > threshold:
                direction = 'SELL'
            else:
                continue
            
            entry = float(df_test['close'].iloc[i])
            exit_idx = min(i + duration * 3, len(df_test) - 1)
            exit_price = float(df_test['close'].iloc[exit_idx])
            
            if direction == 'BUY':
                pnl = (exit_price - entry) / entry * 100
                win = exit_price > entry
            else:
                pnl = (entry - exit_price) / entry * 100
                win = exit_price < entry
            
            results.append({'win': win, 'pnl': pnl})
        
        if not results:
            return {'total': 0, 'wins': 0, 'win_rate': 0}
        
        wins = sum(1 for r in results if r['win'])
        return {
            'total': len(results), 'wins': wins,
            'losses': len(results) - wins,
            'win_rate': wins / len(results)
        }
    
    def predict(self, df: pd.DataFrame) -> Tuple[Optional[str], float]:
        """تنبؤ باستخدام أفضل نموذج"""
        if not self.is_trained:
            return None, 0.0
        
        X = self.create_features(df).iloc[[-1]]
        available = [f for f in self.best_features if f in X.columns]
        
        if len(available) < 5:
            return None, 0.0
        
        X = X[available].fillna(0)
        X_s = self.best_scaler.transform(X)
        
        proba = self.best_model.predict_proba(X_s)[0]
        
        if proba[1] > self.best_threshold:
            return 'BUY', proba[1]
        elif proba[0] > self.best_threshold:
            return 'SELL', proba[0]
        
        return None, max(proba)
    
    def _save_brain(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump({
            'model': self.best_model,
            'scaler': self.best_scaler,
            'features': self.best_features,
            'threshold': self.best_threshold,
            'duration': self.best_duration,
            'name': self.best_name
        }, f"{MODELS_DIR}/best_model.pkl")
        
        with open(BRAIN_FILE, 'w') as f:
            json.dump({
                'threshold': self.best_threshold,
                'duration': self.best_duration,
                'name': self.best_name,
                'is_trained': self.is_trained
            }, f)
    
    def load(self):
        path = f"{MODELS_DIR}/best_model.pkl"
        if os.path.exists(path):
            data = joblib.load(path)
            self.best_model = data['model']
            self.best_scaler = data['scaler']
            self.best_features = data['features']
            self.best_threshold = data['threshold']
            self.best_duration = data['duration']
            self.best_name = data['name']
            self.is_trained = True
            logger.info(f"📂 تحميل: {self.best_name}")
            return True
        return False

# ============================================================================
# PHASE 5: TRADER
# ============================================================================

class Trader:
    def __init__(self, db: Database, brain: Brain, gatherer: DataGatherer):
        self.db = db
        self.brain = brain
        self.gatherer = gatherer
        self.tb = telebot.TeleBot(TELEGRAM_TOKEN)
        self._setup_bot()
    
    def _setup_bot(self):
        try:
            requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=3)
        except:
            pass
        
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != TELEGRAM_CHAT_ID:
                return
            
            if self.brain.is_trained:
                text = (f"🦅 **Falcon V10**\n\n"
                       f"🧠 العقل: نشط\n"
                       f"⭐ {self.brain.best_name}\n"
                       f"🎯 العتبة: {self.brain.best_threshold:.0%}\n"
                       f"⏱️ المدة: {self.brain.best_duration} د")
            else:
                text = "🦅 **Falcon V10**\n\n🧠 العقل: يحتاج تدريب\n📥 استخدم /train"
            
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['train'])
        def train_cmd(msg):
            if str(msg.chat.id) != TELEGRAM_CHAT_ID:
                return
            
            self.tb.reply_to(msg, "📥 جمع البيانات...")
            self.gatherer.fetch_all()
            
            self.tb.reply_to(msg, "🧠 تدريب واختبار...")
            success = self.brain.evolve()
            
            if success:
                self.tb.reply_to(msg, f"✅ جاهز!\n⭐ {self.brain.best_name}\n📈 العتبة: {self.brain.best_threshold:.0%}")
            else:
                self.tb.reply_to(msg, "❌ فشل. جرب تاني.")
        
        @self.tb.message_handler(func=lambda msg: True)
        def analyze_any(msg):
            if str(msg.chat.id) != TELEGRAM_CHAT_ID:
                return
            
            symbol = msg.text.strip().upper()
            
            if not self.brain.is_trained:
                self.tb.reply_to(msg, "❌ العقل مش مدرب. استخدم /train")
                return
            
            self.tb.reply_to(msg, f"🔍 تحليل {symbol}...")
            
            df = self.gatherer.fetch_live(symbol)
            if df is None:
                self.tb.reply_to(msg, "❌ لا بيانات")
                return
            
            direction, confidence = self.brain.predict(df)
            
            if direction is None:
                self.tb.reply_to(msg, f"💰 {float(df['Close'].iloc[-1]):.5f}\n⏳ انتظار...")
                return
            
            emoji = "🟢" if direction == 'BUY' else "🔴"
            dir_ar = "شراء ▲" if direction == 'BUY' else "بيع ▼"
            
            text = (f"📊 **{symbol}**\n\n"
                   f"{emoji} {dir_ar}\n"
                   f"💰 {float(df['Close'].iloc[-1]):.5f}\n"
                   f"💪 {confidence:.1%}")
            
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def analyze(self, symbol: str) -> Optional[Dict]:
        if not self.brain.is_trained:
            return None
        
        if self.db.has_active_signal(symbol):
            return None
        
        if self.db.was_recent(symbol, 5):
            return None
        
        df = self.gatherer.fetch_live(symbol)
        if df is None:
            return None
        
        direction, confidence = self.brain.predict(df)
        
        if direction is None:
            return None
        
        entry = float(df['Close'].iloc[-1])
        
        return {
            'symbol': symbol,
            'direction': direction,
            'entry_price': entry,
            'confidence': confidence,
            'duration': self.brain.best_duration,
            'model_name': self.brain.best_name,
            'expiry_time': (datetime.now() + timedelta(minutes=self.brain.best_duration)).strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def send_signal(self, signal: Dict):
        emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
        
        msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
               f"💰 {signal['entry_price']:.5f}\n"
               f"⏳ {signal['duration']} د\n"
               f"💪 {signal['confidence']:.1%}\n\n"
               f"🤖 Falcon V10")
        
        try:
            self.tb.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
            logger.info(f"✅ {signal['symbol']} {signal['direction']}")
        except:
            pass
    
    def check_trades(self):
        for trade in self.db.get_expired_trades():
            try:
                df = self.gatherer.fetch_live(trade['symbol'])
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
                logger.info(f"{'✅' if result == 'WIN' else '❌'} {trade['symbol']}: {result} | {pnl:+.2f}%")
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
        logger.info("📡 Phase 5: بدء المراقبة...")
        
        def poll():
            while True:
                try:
                    self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except:
                    time.sleep(5)
        threading.Thread(target=poll, daemon=True).start()
        time.sleep(1)
        
        try:
            status_text = f"🦅 **Falcon V10**\n\n✅ جاهز\n⭐ {self.brain.best_name}\n⚡️ المراقبة بدأت..."
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

# ============================================================================
# MAIN
# ============================================================================

class FalconV10:
    def __init__(self):
        self.db = Database()
        self.gatherer = DataGatherer(self.db)
        self.brain = Brain(self.db)
        self.trader = Trader(self.db, self.brain, self.gatherer)
    
    def run(self):
        logger.info("=" * 50)
        logger.info("🦅 Falcon V10 - Born to Learn")
        logger.info("=" * 50)
        
        # حاول تحميل عقل موجود
        if not self.brain.load():
            # Phase 1: جمع البيانات
            self.gatherer.fetch_all()
            
            # Phase 2-4: تدريب واختبار وتعلم
            self.brain.evolve()
        
        # Phase 5: تداول
        self.trader.run()

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    bot = FalconV10()
    bot.run()
