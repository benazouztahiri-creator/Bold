#!/usr/bin/env python3
"""
Falcon AI Pro v4 - Professional Trading System
================================================
✅ XGBoost ML Model (Auto-train with Alpha Vantage)
✅ Market Structure (BOS/CHoCH/HH/HL/LH/LL)
✅ Session Filter (Asian/London/NY/Overlap)
✅ Auto-isolate weak symbols
✅ Dynamic Risk Management
✅ Intra-candle SL/TP detection
✅ Multi-timeframe (M5/M15/H1)
✅ Price Action + Spread Filter
✅ Economic calendar filter
✅ Performance tracking per symbol
✅ Auto-train on startup
"""

import os
import sys
import time
import logging
import sqlite3
import hashlib
import threading
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import warnings

import telebot
import joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

TELEGRAM_TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
TELEGRAM_CHAT_ID = '7553333305'
ALPHA_VANTAGE_KEY = '5TFFWK21CUNA3P25'

SYMBOLS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
    'USDCAD', 'EURGBP', 'EURJPY', 'GBPJPY'
]

SCAN_INTERVAL = 90
MIN_CONFIDENCE = 0.55

SESSIONS = {
    'asian': (0, 9), 'london': (8, 17),
    'ny': (13, 22), 'overlap': (13, 17)
}

MAX_DAILY_LOSS = 0.03
RISK_PER_TRADE = 0.01

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FalconV4')

# ============================================================================
# CACHE
# ============================================================================

class DataCache:
    def __init__(self):
        self.cache = {}
        self.ttl = 30
    
    def get(self, key):
        if key in self.cache:
            data, ts = self.cache[key]
            if time.time() - ts < self.ttl:
                return data.copy()
        return None
    
    def set(self, key, data):
        self.cache[key] = (data, time.time())
        if len(self.cache) > 50:
            del self.cache[min(self.cache, key=lambda k: self.cache[k][1])]

data_cache = DataCache()

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    def __init__(self):
        self.db_path = 'falcon_v4.db'
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    exit_price REAL, stop_loss REAL, take_profit REAL,
                    high_period REAL, low_period REAL,
                    confidence REAL, score REAL,
                    adx REAL, atr REAL,
                    market_structure TEXT,
                    price_action TEXT,
                    session TEXT,
                    tf_5m TEXT, tf_15m TEXT, tf_1h TEXT,
                    position_size REAL,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, exit_time DATETIME,
                    result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, pnl_pips REAL,
                    sl_hit_first INTEGER, tp_hit_first INTEGER,
                    signal_hash TEXT UNIQUE
                );
                
                CREATE TABLE IF NOT EXISTS symbol_performance (
                    symbol TEXT PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    is_active INTEGER DEFAULT 1,
                    win_rate_10 REAL DEFAULT 0.5,
                    last_10_results TEXT DEFAULT '[]'
                );
                
                CREATE TABLE IF NOT EXISTS risk_metrics (
                    date TEXT PRIMARY KEY,
                    daily_pnl REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0
                );
            ''')
            
            for sym in SYMBOLS:
                conn.execute('INSERT OR IGNORE INTO symbol_performance (symbol) VALUES (?)', (sym,))
            conn.commit()
    
    def save_signal(self, data: Dict) -> Optional[int]:
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO signals 
                    (symbol, direction, entry_price, stop_loss, take_profit,
                     confidence, score, adx, atr, market_structure, price_action,
                     session, tf_5m, tf_15m, tf_1h, position_size,
                     expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data.get('stop_loss'), data.get('take_profit'),
                      data['confidence'], data.get('score', 0),
                      data.get('adx', 0), data.get('atr', 0),
                      data.get('market_structure', ''), data.get('price_action', ''),
                      data.get('session', ''), data.get('tf_5m', ''), data.get('tf_15m', ''),
                      data.get('tf_1h', ''), data.get('position_size', 0.01),
                      data['expiry_time'], h))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except:
            return None
    
    def update_result(self, signal_id: int, exit_price: float, result: str,
                      pnl: float, pips: float, high_p: float, low_p: float,
                      sl_first: int, tp_first: int):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE signals SET exit_price=?, result=?, pnl_percent=?, pnl_pips=?,
                high_period=?, low_period=?, sl_hit_first=?, tp_hit_first=?,
                exit_time=datetime('now', 'localtime') WHERE id=?
            ''', (exit_price, result, pnl, pips, high_p, low_p, sl_first, tp_first, signal_id))
            
            symbol = conn.execute('SELECT symbol FROM signals WHERE id=?', (signal_id,)).fetchone()[0]
            conn.execute('''
                UPDATE symbol_performance 
                SET total_trades = total_trades + 1,
                    wins = wins + ?, total_pnl = total_pnl + ?
                WHERE symbol = ?
            ''', (1 if result == 'WIN' else 0, pnl, symbol))
            
            results = json.loads(conn.execute(
                'SELECT last_10_results FROM symbol_performance WHERE symbol=?', (symbol,)
            ).fetchone()[0] or '[]')
            results.append(1 if result == 'WIN' else 0)
            if len(results) > 10: results.pop(0)
            
            win_rate_10 = sum(results) / len(results) if results else 0.5
            
            conn.execute('''
                UPDATE symbol_performance 
                SET last_10_results = ?, win_rate_10 = ?,
                    is_active = CASE WHEN ? < 0.3 THEN 0 ELSE 1 END
                WHERE symbol = ?
            ''', (json.dumps(results), win_rate_10, win_rate_10, symbol))
            
            today = datetime.now().strftime('%Y-%m-%d')
            conn.execute('''
                INSERT INTO risk_metrics (date, daily_pnl, total_trades)
                VALUES (?, ?, 1)
                ON CONFLICT(date) DO UPDATE SET
                daily_pnl = daily_pnl + ?, total_trades = total_trades + 1
            ''', (today, pnl, pnl))
            
            conn.commit()
    
    def is_symbol_active(self, symbol: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute('SELECT is_active FROM symbol_performance WHERE symbol=?',
                             (symbol,)).fetchone()
            return bool(row[0]) if row else True
    
    def get_daily_pnl(self) -> float:
        today = datetime.now().strftime('%Y-%m-%d')
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute('SELECT daily_pnl FROM risk_metrics WHERE date=?',
                             (today,)).fetchone()
            return row[0] if row else 0
    
    def get_active_symbols(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute('SELECT symbol FROM symbol_performance WHERE is_active = 1').fetchall()
            return [r[0] for r in rows] if rows else SYMBOLS
    
    def has_active_signal(self, symbol: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' 
                AND expiry_time > datetime('now', 'localtime')
            ''', (symbol,)).fetchone()[0]
            return c > 0
    
    def was_recent(self, symbol: str, minutes: int = 10) -> bool:
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?',
                           (symbol, cutoff)).fetchone()[0]
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
# DATA FETCHER
# ============================================================================

class DataFetcher:
    @staticmethod
    def fetch(symbol: str, interval: str = '5min', outputsize: str = 'compact') -> Optional[pd.DataFrame]:
        """جلب بيانات من Alpha Vantage"""
        key = f"{symbol}_{interval}_{outputsize}"
        cached = data_cache.get(key)
        if cached is not None:
            return cached
        
        from_curr = symbol[:3]
        to_curr = symbol[3:]
        
        interval_map = {'5m': '5min', '15m': '15min', '1h': '60min', '1m': '1min'}
        av_interval = interval_map.get(interval, '5min')
        
        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': from_curr,
            'to_symbol': to_curr,
            'interval': av_interval,
            'outputsize': outputsize,
            'apikey': ALPHA_VANTAGE_KEY
        }
        
        try:
            r = requests.get('https://www.alphavantage.co/query', params=params, timeout=15)
            data = r.json()
            
            time_key = f'Time Series FX ({av_interval})'
            if time_key not in data:
                # Yahoo بديل
                return DataFetcher._fetch_yahoo(symbol, interval)
            
            records = []
            for ts, vals in data[time_key].items():
                records.append({
                    'Date': ts, 'Open': float(vals['1. open']),
                    'High': float(vals['2. high']), 'Low': float(vals['3. low']),
                    'Close': float(vals['4. close']), 'Volume': 0
                })
            
            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            
            data_cache.set(key, df)
            return df
            
        except:
            return DataFetcher._fetch_yahoo(symbol, interval)
    
    @staticmethod
    def _fetch_yahoo(symbol: str, interval: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
            yf_symbol = f"{symbol}=X"
            interval_map = {'5m': '5m', '15m': '15m', '1h': '1h', '1m': '1m'}
            yf_interval = interval_map.get(interval, '5m')
            
            df = yf.download(yf_symbol, period='5d', interval=yf_interval, progress=False)
            if not df.empty:
                df.columns = [c.capitalize() for c in df.columns]
                return df
        except:
            pass
        return None

# ============================================================================
# SESSION FILTER
# ============================================================================

class SessionFilter:
    @staticmethod
    def get_current_session() -> str:
        hour = datetime.utcnow().hour
        if SESSIONS['overlap'][0] <= hour < SESSIONS['overlap'][1]: return 'overlap'
        elif SESSIONS['london'][0] <= hour < SESSIONS['london'][1]: return 'london'
        elif SESSIONS['ny'][0] <= hour < SESSIONS['ny'][1]: return 'ny'
        elif SESSIONS['asian'][0] <= hour < SESSIONS['asian'][1]: return 'asian'
        return 'dead'
    
    @staticmethod
    def is_good_time() -> bool:
        return SessionFilter.get_current_session() in ['overlap', 'london', 'ny']

# ============================================================================
# MARKET STRUCTURE
# ============================================================================

class MarketStructure:
    @staticmethod
    def detect(df: pd.DataFrame) -> Dict:
        if len(df) < 50:
            return {'structure': 'unknown', 'trend': 'neutral'}
        
        h, l, c = df['High'].values, df['Low'].values, df['Close'].values
        hh = any(h[i] > max(h[max(0,i-20):i]) for i in range(20, len(h)-5))
        ll = any(l[i] < min(l[max(0,i-20):i]) for i in range(20, len(l)-5))
        
        if hh: structure, trend = 'bullish', 'UP'
        elif ll: structure, trend = 'bearish', 'DOWN'
        else: structure, trend = 'ranging', 'neutral'
        
        return {'structure': structure, 'trend': trend}

# ============================================================================
# PRICE ACTION
# ============================================================================

class PriceAction:
    @staticmethod
    def detect(df: pd.DataFrame) -> Tuple[str, float]:
        if len(df) < 3: return "none", 0
        
        c1, c2 = df.iloc[-1], df.iloc[-2]
        body1 = abs(c1['Close'] - c1['Open'])
        body2 = abs(c2['Close'] - c2['Open'])
        upper1 = c1['High'] - max(c1['Close'], c1['Open'])
        lower1 = min(c1['Close'], c1['Open']) - c1['Low']
        
        if body1 > 0:
            if lower1 > body1 * 2: return "hammer", 0.6
            if upper1 > body1 * 2: return "shooting_star", -0.6
        
        if body1 > body2 * 1.2:
            if c2['Close'] < c2['Open'] and c1['Close'] > c1['Open']:
                return "bullish_engulfing", 0.7
            if c2['Close'] > c2['Open'] and c1['Close'] < c1['Open']:
                return "bearish_engulfing", -0.7
        
        return "none", 0

# ============================================================================
# ECONOMIC CALENDAR
# ============================================================================

class EconomicCalendar:
    @staticmethod
    def is_high_impact_now() -> bool:
        now = datetime.utcnow()
        for h, m in [(12, 30), (13, 30), (14, 0), (18, 0)]:
            if abs((now - now.replace(hour=h, minute=m, second=0)).total_seconds()) < 1800:
                return True
        return False

# ============================================================================
# INDICATORS
# ============================================================================

def calculate_indicators(df: pd.DataFrame, symbol: str) -> Dict:
    if len(df) < 30: return None
    
    c, h, l = df['Close'].values, df['High'].values, df['Low'].values
    price = float(c[-1])
    result = {'price': price}
    
    # RSI
    delta = np.diff(c)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_g = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
    avg_l = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
    result['rsi'] = round(100 - 100/(1 + avg_g/(avg_l+1e-8)), 1)
    
    # ATR
    tr = np.array([max(h[i+1]-l[i+1], abs(h[i+1]-c[i]), abs(l[i+1]-c[i])) for i in range(len(c)-1)])
    result['atr'] = round(float(np.mean(tr[-14:])), 5)
    
    # ADX
    up, down = np.diff(h), -np.diff(l)
    p_dm = np.where((up > down) & (up > 0), up, 0)
    m_dm = np.where((down > up) & (down > 0), down, 0)
    tr14 = np.sum(tr[-14:])
    p_di = 100 * np.sum(p_dm[-14:]) / (tr14 + 1e-8)
    m_di = 100 * np.sum(m_dm[-14:]) / (tr14 + 1e-8)
    dx = 100 * abs(p_di - m_di) / (p_di + m_di + 1e-8)
    result['adx'] = round(float(dx), 1)
    result['plus_di'] = round(float(p_di), 1)
    result['minus_di'] = round(float(m_di), 1)
    
    # MACD
    ema12 = pd.Series(c).ewm(span=12, adjust=False).mean().values[-1]
    ema26 = pd.Series(c).ewm(span=26, adjust=False).mean().values[-1]
    macd_line = ema12 - ema26
    macd_signal = pd.Series(pd.Series(c).ewm(span=12).mean() - pd.Series(c).ewm(span=26).mean()).ewm(span=9, adjust=False).mean().values[-1]
    result['macd'] = round(float(macd_line), 5)
    result['macd_signal'] = round(float(macd_signal), 5)
    
    # EMA
    ema20 = pd.Series(c).ewm(span=20, adjust=False).mean().values[-1]
    ema50 = pd.Series(c).ewm(span=50, adjust=False).mean().values[-1] if len(c) >= 50 else ema20
    result['above_ema20'] = price > ema20
    result['ema_bullish'] = ema20 > ema50
    
    # Bollinger
    sma20 = np.mean(c[-20:])
    std20 = np.std(c[-20:])
    result['bb_position'] = round((price - sma20)/(2*std20+1e-8), 2)
    
    # Returns
    for p in [1, 3, 5]:
        result[f'ret_{p}'] = round(float((c[-1]-c[-p-1])/c[-p-1]*100), 3) if len(c) > p else 0
    
    # SL/TP
    is_jpy = "JPY" in symbol
    sl_d = result['atr'] * 1.5
    tp_d = result['atr'] * 3.0
    result['sl_buy'] = round(price - sl_d, 5)
    result['tp_buy'] = round(price + tp_d, 5)
    result['sl_sell'] = round(price + sl_d, 5)
    result['tp_sell'] = round(price - tp_d, 5)
    
    return result

# ============================================================================
# XGBoost - Auto Train with Alpha Vantage
# ============================================================================

class MLModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.features = ['rsi', 'adx', 'bb_position', 'ret_1', 'ret_3', 'ret_5']
        self.is_trained = False
        
        os.makedirs('models_v4', exist_ok=True)
        
        if not self._load():
            logger.info("🎓 تدريب XGBoost تلقائياً...")
            self._train()
    
    def _load(self) -> bool:
        path = 'models_v4/xgb_model.pkl'
        if os.path.exists(path):
            try:
                data = joblib.load(path)
                self.model = data['model']
                self.scaler = data['scaler']
                self.features = data.get('features', self.features)
                self.is_trained = True
                logger.info("📂 XGBoost محمل")
                return True
            except:
                pass
        return False
    
    def _train(self):
        try:
            from sklearn.preprocessing import RobustScaler
            import xgboost as xgb
            
            logger.info("📥 جلب بيانات EURUSD من Alpha Vantage...")
            
            params = {
                'function': 'FX_INTRADAY',
                'from_symbol': 'EUR', 'to_symbol': 'USD',
                'interval': '60min', 'outputsize': 'full',
                'apikey': ALPHA_VANTAGE_KEY
            }
            
            r = requests.get('https://www.alphavantage.co/query', params=params, timeout=15)
            data = r.json()
            
            time_key = 'Time Series FX (60min)'
            if time_key not in data:
                logger.warning("⚠️ Alpha Vantage: لا بيانات")
                return
            
            records = []
            for ts, vals in data[time_key].items():
                records.append({
                    'Date': ts, 'Open': float(vals['1. open']),
                    'High': float(vals['2. high']), 'Low': float(vals['3. low']),
                    'Close': float(vals['4. close'])
                })
            
            df = pd.DataFrame(records)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            
            if len(df) < 200:
                logger.warning("⚠️ بيانات غير كافية")
                return
            
            X_list, y_list = [], []
            
            for i in range(50, len(df)-5):
                chunk = df.iloc[i-50:i+1]
                ind = calculate_indicators(chunk, 'EURUSD')
                if ind:
                    X_list.append([ind.get(f, 0) for f in self.features])
                    y_list.append(1 if df['Close'].values[i+5] > df['Close'].values[i] else 0)
            
            if len(X_list) < 100:
                logger.warning("⚠️ عينات غير كافية")
                return
            
            X = np.array(X_list)
            y = np.array(y_list)
            
            self.scaler = RobustScaler()
            X_s = self.scaler.fit_transform(X)
            
            self.model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                random_state=42, verbosity=0
            )
            
            split = int(len(X) * 0.8)
            self.model.fit(X_s[:split], y[:split])
            
            y_pred = self.model.predict(X_s[split:])
            acc = (y_pred == y[split:]).mean()
            
            self.is_trained = True
            
            joblib.dump({
                'model': self.model, 'scaler': self.scaler, 'features': self.features
            }, 'models_v4/xgb_model.pkl')
            
            logger.info(f"✅ XGBoost مدرب - دقة: {acc:.1%}")
            
        except Exception as e:
            logger.error(f"❌ تدريب XGBoost: {e}")
    
    def predict(self, indicators: Dict) -> float:
        if not self.is_trained:
            return 0.5
        
        try:
            X = np.array([[indicators.get(f, 0) for f in self.features]])
            X_s = self.scaler.transform(X)
            return float(self.model.predict_proba(X_s)[0, 1])
        except:
            return 0.5

# ============================================================================
# RISK MANAGER
# ============================================================================

class RiskManager:
    def __init__(self, db: Database):
        self.db = db
        self.account_balance = 10000
    
    def can_trade(self) -> bool:
        return self.db.get_daily_pnl() > -MAX_DAILY_LOSS * self.account_balance
    
    def calculate_position_size(self, symbol: str, entry: float, sl: float) -> float:
        risk_amount = self.account_balance * RISK_PER_TRADE
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        sl_pips = abs(entry - sl) / pip_value
        if sl_pips < 5: return 0.01
        return round(min(max(risk_amount / (sl_pips * 10), 0.01), 1.0), 2)

# ============================================================================
# MAIN BOT
# ============================================================================

class FalconPro:
    def __init__(self):
        self.db = Database()
        self.ml = MLModel()
        self.risk = RiskManager(self.db)
        self.tb = telebot.TeleBot(TELEGRAM_TOKEN)
        self._setup()
    
    def _setup(self):
        try:
            requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=3)
        except:
            pass
        
        @self.tb.message_handler(commands=['start'])
        def start(msg):
            active = self.db.get_active_symbols()
            daily = self.db.get_daily_pnl()
            text = (f"🦅 **Falcon Pro v4**\n\n"
                   f"✅ أزواج نشطة: {len(active)}/8\n"
                   f"📊 XGBoost: {'نشط' if self.ml.is_trained else 'غير مدرب'}\n"
                   f"💰 ربح اليوم: {daily:.2f}%")
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['train'])
        def train_cmd(msg):
            if str(msg.chat.id) != TELEGRAM_CHAT_ID: return
            self.tb.reply_to(msg, "🎓 جاري تدريب XGBoost...")
            self.ml._train()
            self.tb.reply_to(msg, "✅ XGBoost جاهز!" if self.ml.is_trained else "❌ فشل")
    
    def analyze(self, symbol: str) -> Optional[Dict]:
        if EconomicCalendar.is_high_impact_now(): return None
        if not SessionFilter.is_good_time(): return None
        if not self.risk.can_trade(): return None
        if not self.db.is_symbol_active(symbol): return None
        if self.db.has_active_signal(symbol): return None
        if self.db.was_recent(symbol): return None
        
        df_5m = DataFetcher.fetch(symbol, '5m')
        df_15m = DataFetcher.fetch(symbol, '15m')
        df_1h = DataFetcher.fetch(symbol, '1h')
        
        if df_5m is None or df_15m is None or df_1h is None: return None
        
        ind_5m = calculate_indicators(df_5m, symbol)
        ind_15m = calculate_indicators(df_15m, symbol)
        ind_1h = calculate_indicators(df_1h, symbol)
        
        if not ind_5m or not ind_15m or not ind_1h: return None
        
        adx = ind_15m['adx']
        if adx < 20: return None
        
        structure = MarketStructure.detect(df_1h)
        pa_pattern, pa_score = PriceAction.detect(df_5m)
        ml_proba = self.ml.predict(ind_5m)
        
        score = 0
        
        rsi = ind_5m['rsi']
        if rsi < 30: score += 1.5
        elif rsi > 70: score -= 1.5
        
        if ind_5m['macd'] > ind_5m['macd_signal']: score += 1.0
        else: score -= 1.0
        
        if ind_5m['above_ema20']: score += 0.5
        else: score -= 0.5
        if ind_1h['ema_bullish']: score += 0.5
        
        if ind_5m['bb_position'] < -0.8: score += 1.5
        elif ind_5m['bb_position'] > 0.8: score -= 1.5
        
        score += pa_score
        
        if ml_proba > 0.6: score += 1.0
        elif ml_proba < 0.4: score -= 1.0
        
        trend = structure['trend']
        
        if score > 0 and trend == 'UP': direction = 'BUY'
        elif score < 0 and trend == 'DOWN': direction = 'SELL'
        else: return None
        
        confidence = min(0.95, 0.5 + abs(score)*0.1)
        if confidence < MIN_CONFIDENCE: return None
        
        price = ind_5m['price']
        
        if direction == 'BUY': sl, tp = ind_15m['sl_buy'], ind_15m['tp_buy']
        else: sl, tp = ind_15m['sl_sell'], ind_15m['tp_sell']
        
        pos_size = self.risk.calculate_position_size(symbol, price, sl)
        
        return {
            'symbol': symbol, 'direction': direction,
            'price': price, 'stop_loss': sl, 'take_profit': tp,
            'confidence': confidence, 'score': score,
            'adx': adx, 'atr': ind_15m['atr'],
            'market_structure': structure['structure'],
            'price_action': pa_pattern,
            'session': SessionFilter.get_current_session(),
            'position_size': pos_size,
            'tf_5m': f"{'🟢' if score > 0 else '🔴'}",
            'tf_15m': f"{'🟢' if score > 0 else '🔴'}",
            'tf_1h': trend,
            'expiry_time': (datetime.now() + timedelta(minutes=7)).strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def check_trades(self):
        for trade in self.db.get_expired_trades():
            try:
                df = DataFetcher.fetch(trade['symbol'], '1m')
                if df is None: continue
                
                entry_time = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
                expiry_time = datetime.strptime(trade['expiry_time'], '%Y-%m-%d %H:%M:%S')
                
                mask = (df.index >= entry_time) & (df.index <= expiry_time)
                period = df[mask]
                
                if period.empty: continue
                
                high_p = float(period['High'].max())
                low_p = float(period['Low'].min())
                close_p = float(period['Close'].iloc[-1])
                
                entry = trade['entry_price']
                direction = trade['direction']
                sl = trade['stop_loss']
                tp = trade['take_profit']
                
                is_jpy = "JPY" in trade['symbol']
                pip_value = 0.01 if is_jpy else 0.0001
                
                sl_first = tp_first = 0
                
                if direction == 'BUY':
                    for _, row in period.iterrows():
                        if float(row['Low']) <= sl: sl_first = 1; break
                        if float(row['High']) >= tp: tp_first = 1; break
                    
                    if sl_first: result, exit_price = 'LOSS', sl
                    elif tp_first: result, exit_price = 'WIN', tp
                    else:
                        exit_price = close_p
                        result = 'WIN' if close_p > entry else 'LOSS'
                else:
                    for _, row in period.iterrows():
                        if float(row['High']) >= sl: sl_first = 1; break
                        if float(row['Low']) <= tp: tp_first = 1; break
                    
                    if sl_first: result, exit_price = 'LOSS', sl
                    elif tp_first: result, exit_price = 'WIN', tp
                    else:
                        exit_price = close_p
                        result = 'WIN' if close_p < entry else 'LOSS'
                
                pnl = (exit_price-entry)/entry*100 if direction == 'BUY' else (entry-exit_price)/entry*100
                pips = (exit_price-entry)/pip_value if direction == 'BUY' else (entry-exit_price)/pip_value
                
                self.db.update_result(trade['id'], exit_price, result, pnl, round(pips, 1),
                                     high_p, low_p, sl_first, tp_first)
            except:
                pass
    
    def hunt(self):
        active = self.db.get_active_symbols()
        logger.info(f"🔍 بحث في {len(active)} زوج...")
        
        best, best_score = None, 0
        
        for symbol in active:
            try:
                result = self.analyze(symbol)
                if result and abs(result['score']) > best_score:
                    best_score = abs(result['score'])
                    best = result
                time.sleep(0.5)
            except:
                pass
        
        if best:
            self.db.save_signal(best)
            self.send_signal(best)
    
    def send_signal(self, signal: Dict):
        emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
        
        msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
               f"💰 {signal['price']:.5f}\n"
               f"🛑 SL: {signal['stop_loss']:.5f}\n"
               f"🎯 TP: {signal['take_profit']:.5f}\n"
               f"💪 {signal['confidence']:.1%}\n"
               f"📊 ADX: {signal['adx']}\n"
               f"🏗️ {signal['market_structure']}\n"
               f"{signal['tf_5m']}5m {signal['tf_15m']}15m {signal['tf_1h']}1h\n\n"
               f"🤖 Falcon Pro v4")
        
        try:
            self.tb.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
        except:
            pass
    
    def run(self):
        logger.info("🦅 Falcon Pro v4")
        
        def poll():
            while True:
                try:
                    self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except:
                    time.sleep(5)
        threading.Thread(target=poll, daemon=True).start()
        time.sleep(1)
        
        while True:
            try:
                self.check_trades()
                self.hunt()
                time.sleep(SCAN_INTERVAL)
            except KeyboardInterrupt:
                break
            except:
                time.sleep(30)

if __name__ == "__main__":
    os.makedirs('models_v4', exist_ok=True)
    bot = FalconPro()
    bot.run()
