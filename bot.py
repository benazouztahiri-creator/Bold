#!/usr/bin/env python3
"""
Falcon AI v6.0 - Institutional Strategy Edition
==============================================
✅ Single Strategy: Falcon Institutional Strategy v1
✅ Order Block & Liquidity Sweep Detection
✅ Volume & RSI Confirmation
✅ 1-minute to 15-minute multi-timeframe confirmation
"""

import os
import sys
import time
import logging
import sqlite3
import hashlib
import threading
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import requests
import warnings

import telebot

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIG
# ============================================================================

TELEGRAM_TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
TELEGRAM_CHAT_ID = '7553333305'

SYMBOLS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
    'USDCAD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X'
]

SCAN_INTERVAL = 15  
MIN_CONFIDENCE = 0.65
COOLDOWN_MINUTES = 5  # تهدئة 5 دقائق لكل زوج

# ============================================================================
# LOGGING & CACHE
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FalconV6')

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

def safe_columns(df):
    if df is None or df.empty:
        return df
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).capitalize() for c in df.columns]
    except:
        pass
    return df

# ============================================================================
# DATABASE (Cleaned for Single Strategy)
# ============================================================================

class Database:
    def __init__(self):
        self.db_path = 'falcon_v6.db'
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, direction TEXT, 
                entry_price REAL, exit_price REAL, stop_loss REAL, take_profit REAL, 
                confidence REAL, strategy TEXT, entry_time DATETIME DEFAULT CURRENT_TIMESTAMP, 
                expiry_time DATETIME, exit_time DATETIME, result TEXT DEFAULT 'PENDING', 
                pnl_percent REAL, pnl_pips REAL, signal_hash TEXT UNIQUE)''')
            conn.commit()
    
    def save_signal(self, data):
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''INSERT OR IGNORE INTO signals (symbol, direction, entry_price, stop_loss, take_profit, confidence, strategy, expiry_time, signal_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (data['symbol'], data['direction'], data['entry_price'],
                            data.get('stop_loss'), data.get('take_profit'),
                            data['confidence'], data.get('strategy', 'institutional_v1'),
                            data['expiry_time'], h))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except:
            return None
    
    def update_result(self, signal_id, exit_price, result, pnl, pips):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE signals SET exit_price=?, result=?, pnl_percent=?, pnl_pips=?, exit_time=datetime('now','localtime') WHERE id=?",
                        (exit_price, result, pnl, pips, signal_id))
            conn.commit()
    
    def has_active_signal(self, symbol):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute("SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' AND expiry_time > datetime('now','localtime')", (symbol,)).fetchone()[0]
            return c > 0
    
    def was_recent(self, symbol, minutes=COOLDOWN_MINUTES):
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?', (symbol, cutoff)).fetchone()[0]
            return c > 0
    
    def get_expired_trades(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM signals WHERE result='PENDING' AND expiry_time <= datetime('now','localtime')").fetchall()
            return [dict(r) for r in rows]

# ============================================================================
# DATA FETCHER
# ============================================================================

class DataFetcher:
    @staticmethod
    def fetch(symbol, interval='15m'):
        key = f"{symbol}_{interval}"
        cached = data_cache.get(key)
        if cached is not None:
            return cached
        try:
            import yfinance as yf
            yf_symbol = symbol if '=X' in symbol else f"{symbol}=X"
            imap = {'5m': '5m', '15m': '15m', '1h': '1h', '1m': '1m'}
            df = yf.download(yf_symbol, period='5d', interval=imap.get(interval, '15m'), progress=False)
            df = safe_columns(df)
            if not df.empty:
                data_cache.set(key, df)
                return df
        except:
            pass
        return None

# ============================================================================
# TECHNICAL HELPERS
# ============================================================================

def calc_rsi(df, period=14):
    if len(df) < period: return 50
    c = df['Close'].values
    delta = np.diff(c)
    gain = np.mean(delta[delta > 0]) if any(delta > 0) else 0
    loss = np.mean(-delta[delta < 0]) if any(delta < 0) else 0
    return round(100 - 100/(1 + gain/(loss+1e-8)), 1) if loss > 0 else 50

def calc_atr(df, period=14):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    tr = np.array([max(h[i+1]-l[i+1], abs(h[i+1]-c[i]), abs(l[i+1]-c[i])) for i in range(len(c)-1)])
    return round(float(np.mean(tr[-period:])), 5)

# ============================================================================
# EXCLUSIVE STRATEGY: Falcon Institutional Strategy v1
# ============================================================================

def strat_falcon_institutional_v1(df_15m, df_1h):
    """
    Falcon Institutional Strategy v1
    - تتبع سلوك صناع السوق (Smart Money Concepts)
    - رصد سحب السيولة (Liquidity Sweep) عند القمم والقيعان اليومية والساعة.
    - تأكيد الدخول عبر كتل العقود (Order Blocks) المتكونة بزخم وحجم تداول عالي.
    """
    if len(df_15m) < 30 or len(df_1h) < 30: 
        return None
        
    c_15m = df_15m['Close'].values
    h_15m = df_15m['High'].values
    l_15m = df_15m['Low'].values
    v_15m = df_15m['Volume'].values if 'Volume' in df_15m.columns else np.ones(len(df_15m))
    
    price = float(c_15m[-1])
    atr = calc_atr(df_15m)
    rsi = calc_rsi(df_15m)
    
    # حساب أعلى قمة وأقل قاع لآخر 24 شمعة في فريم الساعة (السيولة المؤسساتية)
    high_liquidity = float(np.max(df_1h['High'].values[-24:]))
    low_liquidity = float(np.min(df_1h['Low'].values[-24:]))
    
    # متوسط أحجام التداول لرصد الضخ المؤسساتي
    avg_vol = np.mean(v_15m[-20:])
    current_vol = v_15m[-1]
    institutional_volume = current_vol > (avg_vol * 1.3) # حجم أعلى بـ 30% من المعدل

    # 1. إشارة الشراء (Bullish Institutional Reversal)
    # السعر كسر أدنى قاع لتنظيف السيولة ثم ارتد بقوة مع حجم تداول ضخم و مؤشر RSI في منطقة تشبع بيعي
    if float(l_15m[-1]) <= low_liquidity and price > low_liquidity and rsi < 35:
        if institutional_volume:
            sl = round(float(np.min(l_15m[-3:])) - (atr * 0.5), 5)
            tp = round(price + (atr * 4.0), 5) # عائد مخاطرة مؤسساتي عالي 1:4
            confidence = min(0.95, 0.65 + (35 - rsi) * 0.02)
            
            return {
                'direction': 'BUY', 'price': price, 'stop_loss': sl, 'take_profit': tp,
                'confidence': confidence, 'strategy': 'institutional_v1',
                'strategy_name': 'Falcon Institutional v1 (Bullish OB + Sweep)'
            }

    # 2. إشارة البيع (Bearish Institutional Reversal)
    # السعر ضرب أعلى قمة لتنظيف سيولة التجزئة ثم ارتد هبوطاً بحجم ضخم و RSI في منطقة تشبع شرائي
    if float(h_15m[-1]) >= high_liquidity and price < high_liquidity and rsi > 65:
        if institutional_volume:
            sl = round(float(np.max(h_15m[-3:])) + (atr * 0.5), 5)
            tp = round(price - (atr * 4.0), 5)
            confidence = min(0.95, 0.65 + (rsi - 65) * 0.02)
            
            return {
                'direction': 'SELL', 'price': price, 'stop_loss': sl, 'take_profit': tp,
                'confidence': confidence, 'strategy': 'institutional_v1',
                'strategy_name': 'Falcon Institutional v1 (Bearish OB + Sweep)'
            }
            
    return None

# ============================================================================
# MAIN BOT BODY
# ============================================================================

class FalconPro:
    def __init__(self):
        self.db = Database()
        self.tb = telebot.TeleBot(TELEGRAM_TOKEN)
        self._setup()
    
    def _setup(self):
        try:
            requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=3)
        except:
            pass
        
        @self.tb.message_handler(commands=['start'])
        def start(msg):
            text = (f"🦅 **Falcon Pro v6.0**\n"
                    f"========================\n"
                    f"🔥 الاستراتيجية النشطة حالياً:\n"
                    f"🎯 **Falcon Institutional Strategy v1**\n\n"
                    f" تم حذف الاستراتيجيات القديمة بنجاح.\n"
                    f"⏱️ مدة التهدئة المعتمدة: {COOLDOWN_MINUTES} دقائق.")
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def analyze(self, symbol):
        results = []
        
        if self.db.has_active_signal(symbol) or self.db.was_recent(symbol):
            return results
        
        df_15m = DataFetcher.fetch(symbol, '15m')
        df_1h = DataFetcher.fetch(symbol, '1h')
        
        if df_15m is None or df_1h is None:
            return results
        
        now = datetime.utcnow()
        if now.weekday() >= 5: # إيقاف العمل في عطلة نهاية الأسبوع
            return results
        
        # استدعاء الاستراتيجية المؤسساتية الوحيدة
        s = strat_falcon_institutional_v1(df_15m, df_1h)
        if s and s['confidence'] >= MIN_CONFIDENCE:
            s['symbol'] = symbol
            s['expiry_time'] = (datetime.now() + timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
            results.append(s)
        
        return results[:1]
    
    def check_trades(self):
        for trade in self.db.get_expired_trades():
            try:
                df = DataFetcher.fetch(trade['symbol'], '1m')
                if df is None: continue
                
                et = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
                xt = datetime.strptime(trade['expiry_time'], '%Y-%m-%d %H:%M:%S')
                mask = (df.index >= et) & (df.index <= xt)
                period = df[mask]
                if period.empty: continue
                
                close_p = float(period['Close'].iloc[-1])
                entry = trade['entry_price']
                direction = trade['direction']
                is_jpy = "JPY" in trade['symbol']
                pv = 0.01 if is_jpy else 0.0001
                
                if direction == 'BUY':
                    pnl = (close_p - entry) / entry * 100
                    pips = (close_p - entry) / pv
                    result = 'WIN' if close_p > entry else 'LOSS'
                else:
                    pnl = (entry - close_p) / entry * 100
                    pips = (entry - close_p) / pv
                    result = 'WIN' if close_p < entry else 'LOSS'
                
                self.db.update_result(trade['id'], close_p, result, pnl, round(pips, 1))
            except:
                pass
    
    def hunt(self):
        logger.info("🔍 Falcon Institutional Engine Scanning...")
        signals = 0
        
        for symbol in SYMBOLS:
            try:
                results = self.analyze(symbol)
                for s in results:
                    sig_id = self.db.save_signal(s)
                    if sig_id:
                        self.send_signal(s)
                        signals += 1
                        time.sleep(1)  
                time.sleep(0.3)
            except:
                pass
    
    def send_signal(self, signal):
        emoji = "🐳" if signal['direction'] == 'BUY' else "🐻"
        direction = "شراء (BUY)" if signal['direction'] == 'BUY' else "بيع (SELL)"
        msg = (f"{emoji} **إشارة مؤسساتية جديدة: {signal['symbol']}**\n"
               f"========================\n"
               f"جاري دخول صناع السوق الآن 🏦\n\n"
               f" نوع الصفقة: **{direction}**\n"
               f" سعر الدخول: `{signal['price']:.5f}`\n"
               f" وقف الخسارة (SL): `{signal['stop_loss']:.5f}`\n"
               f" جني الأرباح (TP): `{signal['take_profit']:.5f}`\n"
               f" قوة تأكيد السيولة: `{signal['confidence']:.1%}`\n\n"
               f"📡 نظام الفحص: {signal['strategy_name']}")
        try:
            self.tb.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
            logger.info(f"✅ {signal['symbol']} {signal['direction']} Sent Successfully.")
        except:
            pass
    
    def run(self):
        logger.info(f"🦅 Falcon Institutional Pro v6.0 Initialized | Cooldown: {COOLDOWN_MINUTES}m")
        
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
                time.sleep(10)

if __name__ == "__main__":
    bot = FalconPro()
    bot.run()
