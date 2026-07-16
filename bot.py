#!/usr/bin/env python3
"""
Falcon AI v6.2 - Precision Institutional Strategy
===================================================
✅ Tight liquidity proximity (pips, not percentage)
✅ JPY pairs use appropriate pip values
✅ Proper Liquidity Sweep detection
✅ Optimized RSI thresholds (45/55)
✅ 15-second scanning
✅ 5-minute cooldown
✅ No polling - No 409 error
✅ Health check for Railway
"""

import os, sys, time, logging, sqlite3, hashlib, threading
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np, pandas as pd, yfinance as yf
import telebot, requests

TELEGRAM_TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
TELEGRAM_CHAT_ID = '7553333305'
PORT = int(os.environ.get('PORT', 8000))

SYMBOLS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X']

SCAN_INTERVAL = 15
MIN_CONFIDENCE = 0.60
COOLDOWN_MINUTES = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('FalconV6')

try:
    requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=5)
    time.sleep(1)
except: pass

tb = telebot.TeleBot(TELEGRAM_TOKEN)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"OK")
    def log_message(self, format, *args): pass

threading.Thread(target=lambda: HTTPServer(('0.0.0.0', PORT), HealthHandler).serve_forever(), daemon=True).start()

class DataCache:
    def __init__(self): self.cache = {}; self.ttl = 30
    def get(self, key):
        if key in self.cache:
            d, t = self.cache[key]
            if time.time() - t < self.ttl: return d.copy()
        return None
    def set(self, key, data): self.cache[key] = (data, time.time())
data_cache = DataCache()

def safe_columns(df):
    if df is None or df.empty: return df
    try:
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).capitalize() for c in df.columns]
    except: pass
    return df

class Database:
    def __init__(self):
        self.db_path = 'falcon_v6.db'
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, direction TEXT, entry_price REAL, exit_price REAL, stop_loss REAL, take_profit REAL, confidence REAL, strategy TEXT, entry_time DATETIME DEFAULT CURRENT_TIMESTAMP, expiry_time DATETIME, exit_time DATETIME, result TEXT DEFAULT 'PENDING', pnl_percent REAL, pnl_pips REAL, signal_hash TEXT UNIQUE)''')
            conn.commit()
    
    def save(self, data):
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''INSERT OR IGNORE INTO signals (symbol, direction, entry_price, stop_loss, take_profit, confidence, strategy, expiry_time, signal_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (data['symbol'], data['direction'], data['entry_price'], data.get('stop_loss'), data.get('take_profit'), data['confidence'], data.get('strategy','institutional_v2'), data['expiry_time'], h))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except: return None
    
    def update(self, sid, exit_price, result, pnl, pips):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("UPDATE signals SET exit_price=?, result=?, pnl_percent=?, pnl_pips=?, exit_time=datetime('now','localtime') WHERE id=?", (exit_price, result, pnl, pips, sid))
                conn.commit()
        except: pass
    
    def has_active(self, symbol):
        try:
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute("SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' AND expiry_time > datetime('now','localtime')", (symbol,)).fetchone()[0] > 0
        except: return False
    
    def was_recent(self, symbol, minutes=COOLDOWN_MINUTES):
        try:
            cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute('SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?', (symbol, cutoff)).fetchone()[0] > 0
        except: return False
    
    def get_expired(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute("SELECT * FROM signals WHERE result='PENDING' AND expiry_time <= datetime('now','localtime')").fetchall()]
        except: return []

class DataFetcher:
    @staticmethod
    def fetch(symbol, interval='15m'):
        key = f"{symbol}_{interval}"
        cached = data_cache.get(key)
        if cached is not None: return cached
        try:
            imap = {'5m':'5m','15m':'15m','1h':'1h','1m':'1m'}
            df = yf.download(symbol, period='5d', interval=imap.get(interval,'15m'), progress=False)
            df = safe_columns(df)
            if not df.empty: data_cache.set(key, df); return df
        except: pass
        return None

def calc_rsi(c):
    try:
        delta = np.diff(c)
        gain = np.mean(delta[delta>0]) if any(delta>0) else 0
        loss = np.mean(-delta[delta<0]) if any(delta<0) else 0
        return round(100-100/(1+gain/(loss+1e-8)),1) if loss>0 else 50
    except: return 50

def calc_atr(h, l, c):
    try:
        tr = np.array([max(h[i+1]-l[i+1], abs(h[i+1]-c[i]), abs(l[i+1]-c[i])) for i in range(len(c)-1)])
        return round(float(np.mean(tr[-14:])),5)
    except: return 0.0001

def get_pip_value(symbol):
    """✅ JPY = 0.01, غير JPY = 0.0001"""
    return 0.01 if 'JPY' in symbol else 0.0001

def is_near_level(price, level, symbol, max_pips=5):
    """
    ✅ قياس القرب من المستوى بالنقاط مش النسبة المئوية
    EURUSD: 5 نقاط = 0.0005
    USDJPY: 5 نقاط = 0.05
    """
    pip = get_pip_value(symbol)
    distance = abs(price - level)
    return distance <= (max_pips * pip)

def strat_institutional_v2(df_15m, df_1h, symbol):
    """✅ Falcon Institutional v2 - Precision Entry"""
    if len(df_15m) < 30 or len(df_1h) < 30: return None
    
    c = df_15m['Close'].values; h = df_15m['High'].values; l = df_15m['Low'].values
    price = float(c[-1]); atr = calc_atr(h, l, c); rsi = calc_rsi(c)
    
    # ✅ مستويات السيولة على 1h
    high_liq = float(np.max(df_1h['High'].values[-24:]))
    low_liq = float(np.min(df_1h['Low'].values[-24:]))
    
    # ✅ شراء: السعر قريب من القاع (بـ 5 نقاط) + ارتد + RSI < 48
    if is_near_level(float(l[-1]), low_liq, symbol, max_pips=5) and price > low_liq and rsi < 48:
        sl = round(float(np.min(l[-3:])) - atr * 0.5, 5)
        tp = round(price + atr * 3.0, 5)
        conf = min(0.88, 0.55 + (48 - rsi) * 0.02)
        return {'direction':'BUY','price':price,'stop_loss':sl,'take_profit':tp,'confidence':conf,'strategy':'institutional_v2','strategy_name':'Inst. Buy (Precision Sweep)'}
    
    # ✅ بيع: السعر قريب من القمة (بـ 5 نقاط) + ارتد + RSI > 52
    if is_near_level(float(h[-1]), high_liq, symbol, max_pips=5) and price < high_liq and rsi > 52:
        sl = round(float(np.max(h[-3:])) + atr * 0.5, 5)
        tp = round(price - atr * 3.0, 5)
        conf = min(0.88, 0.55 + (rsi - 52) * 0.02)
        return {'direction':'SELL','price':price,'stop_loss':sl,'take_profit':tp,'confidence':conf,'strategy':'institutional_v2','strategy_name':'Inst. Sell (Precision Sweep)'}
    
    return None

def send_message(text):
    try: tb.send_message(TELEGRAM_CHAT_ID, text)
    except: pass

def main():
    db = Database()
    logger.info("Falcon Institutional v6.2 - Precision")
    time.sleep(3)
    try: send_message("Falcon Institutional v6.2\nPrecision Mode - Ready")
    except: pass
    
    while True:
        try:
            for trade in db.get_expired():
                try:
                    df = DataFetcher.fetch(trade['symbol'], '1m')
                    if df is None: continue
                    et = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
                    xt = datetime.strptime(trade['expiry_time'], '%Y-%m-%d %H:%M:%S')
                    period = df[(df.index >= et) & (df.index <= xt)]
                    if period.empty: continue
                    close_p = float(period['Close'].iloc[-1])
                    entry = trade['entry_price']; direction = trade['direction']
                    is_jpy = "JPY" in trade['symbol']; pv = 0.01 if is_jpy else 0.0001
                    pnl = (close_p-entry)/entry*100 if direction=='BUY' else (entry-close_p)/entry*100
                    pips = (close_p-entry)/pv if direction=='BUY' else (entry-close_p)/pv
                    result = 'WIN' if pnl > 0 else 'LOSS'
                    db.update(trade['id'], close_p, result, pnl, round(pips,1))
                    logger.info(f"{'WIN' if result=='WIN' else 'LOSS'} {trade['symbol']}: {pnl:+.2f}%")
                except: pass
            
            now = datetime.now(timezone.utc)
            if now.weekday() < 5:
                for symbol in SYMBOLS:
                    try:
                        if db.has_active(symbol): continue
                        if db.was_recent(symbol): continue
                        
                        df_15m = DataFetcher.fetch(symbol, '15m')
                        df_1h = DataFetcher.fetch(symbol, '1h')
                        if df_15m is None or df_1h is None: continue
                        
                        s = strat_institutional_v2(df_15m, df_1h, symbol)
                        if s and s['confidence'] >= MIN_CONFIDENCE:
                            s['symbol'] = symbol
                            s['expiry_time'] = (datetime.now() + timedelta(minutes=20)).strftime('%Y-%m-%d %H:%M:%S')
                            if db.save(s):
                                emoji = "BUY" if s['direction']=='BUY' else "SELL"
                                direction = "Buy" if s['direction']=='BUY' else "Sell"
                                msg = f"{emoji} {symbol} - {direction}\n{s['price']:.5f}\n{s['strategy_name']}\n{s['confidence']:.1%}"
                                send_message(msg)
                                logger.info(f"SIGNAL: {symbol} {s['direction']} | {s['strategy_name']}")
                        time.sleep(1)
                    except: pass
            
            time.sleep(SCAN_INTERVAL)
            
        except KeyboardInterrupt: break
        except Exception as e:
            logger.error(f"Loop: {e}")
            time.sleep(10)

if __name__ == "__main__":
    while True:
        try: main()
        except KeyboardInterrupt: break
        except: time.sleep(30)
