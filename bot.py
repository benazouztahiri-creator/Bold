#!/usr/bin/env python3
"""
Falcon AI v6.2 - Railway Optimized
====================================
✅ 2 symbols only (EURUSD, USDJPY)
✅ 60-second scan interval
✅ Minimal memory usage
✅ Precision liquidity detection
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

SYMBOLS = ['EURUSD', 'USDJPY']
SCAN_INTERVAL = 60
MIN_CONFIDENCE = 0.60
COOLDOWN_MINUTES = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('Falcon')

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

class Database:
    def __init__(self):
        self.db_path = 'falcon_v6.db'
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, direction TEXT, entry_price REAL, exit_price REAL, stop_loss REAL, take_profit REAL, confidence REAL, strategy TEXT, entry_time DATETIME DEFAULT CURRENT_TIMESTAMP, expiry_time DATETIME, exit_time DATETIME, result TEXT DEFAULT 'PENDING', pnl_percent REAL, pnl_pips REAL, signal_hash TEXT UNIQUE)''')
            conn.commit()
    
    def save(self, d):
        try:
            h = hashlib.md5(f"{d['symbol']}_{d['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('INSERT OR IGNORE INTO signals (symbol, direction, entry_price, stop_loss, take_profit, confidence, strategy, expiry_time, signal_hash) VALUES (?,?,?,?,?,?,?,?,?)',
                           (d['symbol'], d['direction'], d['entry_price'], d.get('stop_loss'), d.get('take_profit'), d['confidence'], d.get('strategy','v2'), d['expiry_time'], h))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except: return None
    
    def update(self, sid, ep, r, pnl, pips):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("UPDATE signals SET exit_price=?, result=?, pnl_percent=?, pnl_pips=?, exit_time=datetime('now','localtime') WHERE id=?", (ep, r, pnl, pips, sid))
                conn.commit()
        except: pass
    
    def has_active(self, s):
        try:
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute("SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' AND expiry_time > datetime('now','localtime')", (s,)).fetchone()[0] > 0
        except: return False
    
    def was_recent(self, s, m=COOLDOWN_MINUTES):
        try:
            c = (datetime.now() - timedelta(minutes=m)).strftime('%Y-%m-%d %H:%M:%S')
            with sqlite3.connect(self.db_path) as conn:
                return conn.execute('SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?', (s, c)).fetchone()[0] > 0
        except: return False
    
    def get_expired(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                return [dict(r) for r in conn.execute("SELECT * FROM signals WHERE result='PENDING' AND expiry_time <= datetime('now','localtime')").fetchall()]
        except: return []

def get_data(symbol, interval='15m'):
    try:
        imap = {'15m':'15m','1h':'1h','1m':'1m'}
        df = yf.download(symbol, period='5d', interval=imap.get(interval,'15m'), progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower() for c in df.columns]
            return df
    except: pass
    return None

def rsi(c):
    try:
        d = np.diff(c)
        g = np.mean(d[d>0]) if any(d>0) else 0
        l = np.mean(-d[d<0]) if any(d<0) else 0
        return round(100-100/(1+g/(l+1e-8)),1) if l>0 else 50
    except: return 50

def atr(h, l, c):
    try:
        tr = np.array([max(h[i+1]-l[i+1], abs(h[i+1]-c[i]), abs(l[i+1]-c[i])) for i in range(len(c)-1)])
        return round(float(np.mean(tr[-14:])),5)
    except: return 0.0001

def near(price, level, symbol, max_pips=5):
    pip = 0.01 if 'JPY' in symbol else 0.0001
    return abs(price - level) <= (max_pips * pip)

def analyze(df_15m, df_1h, symbol):
    if df_15m is None or df_1h is None: return None
    if len(df_15m) < 30 or len(df_1h) < 30: return None
    
    c = df_15m['close'].values; h = df_15m['high'].values; l = df_15m['low'].values
    price = float(c[-1]); atr_val = atr(h, l, c); rsi_val = rsi(c)
    
    high_liq = float(np.max(df_1h['high'].values[-24:]))
    low_liq = float(np.min(df_1h['low'].values[-24:]))
    
    if near(float(l[-1]), low_liq, symbol, 5) and price > low_liq and rsi_val < 48:
        sl = round(float(np.min(l[-3:])) - atr_val * 0.5, 5)
        tp = round(price + atr_val * 3.0, 5)
        conf = min(0.88, 0.55 + (48 - rsi_val) * 0.02)
        return {'direction':'BUY','price':price,'stop_loss':sl,'take_profit':tp,'confidence':conf,'strategy':'v2','strategy_name':'Inst. Buy'}
    
    if near(float(h[-1]), high_liq, symbol, 5) and price < high_liq and rsi_val > 52:
        sl = round(float(np.max(h[-3:])) + atr_val * 0.5, 5)
        tp = round(price - atr_val * 3.0, 5)
        conf = min(0.88, 0.55 + (rsi_val - 52) * 0.02)
        return {'direction':'SELL','price':price,'stop_loss':sl,'take_profit':tp,'confidence':conf,'strategy':'v2','strategy_name':'Inst. Sell'}
    
    return None

def send(text):
    try: tb.send_message(TELEGRAM_CHAT_ID, text)
    except: pass

def main():
    db = Database()
    logger.info("Falcon v6.2 - Railway Optimized")
    
    while True:
        try:
            for t in db.get_expired():
                try:
                    df = get_data(t['symbol'], '1m')
                    if df is None: continue
                    et = datetime.strptime(t['entry_time'], '%Y-%m-%d %H:%M:%S')
                    xt = datetime.strptime(t['expiry_time'], '%Y-%m-%d %H:%M:%S')
                    period = df[(df.index >= et) & (df.index <= xt)]
                    if period.empty: continue
                    cp = float(period['close'].iloc[-1])
                    entry = t['entry_price']; direction = t['direction']
                    pv = 0.01 if 'JPY' in t['symbol'] else 0.0001
                    pnl = (cp-entry)/entry*100 if direction=='BUY' else (entry-cp)/entry*100
                    pips = (cp-entry)/pv if direction=='BUY' else (entry-cp)/pv
                    result = 'WIN' if pnl > 0 else 'LOSS'
                    db.update(t['id'], cp, result, pnl, round(pips,1))
                except: pass
            
            now = datetime.now(timezone.utc)
            if now.weekday() < 5:
                for symbol in SYMBOLS:
                    try:
                        if db.has_active(symbol): continue
                        if db.was_recent(symbol): continue
                        
                        df_15m = get_data(symbol, '15m')
                        df_1h = get_data(symbol, '1h')
                        if df_15m is None or df_1h is None: continue
                        
                        s = analyze(df_15m, df_1h, symbol)
                        if s and s['confidence'] >= MIN_CONFIDENCE:
                            s['symbol'] = symbol
                            s['expiry_time'] = (datetime.now() + timedelta(minutes=20)).strftime('%Y-%m-%d %H:%M:%S')
                            if db.save(s):
                                emoji = "BUY" if s['direction']=='BUY' else "SELL"
                                direction = "Buy" if s['direction']=='BUY' else "Sell"
                                send(f"{emoji} {symbol} - {direction}\n{s['price']:.5f}\n{s['strategy_name']}\n{s['confidence']:.1%}")
                                logger.info(f"SIGNAL: {symbol} {s['direction']}")
                        time.sleep(2)
                    except: pass
            
            time.sleep(SCAN_INTERVAL)
            
        except KeyboardInterrupt: break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(30)

if __name__ == "__main__":
    while True:
        try: main()
        except KeyboardInterrupt: break
        except: time.sleep(30)
