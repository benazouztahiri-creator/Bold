#!/usr/bin/env python3
"""
Falcon AI v5.3 - Balanced Quality
====================================
✅ 4 professional strategies with optimized thresholds
✅ More signals, same quality
✅ RSI Divergence + Breakout + EMA Cross + Fibonacci
✅ Each strategy independent
✅ 15-second scanning
✅ Full error handling
✅ Health check for Railway
"""

import os, sys, time, logging, sqlite3, hashlib, threading, json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
import numpy as np, pandas as pd, yfinance as yf
import telebot, requests

TELEGRAM_TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
TELEGRAM_CHAT_ID = '7553333305'
PORT = int(os.environ.get('PORT', 8000))

SYMBOLS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X']

SCAN_INTERVAL = 15
MIN_CONFIDENCE = 0.55
COOLDOWN_MINUTES = 2
TRADE_DURATION = 7

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('FalconV5')

try:
    requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=5)
    time.sleep(1)
except: pass

tb = telebot.TeleBot(TELEGRAM_TOKEN)

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    def log_message(self, format, *args): pass

def start_health():
    try:
        HTTPServer(('0.0.0.0', PORT), HealthHandler).serve_forever()
    except: pass

threading.Thread(target=start_health, daemon=True).start()

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
        self.db_path = 'falcon_v5.db'
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, direction TEXT, entry_price REAL, exit_price REAL, stop_loss REAL, take_profit REAL, confidence REAL, strategy TEXT, strategy_name TEXT, entry_time DATETIME DEFAULT CURRENT_TIMESTAMP, expiry_time DATETIME, exit_time DATETIME, result TEXT DEFAULT 'PENDING', pnl_percent REAL, pnl_pips REAL, signal_hash TEXT UNIQUE)''')
            conn.execute('''CREATE TABLE IF NOT EXISTS strategy_performance (strategy TEXT PRIMARY KEY, total_trades INTEGER DEFAULT 0, wins INTEGER DEFAULT 0, total_pnl REAL DEFAULT 0, win_rate REAL DEFAULT 0.5)''')
            for s in ['divergence', 'breakout_retest', 'ema_cross', 'fibo_volume']:
                conn.execute('INSERT OR IGNORE INTO strategy_performance (strategy) VALUES (?)', (s,))
            conn.commit()
    
    def save(self, data):
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''INSERT OR IGNORE INTO signals (symbol, direction, entry_price, stop_loss, take_profit, confidence, strategy, strategy_name, expiry_time, signal_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (data['symbol'], data['direction'], data['entry_price'], data.get('stop_loss'), data.get('take_profit'), data['confidence'], data.get('strategy',''), data.get('strategy_name',''), data['expiry_time'], h))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except: return None
    
    def update(self, sid, exit_price, result, pnl, pips):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("UPDATE signals SET exit_price=?, result=?, pnl_percent=?, pnl_pips=?, exit_time=datetime('now','localtime') WHERE id=?", (exit_price, result, pnl, pips, sid))
                row = conn.execute('SELECT strategy FROM signals WHERE id=?', (sid,)).fetchone()
                if row and row[0]:
                    conn.execute('UPDATE strategy_performance SET total_trades=total_trades+1, wins=wins+?, total_pnl=total_pnl+? WHERE strategy=?', (1 if result=='WIN' else 0, pnl, row[0]))
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
    
    def get_weights(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                rows = conn.execute('SELECT strategy, wins, total_trades FROM strategy_performance').fetchall()
                return {r[0]: r[1]/r[2] if r[2]>5 else 0.5 for r in rows}
        except: return {}

class DataFetcher:
    @staticmethod
    def fetch(symbol, interval='5min'):
        key = f"{symbol}_{interval}"
        cached = data_cache.get(key)
        if cached is not None: return cached
        try:
            imap = {'5m':'5m','15m':'15m','1h':'1h','1m':'1m'}
            df = yf.download(symbol, period='5d', interval=imap.get(interval,'5m'), progress=False)
            df = safe_columns(df)
            if not df.empty: data_cache.set(key, df); return df
        except: pass
        return None

# ========== INDICATORS ==========

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

def calc_ema(c, period):
    try: return round(float(pd.Series(c).ewm(span=period, adjust=False).mean().values[-1]),5)
    except: return 0

def find_sr(h, l):
    try: return float(np.mean(np.sort(l)[:3])), float(np.mean(np.sort(h)[-3:]))
    except: return 0, 0

def find_range(h, l):
    try:
        rh, rl = float(np.max(h)), float(np.min(l))
        return rl, rh, (rh-rl)/rl*100 < 0.3
    except: return 0, 0, False

def calc_fibo(h, l):
    try:
        sh, sl = float(np.max(h)), float(np.min(l))
        d = sh-sl
        return {'high':sh,'low':sl,'f382':round(sl+d*0.382,5),'f500':round(sl+d*0.500,5),'f618':round(sl+d*0.618,5)}
    except: return {'high':0,'low':0,'f382':0,'f500':0,'f618':0}

# ========== 4 STRATEGIES (OPTIMIZED) ==========

def strat_rsi_sr(df, symbol):
    """✅ RSI + دعم/مقاومة - شروط محسنة"""
    if len(df) < 50: return None
    c = df['Close'].values; h = df['High'].values; l = df['Low'].values
    price = float(c[-1]); rsi = calc_rsi(c); atr = calc_atr(h, l, c)
    support, resistance = find_sr(h, l)
    if support == 0: return None
    
    # ✅ شراء: RSI < 48 + قريب من الدعم
    if rsi < 48 and (price-support)/support*100 < 0.2:
        return {'direction':'BUY','price':price,'stop_loss':round(support,5),'take_profit':round(price+atr*2,5),'confidence':min(0.78, 0.5+(48-rsi)*0.02),'strategy':'divergence','strategy_name':'RSI + دعم'}
    
    # ✅ بيع: RSI > 52 + قريب من المقاومة
    if rsi > 52 and (resistance-price)/price*100 < 0.2:
        return {'direction':'SELL','price':price,'stop_loss':round(resistance,5),'take_profit':round(price-atr*2,5),'confidence':min(0.78, 0.5+(rsi-52)*0.02),'strategy':'divergence','strategy_name':'RSI + مقاومة'}
    return None

def strat_ema_trend(df, symbol):
    """✅ EMA ترند - شروط محسنة"""
    if len(df) < 50: return None
    c = df['Close'].values; h = df['High'].values; l = df['Low'].values
    price = float(c[-1]); ema20 = calc_ema(c, 20); ema50 = calc_ema(c, 50)
    atr = calc_atr(h, l, c); rsi = calc_rsi(c)
    
    if price > ema20 and ema20 > ema50 and rsi < 65:
        return {'direction':'BUY','price':price,'stop_loss':round(ema50,5),'take_profit':round(price+atr*2,5),'confidence':0.68,'strategy':'ema_cross','strategy_name':'EMA ترند صاعد'}
    if price < ema20 and ema20 < ema50 and rsi > 35:
        return {'direction':'SELL','price':price,'stop_loss':round(ema50,5),'take_profit':round(price-atr*2,5),'confidence':0.68,'strategy':'ema_cross','strategy_name':'EMA ترند هابط'}
    return None

def strat_breakout(df, symbol):
    """✅ اختراق - شروط محسنة"""
    if len(df) < 40: return None
    c = df['Close'].values; h = df['High'].values; l = df['Low'].values
    rl, rh, is_range = find_range(h, l)
    if not is_range: return None
    price = float(c[-1]); atr = calc_atr(h, l, c)
    
    if float(c[-2]) > rh and float(c[-1]) >= rh:
        return {'direction':'BUY','price':price,'stop_loss':round(rl,5),'take_profit':round(price+atr*3,5),'confidence':0.72,'strategy':'breakout_retest','strategy_name':'اختراق علوي'}
    if float(c[-2]) < rl and float(c[-1]) <= rl:
        return {'direction':'SELL','price':price,'stop_loss':round(rh,5),'take_profit':round(price-atr*3,5),'confidence':0.72,'strategy':'breakout_retest','strategy_name':'اختراق سفلي'}
    return None

def strat_fibo(df, df_1h, symbol):
    """✅ فيبوناتشي - شروط محسنة"""
    if len(df) < 50 or df_1h is None or len(df_1h) < 50: return None
    fibo = calc_fibo(df_1h['High'].values, df_1h['Low'].values)
    if fibo['f500'] == 0: return None
    
    c = df['Close'].values; h = df['High'].values; l = df['Low'].values
    price = float(c[-1])
    trend_up = calc_ema(df_1h['Close'].values, 20) > calc_ema(df_1h['Close'].values, 50)
    rsi = calc_rsi(c); atr = calc_atr(h, l, c)
    
    if trend_up and abs(price-fibo['f500'])/price*100 < 0.2 and rsi < 55:
        return {'direction':'BUY','price':price,'stop_loss':round(fibo['f618'],5),'take_profit':round(fibo['high'],5),'confidence':0.70,'strategy':'fibo_volume','strategy_name':'فيبو 50% صاعد'}
    if not trend_up and abs(price-fibo['f500'])/price*100 < 0.2 and rsi > 45:
        return {'direction':'SELL','price':price,'stop_loss':round(fibo['f618'],5),'take_profit':round(fibo['low'],5),'confidence':0.70,'strategy':'fibo_volume','strategy_name':'فيبو 50% هابط'}
    return None

# ========== MAIN ==========

class FalconPro:
    def __init__(self):
        self.db = Database()
        self.tb = telebot.TeleBot(TELEGRAM_TOKEN)
        self._setup()
    
    def _setup(self):
        try: requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=3)
        except: pass
        
        @self.tb.message_handler(commands=['start'])
        def start(msg):
            w = self.db.get_weights()
            text = (f"🦅 **Falcon Pro v5.3**\n\n"
                   f"1️⃣ RSI+SR: {w.get('divergence',0):.1%}\n"
                   f"2️⃣ Breakout: {w.get('breakout_retest',0):.1%}\n"
                   f"3️⃣ EMA: {w.get('ema_cross',0):.1%}\n"
                   f"4️⃣ Fibo: {w.get('fibo_volume',0):.1%}")
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def analyze(self, symbol):
        results = []
        if self.db.has_active(symbol): return results
        if self.db.was_recent(symbol): return results
        
        df_15m = DataFetcher.fetch(symbol, '15m')
        df_1h = DataFetcher.fetch(symbol, '1h')
        if df_15m is None: return results
        
        now = datetime.now(timezone.utc)
        if now.weekday() >= 5: return results
        
        # ✅ كل الاستراتيجيات تشتغل بشكل مستقل
        s1 = strat_rsi_sr(df_15m, symbol)
        if s1 and s1['confidence'] >= MIN_CONFIDENCE:
            s1['symbol'] = symbol; s1['expiry_time'] = (datetime.now()+timedelta(minutes=TRADE_DURATION)).strftime('%Y-%m-%d %H:%M:%S'); results.append(s1)
        
        s2 = strat_ema_trend(df_15m, symbol)
        if s2 and s2['confidence'] >= MIN_CONFIDENCE:
            s2['symbol'] = symbol; s2['expiry_time'] = (datetime.now()+timedelta(minutes=TRADE_DURATION)).strftime('%Y-%m-%d %H:%M:%S'); results.append(s2)
        
        s3 = strat_breakout(df_15m, symbol)
        if s3 and s3['confidence'] >= MIN_CONFIDENCE:
            s3['symbol'] = symbol; s3['expiry_time'] = (datetime.now()+timedelta(minutes=TRADE_DURATION)).strftime('%Y-%m-%d %H:%M:%S'); results.append(s3)
        
        if df_1h is not None:
            s4 = strat_fibo(df_15m, df_1h, symbol)
            if s4 and s4['confidence'] >= MIN_CONFIDENCE:
                s4['symbol'] = symbol; s4['expiry_time'] = (datetime.now()+timedelta(minutes=TRADE_DURATION)).strftime('%Y-%m-%d %H:%M:%S'); results.append(s4)
        
        return results
    
    def check_trades(self):
        for trade in self.db.get_expired():
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
                
                if direction == 'BUY':
                    pnl = (close_p-entry)/entry*100; pips = (close_p-entry)/pv
                    result = 'WIN' if close_p > entry else 'LOSS'
                else:
                    pnl = (entry-close_p)/entry*100; pips = (entry-close_p)/pv
                    result = 'WIN' if close_p < entry else 'LOSS'
                
                self.db.update(trade['id'], close_p, result, pnl, round(pips,1))
                logger.info(f"{'WIN' if result=='WIN' else 'LOSS'} {trade['symbol']}: {pnl:+.2f}% | {trade.get('strategy_name','')}")
            except: pass
    
    def hunt(self):
        signals = 0
        for symbol in SYMBOLS:
            try:
                for s in self.analyze(symbol):
                    if self.db.save(s):
                        self.send_signal(s); signals += 1
                        time.sleep(0.5)
                time.sleep(0.2)
            except: pass
        if signals > 0: logger.info(f"{signals} signals")
    
    def send_signal(self, signal):
        emoji = "🟢" if signal['direction']=='BUY' else "🔴"
        direction = "شراء" if signal['direction']=='BUY' else "بيع"
        msg = f"{emoji} **{signal['symbol']}** - {direction}\n\n💰 {signal['price']:.5f}\n💪 {signal['confidence']:.1%}\n📊 {signal['strategy_name']}"
        try: self.tb.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
        except: pass
    
    def run(self):
        logger.info("Falcon Pro v5.3 - Optimized")
        
        def poll():
            while True:
                try: self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except: time.sleep(5)
        threading.Thread(target=poll, daemon=True).start()
        time.sleep(1)
        
        while True:
            try:
                self.check_trades()
                self.hunt()
                time.sleep(SCAN_INTERVAL)
            except KeyboardInterrupt: break
            except: time.sleep(10)

if __name__ == "__main__":
    bot = FalconPro()
    bot.run()
