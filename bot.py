#!/usr/bin/env python3
"""
Falcon AI v5.3 - Simplified for Success
==========================================
✅ 2 simple strategies (Support/Resistance + EMA)
✅ 5-minute trades
✅ Low confidence threshold
✅ Fast scanning
"""

import os, sys, time, logging, sqlite3, hashlib, threading, json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np, pandas as pd, requests, warnings, telebot

warnings.filterwarnings('ignore')

TELEGRAM_TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
TELEGRAM_CHAT_ID = '7553333305'
SYMBOLS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X']
SCAN_INTERVAL = 10
MIN_CONFIDENCE = 0.50
COOLDOWN_MINUTES = 2

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('FalconV5')

class DataCache:
    def __init__(self): self.cache = {}; self.ttl = 30
    def get(self, key):
        if key in self.cache:
            d, t = self.cache[key]
            if time.time() - t < self.ttl: return d.copy()
        return None
    def set(self, key, data):
        self.cache[key] = (data, time.time())
        if len(self.cache) > 50: del self.cache[min(self.cache, key=lambda k: self.cache[k][1])]
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
            for s in ['divergence', 'ema_cross']: conn.execute('INSERT OR IGNORE INTO strategy_performance (strategy) VALUES (?)', (s,))
            conn.commit()
    
    def save_signal(self, data):
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''INSERT OR IGNORE INTO signals (symbol, direction, entry_price, stop_loss, take_profit, confidence, strategy, strategy_name, expiry_time, signal_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (data['symbol'], data['direction'], data['entry_price'], data.get('stop_loss'), data.get('take_profit'), data['confidence'], data.get('strategy', ''), data.get('strategy_name', ''), data['expiry_time'], h))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except: return None
    
    def update_result(self, sid, exit_price, result, pnl, pips):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE signals SET exit_price=?, result=?, pnl_percent=?, pnl_pips=?, exit_time=datetime('now','localtime') WHERE id=?", (exit_price, result, pnl, pips, sid))
            row = conn.execute('SELECT strategy FROM signals WHERE id=?', (sid,)).fetchone()
            if row and row[0]: conn.execute('UPDATE strategy_performance SET total_trades=total_trades+1, wins=wins+?, total_pnl=total_pnl+? WHERE strategy=?', (1 if result=='WIN' else 0, pnl, row[0]))
            conn.commit()
    
    def has_active_signal(self, symbol):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' AND expiry_time > datetime('now','localtime')", (symbol,)).fetchone()[0] > 0
    
    def was_recent(self, symbol, minutes=COOLDOWN_MINUTES):
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute('SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?', (symbol, cutoff)).fetchone()[0] > 0
    
    def get_expired_trades(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM signals WHERE result='PENDING' AND expiry_time <= datetime('now','localtime')").fetchall()]
    
    def get_strategy_weights(self):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute('SELECT strategy, wins, total_trades FROM strategy_performance').fetchall()
            return {r[0]: r[1]/r[2] if r[2]>5 else 0.5 for r in rows}

class DataFetcher:
    @staticmethod
    def fetch(symbol, interval='5min'):
        key = f"{symbol}_{interval}"
        cached = data_cache.get(key)
        if cached is not None: return cached
        try:
            import yfinance as yf
            imap = {'5m':'5m','15m':'15m','1h':'1h','1m':'1m'}
            df = yf.download(symbol, period='5d', interval=imap.get(interval,'5m'), progress=False)
            df = safe_columns(df)
            if not df.empty: data_cache.set(key, df); return df
        except: pass
        return None

def calc_rsi(df, period=14):
    c = df['Close'].values; delta = np.diff(c)
    gain = np.mean(delta[delta>0]) if any(delta>0) else 0
    loss = np.mean(-delta[delta<0]) if any(delta<0) else 0
    return round(100-100/(1+gain/(loss+1e-8)),1) if loss>0 else 50

def calc_atr(df, period=14):
    h,l,c = df['High'].values, df['Low'].values, df['Close'].values
    tr = np.array([max(h[i+1]-l[i+1], abs(h[i+1]-c[i]), abs(l[i+1]-c[i])) for i in range(len(c)-1)])
    return round(float(np.mean(tr[-period:])),5)

def calc_ema(df, period):
    return round(float(pd.Series(df['Close'].values).ewm(span=period, adjust=False).mean().values[-1]),5)

def find_sr(df, lookback=50):
    h = df['High'].values[-lookback:]; l = df['Low'].values[-lookback:]
    return float(np.mean(np.sort(l)[:5])), float(np.mean(np.sort(h)[-5:]))

def strat_support_resistance(df, symbol):
    """✅ ارتداد من دعم/مقاومة مع RSI"""
    if len(df) < 50: return None
    price = float(df['Close'].iloc[-1])
    support, resistance = find_sr(df)
    atr = calc_atr(df)
    rsi = calc_rsi(df)
    
    if rsi < 40 and (price - support) / support * 100 < 0.15:
        return {'direction':'BUY','price':price,'stop_loss':round(support,5),
                'take_profit':round(price+atr*2.0,5),'confidence':min(0.75, 0.5+(40-rsi)*0.015),
                'strategy':'divergence','strategy_name':'ارتداد من الدعم','score':3}
    
    if rsi > 60 and (resistance - price) / price * 100 < 0.15:
        return {'direction':'SELL','price':price,'stop_loss':round(resistance,5),
                'take_profit':round(price-atr*2.0,5),'confidence':min(0.75, 0.5+(rsi-60)*0.015),
                'strategy':'divergence','strategy_name':'ارتداد من المقاومة','score':3}
    return None

def strat_ema_trend(df, symbol):
    """✅ EMA ترند"""
    if len(df) < 60: return None
    ema20 = calc_ema(df, 20); ema50 = calc_ema(df, 50)
    price = float(df['Close'].iloc[-1])
    atr = calc_atr(df); rsi = calc_rsi(df)
    
    if price > ema20 and ema20 > ema50 and rsi < 60:
        return {'direction':'BUY','price':price,'stop_loss':round(ema50,5),
                'take_profit':round(price+atr*2.0,5),'confidence':0.65,
                'strategy':'ema_cross','strategy_name':'EMA ترند صاعد','score':3}
    
    if price < ema20 and ema20 < ema50 and rsi > 40:
        return {'direction':'SELL','price':price,'stop_loss':round(ema50,5),
                'take_profit':round(price-atr*2.0,5),'confidence':0.65,
                'strategy':'ema_cross','strategy_name':'EMA ترند هابط','score':3}
    return None

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
            w = self.db.get_strategy_weights()
            self.tb.reply_to(msg, f"🦅 **Falcon Pro v5.3**\n\n📊 ارتداد: {w.get('divergence',0):.1%}\n📊 EMA: {w.get('ema_cross',0):.1%}", parse_mode='Markdown')
    
    def analyze(self, symbol):
        results = []
        if self.db.has_active_signal(symbol): return results
        if self.db.was_recent(symbol): return results
        
        df_15m = DataFetcher.fetch(symbol, '15m')
        if df_15m is None: return results
        
        for strat_fn in [strat_support_resistance, strat_ema_trend]:
            s = strat_fn(df_15m, symbol)
            if s and s['confidence'] >= MIN_CONFIDENCE:
                s['symbol'] = symbol
                s['expiry_time'] = (datetime.now() + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')
                results.append(s)
                break
        
        return results[:1]
    
    def check_trades(self):
        for trade in self.db.get_expired_trades():
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
                
                self.db.update_result(trade['id'], close_p, result, pnl, round(pips,1))
            except: pass
    
    def hunt(self):
        signals = 0
        for symbol in SYMBOLS:
            try:
                results = self.analyze(symbol)
                for s in results:
                    if self.db.save_signal(s):
                        self.send_signal(s); signals += 1
                        time.sleep(1)
                time.sleep(0.3)
            except: pass
        if signals > 0: logger.info(f"📊 {signals} إشارة")
    
    def send_signal(self, signal):
        emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        direction = "شراء" if signal['direction'] == 'BUY' else "بيع"
        msg = f"{emoji} **{signal['symbol']}** - {direction}\n\n💰 {signal['price']:.5f}\n💪 {signal['confidence']:.1%}\n📊 {signal['strategy_name']}"
        try: self.tb.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
        except: pass
    
    def run(self):
        logger.info("🦅 Falcon Pro v5.3 - Simple & Effective")
        def poll():
            while True:
                try: self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except: time.sleep(5)
        threading.Thread(target=poll, daemon=True).start()
        time.sleep(1)
        
        while True:
            try:
                self.check_trades(); self.hunt()
                time.sleep(SCAN_INTERVAL)
            except KeyboardInterrupt: break
            except: time.sleep(10)

if __name__ == "__main__":
    bot = FalconPro(); bot.run()
