#!/usr/bin/env python3
"""
Falcon Institutional Strategy - Clean
=======================================
✅ SMC Strategy: BOS/CHoCH + Order Blocks + Liquidity
✅ Multi-timeframe (H1/M15/M5)
✅ Full error handling
✅ Auto-restart on failure
"""

import os, sys, time, logging, sqlite3, hashlib
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd, yfinance as yf
import telebot, requests

TELEGRAM_TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
TELEGRAM_CHAT_ID = '7553333305'

SYMBOLS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
SCAN_INTERVAL = 60
MIN_CONFIDENCE = 0.70
COOLDOWN_MINUTES = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('Falcon')

try:
    requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=5)
    time.sleep(1)
except: pass

tb = telebot.TeleBot(TELEGRAM_TOKEN)

class Database:
    def __init__(self):
        self.db_path = 'falcon_inst.db'
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, direction TEXT, entry_price REAL, stop_loss REAL, take_profit REAL, confidence REAL, setup_type TEXT, entry_time DATETIME DEFAULT CURRENT_TIMESTAMP, expiry_time DATETIME, result TEXT DEFAULT 'PENDING', pnl_percent REAL, signal_hash TEXT UNIQUE)''')
                conn.commit()
        except: pass
    
    def save(self, data):
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('INSERT OR IGNORE INTO signals (symbol, direction, entry_price, stop_loss, take_profit, confidence, setup_type, expiry_time, signal_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                           (data['symbol'], data['direction'], data['entry_price'], data['stop_loss'], data['take_profit'], data['confidence'], data.get('setup_type',''), data['expiry_time'], h))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except: return None
    
    def update(self, sid, exit_price, result, pnl):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("UPDATE signals SET exit_price=?, result=?, pnl_percent=?, exit_time=datetime('now','localtime') WHERE id=?", (exit_price, result, pnl, sid))
                conn.commit()
        except: pass
    
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

def get_data(symbol, interval='15m', period='5d'):
    try:
        imap = {'5m':'5m','15m':'15m','1h':'1h'}
        df = yf.download(symbol, period=period, interval=imap.get(interval,'15m'), progress=False)
        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            df.columns = [str(c).lower() for c in df.columns]
            return df
    except: pass
    return None

def detect_structure(df):
    try:
        if len(df) < 20: return {'bos': False, 'choch': False, 'trend': 'neutral'}
        h = df['high'].values; l = df['low'].values; c = df['close'].values
        swing_high = max(h[-20:]); swing_low = min(l[-20:])
        recent_high = max(h[-5:]); recent_low = min(l[-5:])
        ema50 = pd.Series(c).ewm(span=50).mean().values[-1] if len(c) >= 50 else np.mean(c)
        trend = 'UP' if c[-1] > ema50 else 'DOWN'
        bos = (trend == 'UP' and recent_high > swing_high) or (trend == 'DOWN' and recent_low < swing_low)
        choch = (trend == 'UP' and recent_low < swing_low) or (trend == 'DOWN' and recent_high > swing_high)
        return {'bos': bos, 'choch': choch, 'trend': trend}
    except: return {'bos': False, 'choch': False, 'trend': 'neutral'}

def detect_liquidity_sweep(df):
    try:
        if len(df) < 20: return False
        h = df['high'].values; l = df['low'].values; c = df['close'].values
        return c[-1] > max(h[-20:]) * 1.001 or c[-1] < min(l[-20:]) * 0.999
    except: return False

def find_order_blocks(df):
    try:
        if len(df) < 30: return None, None
        h = df['high'].values; l = df['low'].values; c = df['close'].values
        demand = []; supply = []
        for i in range(5, len(c)-1):
            if c[i] < c[i-1] and c[i] < c[i+1]: demand.append({'low': float(l[i]), 'high': float(h[i])})
            if c[i] > c[i-1] and c[i] > c[i+1]: supply.append({'low': float(l[i]), 'high': float(h[i])})
        return demand[-1] if demand else None, supply[-1] if supply else None
    except: return None, None

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
        return float(np.mean(tr[-14:]))
    except: return 0.0001

def analyze_institutional(symbol):
    try:
        df_h1 = get_data(symbol, '1h', '5d')
        df_m15 = get_data(symbol, '15m', '3d')
        df_m5 = get_data(symbol, '5m', '2d')
        
        if df_h1 is None or df_m15 is None or df_m5 is None: return None
        
        struct = detect_structure(df_h1)
        if not struct['bos'] and not struct['choch']: return None
        
        liq = detect_liquidity_sweep(df_m15)
        demand, supply = find_order_blocks(df_m15)
        
        price = float(df_m5['close'].iloc[-1])
        rsi = calc_rsi(df_m5['close'].values)
        atr = calc_atr(df_m15['high'].values, df_m15['low'].values, df_m15['close'].values)
        
        d = None; setup = ""; conf = 0; sl = None; tp = None
        
        if struct['bos'] and struct['trend'] == 'UP' and demand and price <= demand['high'] * 1.002 and rsi < 50:
            d = 'BUY'; setup = 'BOS + Demand'; conf = min(0.88, 0.6+struct['bos']*0.1+liq*0.1)
            sl = round(demand['low'] - atr*0.3, 5); tp = round(price + atr*3.0, 5)
        
        elif struct['bos'] and struct['trend'] == 'DOWN' and supply and price >= supply['low'] * 0.998 and rsi > 50:
            d = 'SELL'; setup = 'BOS + Supply'; conf = min(0.88, 0.6+struct['bos']*0.1+liq*0.1)
            sl = round(supply['high'] + atr*0.3, 5); tp = round(price - atr*3.0, 5)
        
        elif struct['choch'] and liq:
            if struct['trend'] == 'UP' and demand:
                d = 'BUY'; setup = 'CHoCH + Liq Sweep'; conf = 0.82
                sl = round(demand['low']-atr*0.3, 5) if demand else round(price-atr*1.5, 5)
                tp = round(price + atr*3.0, 5)
            elif struct['trend'] == 'DOWN' and supply:
                d = 'SELL'; setup = 'CHoCH + Liq Sweep'; conf = 0.82
                sl = round(supply['high']+atr*0.3, 5) if supply else round(price+atr*1.5, 5)
                tp = round(price - atr*3.0, 5)
        
        if d is None or conf < MIN_CONFIDENCE: return None
        
        return {
            'symbol': symbol, 'direction': d, 'price': price,
            'stop_loss': sl, 'take_profit': tp, 'confidence': conf,
            'setup_type': setup,
            'expiry_time': (datetime.now() + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        logger.error(f"analyze {symbol}: {e}")
        return None

def send_message(text):
    try: tb.send_message(TELEGRAM_CHAT_ID, text)
    except: pass

def main():
    db = Database()
    logger.info("Falcon Institutional - Started")
    
    try: send_message("Falcon Institutional\nReady")
    except: pass
    
    while True:
        try:
            for trade in db.get_expired():
                try:
                    df = get_data(trade['symbol'], '5m', '1d')
                    if df is not None and len(df) > 0:
                        close_p = float(df['close'].iloc[-1])
                        entry = trade['entry_price']; direction = trade['direction']
                        pnl = (close_p-entry)/entry*100 if direction=='BUY' else (entry-close_p)/entry*100
                        result = 'WIN' if pnl > 0 else 'LOSS'
                        db.update(trade['id'], close_p, result, pnl)
                        logger.info(f"{'WIN' if result=='WIN' else 'LOSS'} {trade['symbol']}: {pnl:+.2f}%")
                except: pass
            
            now = datetime.now(timezone.utc)
            if now.weekday() < 5:
                for symbol in SYMBOLS:
                    try:
                        if db.was_recent(symbol): continue
                        signal = analyze_institutional(symbol)
                        if signal and db.save(signal):
                            emoji = "BUY" if signal['direction']=='BUY' else "SELL"
                            direction = "Buy" if signal['direction']=='BUY' else "Sell"
                            msg = f"{emoji} {symbol} - {direction}\n{signal['price']:.5f}\n{signal['setup_type']}\n{signal['confidence']:.1%}"
                            send_message(msg)
                            logger.info(f"SIGNAL: {symbol} {signal['direction']} | {signal['setup_type']}")
                        time.sleep(2)
                    except: pass
            
            time.sleep(SCAN_INTERVAL)
            
        except KeyboardInterrupt: break
        except Exception as e:
            logger.error(f"Loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    while True:
        try:
            main()
        except KeyboardInterrupt: break
        except Exception as e:
            logger.error(f"Fatal: {e}")
            time.sleep(30)
