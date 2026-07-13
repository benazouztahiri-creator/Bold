#!/usr/bin/env python3
"""
Falcon AI v6 - One Proven Strategy
====================================
✅ RSI + Support/Resistance only
✅ High confidence threshold (>65%)
✅ 5-minute trades
✅ Max 3 trades per day
✅ Quality over quantity
"""

import os, sys, time, logging, sqlite3, hashlib
from datetime import datetime, timedelta, timezone
import numpy as np, pandas as pd, yfinance as yf
import telebot, requests

TELEGRAM_TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
TELEGRAM_CHAT_ID = '7553333305'
SYMBOLS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
SCAN_INTERVAL = 60
MIN_CONFIDENCE = 0.65
MAX_DAILY_TRADES = 3

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('FalconV6')

try:
    requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=5)
    time.sleep(1)
except: pass

tb = telebot.TeleBot(TELEGRAM_TOKEN)

class Database:
    def __init__(self):
        self.db_path = 'falcon_v6.db'
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, direction TEXT, entry_price REAL, exit_price REAL, stop_loss REAL, take_profit REAL, confidence REAL, rsi REAL, entry_time DATETIME DEFAULT CURRENT_TIMESTAMP, expiry_time DATETIME, result TEXT DEFAULT 'PENDING', pnl_percent REAL, signal_hash TEXT UNIQUE)''')
            conn.commit()
    
    def save(self, data):
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('INSERT OR IGNORE INTO signals (symbol, direction, entry_price, stop_loss, take_profit, confidence, rsi, expiry_time, signal_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                           (data['symbol'], data['direction'], data['entry_price'], data['stop_loss'], data['take_profit'], data['confidence'], data.get('rsi', 0), data['expiry_time'], h))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except: return None
    
    def update(self, sid, exit_price, result, pnl):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE signals SET exit_price=?, result=?, pnl_percent=?, exit_time=datetime('now','localtime') WHERE id=?", (exit_price, result, pnl, sid))
            conn.commit()
    
    def today_count(self):
        today = datetime.now().strftime('%Y-%m-%d')
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM signals WHERE entry_time LIKE ?", (f"{today}%",)).fetchone()[0]
    
    def was_recent(self, symbol, minutes=30):
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute('SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?', (symbol, cutoff)).fetchone()[0] > 0
    
    def get_expired(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in conn.execute("SELECT * FROM signals WHERE result='PENDING' AND expiry_time <= datetime('now','localtime')").fetchall()]
    
    def stats(self):
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM signals WHERE result!='PENDING'").fetchone()[0]
            wins = conn.execute("SELECT COUNT(*) FROM signals WHERE result='WIN'").fetchone()[0]
            return total, wins, wins/total if total > 0 else 0

def get_data(symbol):
    try:
        df = yf.download(symbol, period='5d', interval='15m', progress=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        if 'close' not in df.columns: return None
        return df
    except: return None

def calc_rsi(df):
    try:
        c = df['close'].values; delta = np.diff(c)
        gain = np.mean(delta[delta>0]) if any(delta>0) else 0
        loss = np.mean(-delta[delta<0]) if any(delta<0) else 0
        return round(100-100/(1+gain/(loss+1e-8)),1) if loss>0 else 50
    except: return 50

def calc_atr(df):
    try:
        h=df['high'].values; l=df['low'].values; c=df['close'].values
        tr = np.array([max(h[i+1]-l[i+1], abs(h[i+1]-c[i]), abs(l[i+1]-c[i])) for i in range(len(c)-1)])
        return float(np.mean(tr[-14:]))
    except: return 0.0001

def find_levels(df):
    try:
        h=df['high'].values[-50:]; l=df['low'].values[-50:]
        support = float(np.mean(np.sort(l)[:3]))
        resistance = float(np.mean(np.sort(h)[-3:]))
        return support, resistance
    except: return 0, 0

def analyze(symbol):
    """✅ استراتيجية واحدة: RSI + دعم/مقاومة"""
    df = get_data(symbol)
    if df is None or len(df) < 50: return None
    
    price = float(df['close'].iloc[-1])
    rsi = calc_rsi(df)
    atr = calc_atr(df)
    support, resistance = find_levels(df)
    
    if support == 0: return None
    
    # ✅ شراء: RSI منخفض + السعر قريب من الدعم
    if rsi < 35 and (price - support) / support * 100 < 0.15:
        sl = round(support - atr * 0.3, 5)
        tp = round(price + atr * 2.5, 5)
        confidence = min(0.85, 0.6 + (35 - rsi) * 0.025)
        return {'symbol': symbol, 'direction': 'BUY', 'price': price, 'stop_loss': sl, 
                'take_profit': tp, 'confidence': confidence, 'rsi': rsi,
                'expiry_time': (datetime.now() + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')}
    
    # ✅ بيع: RSI مرتفع + السعر قريب من المقاومة
    if rsi > 65 and (resistance - price) / price * 100 < 0.15:
        sl = round(resistance + atr * 0.3, 5)
        tp = round(price - atr * 2.5, 5)
        confidence = min(0.85, 0.6 + (rsi - 65) * 0.025)
        return {'symbol': symbol, 'direction': 'SELL', 'price': price, 'stop_loss': sl,
                'take_profit': tp, 'confidence': confidence, 'rsi': rsi,
                'expiry_time': (datetime.now() + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')}
    
    return None

def send_message(text):
    try: tb.send_message(TELEGRAM_CHAT_ID, text, parse_mode='Markdown')
    except:
        try: tb.send_message(TELEGRAM_CHAT_ID, text)
        except: pass

def main():
    db = Database()
    logger.info("Falcon v6 - One Strategy")
    
    total, wins, rate = db.stats()
    send_message(f"Falcon v6\n\nTotal: {total}\nWins: {wins}\nRate: {rate:.1%}")
    
    while True:
        try:
            # فحص الصفقات المنتهية
            for trade in db.get_expired():
                df = get_data(trade['symbol'])
                if df is not None and len(df) > 0:
                    close_p = float(df['close'].iloc[-1])
                    entry = trade['entry_price']
                    direction = trade['direction']
                    pnl = (close_p-entry)/entry*100 if direction=='BUY' else (entry-close_p)/entry*100
                    result = 'WIN' if pnl > 0 else 'LOSS'
                    db.update(trade['id'], close_p, result, pnl)
                    logger.info(f"{'WIN' if result=='WIN' else 'LOSS'} {trade['symbol']}: {pnl:+.2f}%")
            
            # ✅ حد أقصى 3 صفقات في اليوم
            if db.today_count() >= MAX_DAILY_TRADES:
                time.sleep(300)
                continue
            
            # بحث عن فرصة
            now = datetime.now(timezone.utc)
            if now.weekday() < 5:
                for symbol in SYMBOLS:
                    if db.was_recent(symbol): continue
                    
                    signal = analyze(symbol)
                    if signal and signal['confidence'] >= MIN_CONFIDENCE:
                        if db.save(signal):
                            emoji = "BUY" if signal['direction']=='BUY' else "SELL"
                            direction = "Buy" if signal['direction']=='BUY' else "Sell"
                            msg = f"{emoji} {symbol} - {direction}\n\n{signal['price']:.5f}\nRSI: {signal['rsi']}\nConf: {signal['confidence']:.1%}"
                            send_message(msg)
                            logger.info(f"SIGNAL: {symbol} {signal['direction']} RSI:{signal['rsi']}")
                    time.sleep(2)
            
            time.sleep(SCAN_INTERVAL)
            
        except KeyboardInterrupt: break
        except Exception as e:
            logger.error(f"Error: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
