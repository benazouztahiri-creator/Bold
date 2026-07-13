#!/usr/bin/env python3
"""
Falcon AI v6 - Quality Over Quantity
======================================
✅ One proven strategy: RSI + Support/Resistance
✅ High confidence only (>65%)
✅ 5-min trades
✅ Max 2 trades per hour
"""

import os, sys, time, logging, sqlite3, hashlib, threading
from datetime import datetime, timedelta
import numpy as np, pandas as pd, yfinance as yf
import telebot

TELEGRAM_TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
TELEGRAM_CHAT_ID = '7553333305'
SYMBOLS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']
SCAN_INTERVAL = 30
MIN_CONFIDENCE = 0.65

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('FalconV6')

class Database:
    def __init__(self):
        self.db_path = 'falcon_v6.db'
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, direction TEXT, entry_price REAL, exit_price REAL, stop_loss REAL, take_profit REAL, confidence REAL, entry_time DATETIME DEFAULT CURRENT_TIMESTAMP, expiry_time DATETIME, result TEXT DEFAULT 'PENDING', pnl_percent REAL, signal_hash TEXT UNIQUE)''')
            conn.commit()
    
    def save(self, data):
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('INSERT OR IGNORE INTO signals (symbol, direction, entry_price, stop_loss, take_profit, confidence, expiry_time, signal_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                           (data['symbol'], data['direction'], data['entry_price'], data['stop_loss'], data['take_profit'], data['confidence'], data['expiry_time'], h))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except: return None
    
    def update(self, sid, exit_price, result, pnl):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE signals SET exit_price=?, result=?, pnl_percent=?, exit_time=datetime('now','localtime') WHERE id=?", (exit_price, result, pnl, sid))
            conn.commit()
    
    def recent_count(self, hours=1):
        cutoff = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM signals WHERE entry_time > ?", (cutoff,)).fetchone()[0]
    
    def was_recent(self, symbol, minutes=15):
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
        if not df.empty:
            df.columns = [str(c).capitalize() for c in df.columns]
            return df
    except: pass
    return None

def calc_rsi(df, period=14):
    c = df['Close'].values; delta = np.diff(c)
    gain = np.mean(delta[delta>0]) if any(delta>0) else 0
    loss = np.mean(-delta[delta<0]) if any(delta<0) else 0
    return 100 - 100/(1 + gain/(loss+1e-8)) if loss > 0 else 50

def calc_atr(df, period=14):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    tr = np.array([max(h[i+1]-l[i+1], abs(h[i+1]-c[i]), abs(l[i+1]-c[i])) for i in range(len(c)-1)])
    return float(np.mean(tr[-period:]))

def find_levels(df):
    h = df['High'].values[-50:]; l = df['Low'].values[-50:]
    support = float(np.mean(np.sort(l)[:3]))
    resistance = float(np.mean(np.sort(h)[-3:]))
    return support, resistance

def analyze(symbol):
    df = get_data(symbol)
    if df is None or len(df) < 50: return None
    
    price = float(df['Close'].iloc[-1])
    rsi = calc_rsi(df)
    atr = calc_atr(df)
    support, resistance = find_levels(df)
    
    # ✅ RSI متطرف + قريب من المستوى
    if rsi < 35 and abs(price - support) / support < 0.001:
        sl = round(support - atr * 0.5, 5)
        tp = round(price + atr * 2.0, 5)
        confidence = min(0.85, 0.6 + (35 - rsi) * 0.02)
        return {'symbol': symbol, 'direction': 'BUY', 'price': price, 'stop_loss': sl, 
                'take_profit': tp, 'confidence': confidence,
                'expiry_time': (datetime.now() + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')}
    
    if rsi > 65 and abs(resistance - price) / price < 0.001:
        sl = round(resistance + atr * 0.5, 5)
        tp = round(price - atr * 2.0, 5)
        confidence = min(0.85, 0.6 + (rsi - 65) * 0.02)
        return {'symbol': symbol, 'direction': 'SELL', 'price': price, 'stop_loss': sl,
                'take_profit': tp, 'confidence': confidence,
                'expiry_time': (datetime.now() + timedelta(minutes=5)).strftime('%Y-%m-%d %H:%M:%S')}
    
    return None

class FalconPro:
    def __init__(self):
        self.db = Database()
        self.tb = telebot.TeleBot(TELEGRAM_TOKEN)
        self._setup()
    
    def _setup(self):
        try:
            import requests
            requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=3)
        except: pass
        
        @self.tb.message_handler(commands=['start'])
        def start(msg):
            t, w, r = self.db.stats()
            self.tb.reply_to(msg, f"🦅 **Falcon v6**\n\n📊 الصفقات: {t}\n✅ الرابحة: {w}\n📈 النسبة: {r:.1%}", parse_mode='Markdown')
    
    def run(self):
        logger.info("🦅 Falcon v6 - Quality First")
        
        def poll():
            while True:
                try: self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except: time.sleep(5)
        threading.Thread(target=poll, daemon=True).start()
        time.sleep(1)
        
        while True:
            try:
                # فحص الصفقات المنتهية
                for trade in self.db.get_expired():
                    df = get_data(trade['symbol'])
                    if df is not None:
                        close_p = float(df['Close'].iloc[-1])
                        entry = trade['entry_price']
                        direction = trade['direction']
                        pnl = (close_p-entry)/entry*100 if direction=='BUY' else (entry-close_p)/entry*100
                        result = 'WIN' if pnl > 0 else 'LOSS'
                        self.db.update(trade['id'], close_p, result, pnl)
                        logger.info(f"{'✅' if result=='WIN' else '❌'} {trade['symbol']}: {result} | {pnl:+.2f}%")
                
                # ✅ حد أقصى 2 صفقة في الساعة
                if self.db.recent_count(1) >= 2:
                    time.sleep(60)
                    continue
                
                # بحث عن فرصة
                for symbol in SYMBOLS:
                    if self.db.was_recent(symbol): continue
                    
                    signal = analyze(symbol)
                    if signal and signal['confidence'] >= MIN_CONFIDENCE:
                        if self.db.save(signal):
                            emoji = "🟢" if signal['direction']=='BUY' else "🔴"
                            direction = "شراء" if signal['direction']=='BUY' else "بيع"
                            msg = f"{emoji} **{symbol}** - {direction}\n\n💰 {signal['price']:.5f}\n💪 {signal['confidence']:.1%}"
                            try: self.tb.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
                            except: pass
                            logger.info(f"✅ {symbol} {signal['direction']} | {signal['confidence']:.1%}")
                    time.sleep(2)
                
                time.sleep(SCAN_INTERVAL)
                
            except KeyboardInterrupt: break
            except: time.sleep(10)

if __name__ == "__main__":
    bot = FalconPro(); bot.run()
