#!/usr/bin/env python3
"""
Falcon AI v5.3 - No Polling (Zero 409)
========================================
✅ 4 independent strategies
✅ No infinity_polling - No 409 error
✅ Send-only Telegram mode
✅ Each strategy works independently
"""

import os, sys, time, logging, sqlite3, hashlib
from datetime import datetime, timedelta
import numpy as np, pandas as pd, yfinance as yf
import telebot
import requests

TELEGRAM_TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
TELEGRAM_CHAT_ID = '7553333305'

SYMBOLS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
    'USDCAD=X', 'EURGBP=X', 'EURJPY=X', 'GBPJPY=X'
]

SCAN_INTERVAL = 15
MIN_CONFIDENCE = 0.50
COOLDOWN_MINUTES = 2
TRADE_DURATION = 5

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s', datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger('FalconV5')

# ✅ حذف Webhook أول حاجة
try:
    requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=5)
    time.sleep(1)
    logger.info("✅ Webhook deleted")
except: pass

# ✅ بوت إرسال فقط - مفيش polling
tb = telebot.TeleBot(TELEGRAM_TOKEN)

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
            if row and row[0]:
                conn.execute('UPDATE strategy_performance SET total_trades=total_trades+1, wins=wins+?, total_pnl=total_pnl+? WHERE strategy=?', (1 if result=='WIN' else 0, pnl, row[0]))
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

def calc_rsi(df):
    try:
        c = df['Close'].values; delta = np.diff(c)
        gain = np.mean(delta[delta>0]) if any(delta>0) else 0
        loss = np.mean(-delta[delta<0]) if any(delta<0) else 0
        return round(100-100/(1+gain/(loss+1e-8)),1) if loss>0 else 50
    except: return 50

def calc_atr(df):
    try:
        h,l,c = df['High'].values, df['Low'].values, df['Close'].values
        tr = np.array([max(h[i+1]-l[i+1], abs(h[i+1]-c[i]), abs(l[i+1]-c[i])) for i in range(len(c)-1)])
        return round(float(np.mean(tr[-14:])),5)
    except: return 0.0001

def calc_ema(df, period):
    try: return round(float(pd.Series(df['Close'].values).ewm(span=period, adjust=False).mean().values[-1]),5)
    except: return 0

def find_sr(df):
    try:
        h = df['High'].values[-50:]; l = df['Low'].values[-50:]
        return float(np.mean(np.sort(l)[:3])), float(np.mean(np.sort(h)[-3:]))
    except: return 0, 0

def find_range(df):
    try:
        h = df['High'].values[-30:]; l = df['Low'].values[-30:]
        rh, rl = float(np.max(h)), float(np.min(l))
        return rl, rh, (rh-rl)/rl*100 < 0.3
    except: return 0, 0, False

def calc_fibo(df):
    try:
        h = df['High'].values[-50:]; l = df['Low'].values[-50:]
        sh, sl = float(np.max(h)), float(np.min(l)); d = sh-sl
        return {'high':sh,'low':sl,'f382':round(sl+d*0.382,5),'f500':round(sl+d*0.500,5),'f618':round(sl+d*0.618,5)}
    except: return {'high':0,'low':0,'f382':0,'f500':0,'f618':0}

# ========== 4 STRATEGIES ==========

def strat_rsi_sr(df, symbol):
    if len(df) < 50: return None
    price = float(df['Close'].iloc[-1]); rsi = calc_rsi(df); atr = calc_atr(df)
    support, resistance = find_sr(df)
    if support == 0: return None
    
    if rsi < 40 and (price-support)/support*100 < 0.15:
        return {'direction':'BUY','price':price,'stop_loss':round(support,5),'take_profit':round(price+atr*2,5),'confidence':min(0.80,0.5+(40-rsi)*0.02),'strategy':'divergence','strategy_name':'RSI + دعم'}
    if rsi > 60 and (resistance-price)/price*100 < 0.15:
        return {'direction':'SELL','price':price,'stop_loss':round(resistance,5),'take_profit':round(price-atr*2,5),'confidence':min(0.80,0.5+(rsi-60)*0.02),'strategy':'divergence','strategy_name':'RSI + مقاومة'}
    return None

def strat_ema_trend(df, symbol):
    if len(df) < 60: return None
    price = float(df['Close'].iloc[-1]); ema20=calc_ema(df,20); ema50=calc_ema(df,50)
    atr = calc_atr(df); rsi = calc_rsi(df)
    
    if price > ema20 and ema20 > ema50 and rsi < 60:
        return {'direction':'BUY','price':price,'stop_loss':round(ema50,5),'take_profit':round(price+atr*2,5),'confidence':0.65,'strategy':'ema_cross','strategy_name':'EMA ترند صاعد'}
    if price < ema20 and ema20 < ema50 and rsi > 40:
        return {'direction':'SELL','price':price,'stop_loss':round(ema50,5),'take_profit':round(price-atr*2,5),'confidence':0.65,'strategy':'ema_cross','strategy_name':'EMA ترند هابط'}
    return None

def strat_breakout(df, symbol):
    if len(df) < 40: return None
    rl, rh, is_range = find_range(df)
    if not is_range: return None
    c = df['Close'].values; price = float(c[-1]); atr = calc_atr(df)
    
    if float(c[-2]) > rh and float(c[-1]) >= rh:
        return {'direction':'BUY','price':price,'stop_loss':round(rl,5),'take_profit':round(price+atr*3,5),'confidence':0.70,'strategy':'breakout_retest','strategy_name':'اختراق علوي'}
    if float(c[-2]) < rl and float(c[-1]) <= rl:
        return {'direction':'SELL','price':price,'stop_loss':round(rh,5),'take_profit':round(price-atr*3,5),'confidence':0.70,'strategy':'breakout_retest','strategy_name':'اختراق سفلي'}
    return None

def strat_fibo(df, df_1h, symbol):
    if len(df) < 50 or df_1h is None or len(df_1h) < 50: return None
    fibo = calc_fibo(df_1h)
    if fibo['f500'] == 0: return None
    price = float(df['Close'].iloc[-1])
    trend_up = calc_ema(df_1h,20) > calc_ema(df_1h,50)
    rsi = calc_rsi(df); atr = calc_atr(df)
    
    if trend_up and abs(price-fibo['f500'])/price*100 < 0.15 and rsi < 50:
        return {'direction':'BUY','price':price,'stop_loss':round(fibo['f618'],5),'take_profit':round(fibo['high'],5),'confidence':0.68,'strategy':'fibo_volume','strategy_name':'فيبو 50% صاعد'}
    if not trend_up and abs(price-fibo['f500'])/price*100 < 0.15 and rsi > 50:
        return {'direction':'SELL','price':price,'stop_loss':round(fibo['f618'],5),'take_profit':round(fibo['low'],5),'confidence':0.68,'strategy':'fibo_volume','strategy_name':'فيبو 50% هابط'}
    return None

# ========== SEND MESSAGE SAFELY ==========

def send_message(text):
    try:
        tb.send_message(TELEGRAM_CHAT_ID, text, parse_mode='Markdown')
    except:
        try:
            tb.send_message(TELEGRAM_CHAT_ID, text)
        except:
            pass

# ========== MAIN ==========

def main():
    db = Database()
    logger.info("🦅 Falcon Pro v5.3 - No Polling")
    
    send_message("🦅 **Falcon Pro v5.3**\n✅ 4 استراتيجيات\n⚡️ يعمل...")
    
    while True:
        try:
            # فحص الصفقات المنتهية
            for trade in db.get_expired_trades():
                df = DataFetcher.fetch(trade['symbol'], '1m')
                if df is not None and len(df) > 0:
                    close_p = float(df['Close'].iloc[-1])
                    entry = trade['entry_price']; direction = trade['direction']
                    is_jpy = "JPY" in trade['symbol']; pv = 0.01 if is_jpy else 0.0001
                    
                    if direction == 'BUY':
                        pnl = (close_p-entry)/entry*100; pips = (close_p-entry)/pv
                        result = 'WIN' if close_p > entry else 'LOSS'
                    else:
                        pnl = (entry-close_p)/entry*100; pips = (entry-close_p)/pv
                        result = 'WIN' if close_p < entry else 'LOSS'
                    
                    db.update_result(trade['id'], close_p, result, pnl, round(pips,1))
                    logger.info(f"{'✅' if result=='WIN' else '❌'} {trade['symbol']}: {result} | {trade.get('strategy_name','')}")
            
            # بحث عن فرص
            now = datetime.utcnow()
            if now.weekday() < 5:  # مش ويكند
                for symbol in SYMBOLS:
                    if db.has_active_signal(symbol): continue
                    if db.was_recent(symbol): continue
                    
                    df_15m = DataFetcher.fetch(symbol, '15m')
                    df_1h = DataFetcher.fetch(symbol, '1h')
                    
                    if df_15m is None: continue
                    
                    # ✅ كل الاستراتيجيات تشتغل
                    strategies = [
                        strat_rsi_sr(df_15m, symbol),
                        strat_ema_trend(df_15m, symbol),
                        strat_breakout(df_15m, symbol),
                    ]
                    if df_1h is not None:
                        strategies.append(strat_fibo(df_15m, df_1h, symbol))
                    
                    for s in strategies:
                        if s and s['confidence'] >= MIN_CONFIDENCE:
                            s['symbol'] = symbol
                            s['expiry_time'] = (datetime.now() + timedelta(minutes=TRADE_DURATION)).strftime('%Y-%m-%d %H:%M:%S')
                            if db.save_signal(s):
                                emoji = "🟢" if s['direction']=='BUY' else "🔴"
                                direction = "شراء" if s['direction']=='BUY' else "بيع"
                                msg = f"{emoji} **{symbol}** - {direction}\n\n💰 {s['price']:.5f}\n💪 {s['confidence']:.1%}\n📊 {s['strategy_name']}"
                                send_message(msg)
                                logger.info(f"✅ {symbol} {s['direction']} | {s['strategy_name']}")
                    
                    time.sleep(1)
            
            time.sleep(SCAN_INTERVAL)
            
        except KeyboardInterrupt: break
        except Exception as e:
            logger.error(f"خطأ: {e}")
            time.sleep(10)

if __name__ == "__main__":
    main()
