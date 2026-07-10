#!/usr/bin/env python3
"""
Falcon AI v5.1 - Enhanced 4 Strategies (Fixed)
================================================
Strategy 1: RSI Divergence + Support/Resistance
Strategy 2: Breakout with Confirmed Retest (Candle Close)
Strategy 3: EMA 20/50 Crossover (Price Proximity Check)
Strategy 4: Fibonacci + Volume Confirmation
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

SCAN_INTERVAL = 60
MIN_CONFIDENCE = 0.55

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FalconV5')

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
# HELPER
# ============================================================================

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
# DATABASE
# ============================================================================

class Database:
    def __init__(self):
        self.db_path = 'falcon_v5.db'
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY AUTOINCREMENT, symbol TEXT, direction TEXT, entry_price REAL, exit_price REAL, stop_loss REAL, take_profit REAL, confidence REAL, score REAL, strategy TEXT, strategy_name TEXT, entry_time DATETIME DEFAULT CURRENT_TIMESTAMP, expiry_time DATETIME, exit_time DATETIME, result TEXT DEFAULT 'PENDING', pnl_percent REAL, pnl_pips REAL, signal_hash TEXT UNIQUE)''')
            
            conn.execute('''CREATE TABLE IF NOT EXISTS strategy_performance (strategy TEXT PRIMARY KEY, total_trades INTEGER DEFAULT 0, wins INTEGER DEFAULT 0, total_pnl REAL DEFAULT 0, win_rate REAL DEFAULT 0.5)''')
            
            for strat in ['divergence', 'breakout_retest', 'ema_cross', 'fibo_volume']:
                conn.execute('INSERT OR IGNORE INTO strategy_performance (strategy) VALUES (?)', (strat,))
            
            conn.commit()
    
    def save_signal(self, data):
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''INSERT OR IGNORE INTO signals (symbol, direction, entry_price, stop_loss, take_profit, confidence, score, strategy, strategy_name, expiry_time, signal_hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                           (data['symbol'], data['direction'], data['entry_price'],
                            data.get('stop_loss'), data.get('take_profit'),
                            data['confidence'], data.get('score', 0),
                            data.get('strategy', ''), data.get('strategy_name', ''),
                            data['expiry_time'], h))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except:
            return None
    
    def update_result(self, signal_id, exit_price, result, pnl, pips):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE signals SET exit_price=?, result=?, pnl_percent=?, pnl_pips=?, exit_time=datetime('now','localtime') WHERE id=?",
                        (exit_price, result, pnl, pips, signal_id))
            
            row = conn.execute('SELECT strategy FROM signals WHERE id=?', (signal_id,)).fetchone()
            if row and row[0]:
                conn.execute('UPDATE strategy_performance SET total_trades=total_trades+1, wins=wins+?, total_pnl=total_pnl+? WHERE strategy=?',
                           (1 if result == 'WIN' else 0, pnl, row[0]))
            conn.commit()
    
    def has_active_signal(self, symbol):
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute("SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' AND expiry_time > datetime('now','localtime')", (symbol,)).fetchone()[0]
            return c > 0
    
    def was_recent(self, symbol, minutes=10):
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?', (symbol, cutoff)).fetchone()[0]
            return c > 0
    
    def get_expired_trades(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM signals WHERE result='PENDING' AND expiry_time <= datetime('now','localtime')").fetchall()
            return [dict(r) for r in rows]
    
    def get_strategy_weights(self):
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute('SELECT strategy, wins, total_trades FROM strategy_performance').fetchall()
            weights = {}
            for strat, wins, total in rows:
                weights[strat] = wins / total if total > 5 else 0.5
            return weights

# ============================================================================
# DATA FETCHER
# ============================================================================

class DataFetcher:
    @staticmethod
    def fetch(symbol, interval='5min'):
        key = f"{symbol}_{interval}"
        cached = data_cache.get(key)
        if cached is not None:
            return cached
        
        try:
            import yfinance as yf
            yf_symbol = symbol if '=X' in symbol else f"{symbol}=X"
            imap = {'5m': '5m', '15m': '15m', '1h': '1h', '1m': '1m'}
            
            df = yf.download(yf_symbol, period='5d', interval=imap.get(interval, '5m'), progress=False)
            df = safe_columns(df)
            if not df.empty:
                data_cache.set(key, df)
                return df
        except:
            pass
        return None

# ============================================================================
# INDICATORS
# ============================================================================

def calc_rsi_series(df, period=14):
    c = df['Close'].values
    delta = np.diff(c)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    rsi = np.zeros(len(c) - period)
    avg_g = np.mean(gain[:period])
    avg_l = np.mean(loss[:period])
    for i in range(period, len(c) - 1):
        avg_g = (avg_g * (period - 1) + gain[i]) / period
        avg_l = (avg_l * (period - 1) + loss[i]) / period
        rsi[i - period + 1] = 100 - 100 / (1 + avg_g / (avg_l + 1e-8))
    return rsi

def calc_rsi(df, period=14):
    c = df['Close'].values
    delta = np.diff(c)
    gain = np.mean(delta[delta > 0]) if any(delta > 0) else 0
    loss = np.mean(-delta[delta < 0]) if any(delta < 0) else 0
    return round(100 - 100/(1 + gain/(loss+1e-8)), 1) if loss > 0 else 50

def calc_atr(df, period=14):
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    tr = np.array([max(h[i+1]-l[i+1], abs(h[i+1]-c[i]), abs(l[i+1]-c[i])) for i in range(len(c)-1)])
    return round(float(np.mean(tr[-period:])), 5)

def calc_ema(df, period):
    return round(float(pd.Series(df['Close'].values).ewm(span=period, adjust=False).mean().values[-1]), 5)

def find_sr(df, lookback=50):
    h = df['High'].values[-lookback:]
    l = df['Low'].values[-lookback:]
    return float(np.mean(np.sort(l)[:5])), float(np.mean(np.sort(h)[-5:]))

def find_range(df, lookback=30):
    h = df['High'].values[-lookback:]
    l = df['Low'].values[-lookback:]
    rh, rl = float(np.max(h)), float(np.min(l))
    return rl, rh, (rh - rl) / rl * 100 < 0.3

def calc_fibo(df):
    h = df['High'].values[-50:]
    l = df['Low'].values[-50:]
    sh, sl = float(np.max(h)), float(np.min(l))
    d = sh - sl
    return {
        'high': sh, 'low': sl,
        'f382': round(sl + d * 0.382, 5),
        'f500': round(sl + d * 0.500, 5),
        'f618': round(sl + d * 0.618, 5),
    }

def get_vol(df):
    if 'Volume' not in df.columns: return 1.0
    v = df['Volume'].values
    return float(v[-1] / (np.mean(v[-20:]) + 1e-8)) if len(v) >= 20 else 1.0

# ============================================================================
# STRATEGY 1: RSI Divergence
# ============================================================================

def strat_divergence(df, symbol):
    if len(df) < 50: return None
    
    c = df['Close'].values
    rsi_series = calc_rsi_series(df)
    if len(rsi_series) < 20: return None
    
    price = float(c[-1])
    support, resistance = find_sr(df)
    atr = calc_atr(df)
    rsi_now = calc_rsi(df)
    
    p20l = float(np.min(c[-20:]))
    p40l = float(np.min(c[-40:-20]))
    r20l = float(np.min(rsi_series[-20:]))
    r40l = float(np.min(rsi_series[-40:-20])) if len(rsi_series) >= 40 else r20l
    
    if p20l < p40l and r20l > r40l and rsi_now < 45:
        if (price - support) / support * 100 < 0.1:
            return {
                'direction': 'BUY', 'price': price,
                'stop_loss': round(support - atr * 0.5, 5),
                'take_profit': round(price + atr * 3.0, 5),
                'confidence': min(0.88, 0.5 + (45 - rsi_now) * 0.02),
                'strategy': 'divergence',
                'strategy_name': 'RSI انحراف صعودي + دعم', 'score': 5
            }
    
    p20h = float(np.max(c[-20:]))
    p40h = float(np.max(c[-40:-20]))
    r20h = float(np.max(rsi_series[-20:]))
    r40h = float(np.max(rsi_series[-40:-20])) if len(rsi_series) >= 40 else r20h
    
    if p20h > p40h and r20h < r40h and rsi_now > 55:
        if (resistance - price) / price * 100 < 0.1:
            return {
                'direction': 'SELL', 'price': price,
                'stop_loss': round(resistance + atr * 0.5, 5),
                'take_profit': round(price - atr * 3.0, 5),
                'confidence': min(0.88, 0.5 + (rsi_now - 55) * 0.02),
                'strategy': 'divergence',
                'strategy_name': 'RSI انحراف هبوطي + مقاومة', 'score': 5
            }
    return None

# ============================================================================
# STRATEGY 2: Breakout Retest
# ============================================================================

def strat_breakout(df, symbol):
    if len(df) < 40: return None
    
    rl, rh, is_range = find_range(df)
    if not is_range: return None
    
    c = df['Close'].values
    h = df['High'].values
    l = df['Low'].values
    price = float(c[-1])
    atr = calc_atr(df)
    
    if float(c[-2]) > rh and abs(float(l[-1]) - rh) / rh < 0.015 and float(c[-1]) >= rh:
        return {
            'direction': 'BUY', 'price': price,
            'stop_loss': round(rl, 5),
            'take_profit': round(price + atr * 3.5, 5),
            'confidence': 0.80, 'strategy': 'breakout_retest',
            'strategy_name': 'اختراق علوي + Retest مؤكد', 'score': 5
        }
    
    if float(c[-2]) < rl and abs(float(h[-1]) - rl) / rl < 0.015 and float(c[-1]) <= rl:
        return {
            'direction': 'SELL', 'price': price,
            'stop_loss': round(rh, 5),
            'take_profit': round(price - atr * 3.5, 5),
            'confidence': 0.80, 'strategy': 'breakout_retest',
            'strategy_name': 'اختراق سفلي + Retest مؤكد', 'score': 5
        }
    return None

# ============================================================================
# STRATEGY 3: EMA Cross
# ============================================================================

def strat_ema(df, symbol):
    if len(df) < 60: return None
    
    ema20 = calc_ema(df, 20)
    ema50 = calc_ema(df, 50)
    df_prev = df.iloc[:-2]
    ema20p = calc_ema(df_prev, 20)
    ema50p = calc_ema(df_prev, 50)
    
    price = float(df['Close'].iloc[-1])
    atr = calc_atr(df)
    rsi = calc_rsi(df)
    dist = abs(price - ema20) / price * 100
    
    if ema20p <= ema50p and ema20 > ema50 and dist < 0.15 and rsi > 40:
        return {
            'direction': 'BUY', 'price': price,
            'stop_loss': round(ema50 - atr, 5),
            'take_profit': round(price + atr * 3.0, 5),
            'confidence': 0.72, 'strategy': 'ema_cross',
            'strategy_name': 'تقاطع EMA صعودي (قريب)', 'score': 4
        }
    
    if ema20p >= ema50p and ema20 < ema50 and dist < 0.15 and rsi < 60:
        return {
            'direction': 'SELL', 'price': price,
            'stop_loss': round(ema50 + atr, 5),
            'take_profit': round(price - atr * 3.0, 5),
            'confidence': 0.72, 'strategy': 'ema_cross',
            'strategy_name': 'تقاطع EMA هبوطي (قريب)', 'score': 4
        }
    return None

# ============================================================================
# STRATEGY 4: Fibo Volume
# ============================================================================

def strat_fibo(df, df_1h, symbol):
    if len(df) < 50 or len(df_1h) < 50: return None
    
    fibo = calc_fibo(df_1h)
    price = float(df['Close'].iloc[-1])
    trend_up = calc_ema(df_1h, 20) > calc_ema(df_1h, 50)
    atr = calc_atr(df)
    rsi = calc_rsi(df)
    vol = get_vol(df)
    
    if trend_up:
        if abs(price - fibo['f618']) / price * 100 < 0.1 and rsi < 45 and vol > 1.2:
            return {
                'direction': 'BUY', 'price': price,
                'stop_loss': round(fibo['f500'], 5),
                'take_profit': round(fibo['high'], 5),
                'confidence': min(0.85, 0.5 + vol * 0.2),
                'strategy': 'fibo_volume',
                'strategy_name': 'فيبوناتشي 61.8% + حجم عالي', 'score': 5
            }
        if abs(price - fibo['f500']) / price * 100 < 0.1 and rsi < 40 and vol > 1.3:
            return {
                'direction': 'BUY', 'price': price,
                'stop_loss': round(fibo['f382'], 5),
                'take_profit': round(fibo['high'], 5),
                'confidence': min(0.88, 0.5 + vol * 0.2),
                'strategy': 'fibo_volume',
                'strategy_name': 'فيبوناتشي 50% + حجم قوي جداً', 'score': 5
            }
    
    if not trend_up:
        if abs(price - fibo['f618']) / price * 100 < 0.1 and rsi > 55 and vol > 1.2:
            return {
                'direction': 'SELL', 'price': price,
                'stop_loss': round(fibo['f500'], 5),
                'take_profit': round(fibo['low'], 5),
                'confidence': min(0.85, 0.5 + vol * 0.2),
                'strategy': 'fibo_volume',
                'strategy_name': 'فيبوناتشي 61.8% + حجم عالي', 'score': 5
            }
    return None

# ============================================================================
# MAIN BOT
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
            w = self.db.get_strategy_weights()
            text = (f"🦅 **Falcon Pro v5.1**\n\n"
                   f"1️⃣ انحراف RSI: {w.get('divergence', 0):.1%}\n"
                   f"2️⃣ اختراق+Retest: {w.get('breakout_retest', 0):.1%}\n"
                   f"3️⃣ تقاطع EMA: {w.get('ema_cross', 0):.1%}\n"
                   f"4️⃣ فيبو+حجم: {w.get('fibo_volume', 0):.1%}")
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def analyze(self, symbol):
        results = []
        df_5m = DataFetcher.fetch(symbol, '5m')
        df_15m = DataFetcher.fetch(symbol, '15m')
        df_1h = DataFetcher.fetch(symbol, '1h')
        
        if df_5m is None or df_15m is None or df_1h is None:
            return results
        
        now = datetime.utcnow()
        if now.weekday() >= 5: return results
        if self.db.has_active_signal(symbol): return results
        if self.db.was_recent(symbol): return results
        
        for strat_fn, strat_key in [(strat_divergence, 'divergence'), 
                                     (strat_breakout, 'breakout_retest'),
                                     (strat_ema, 'ema_cross')]:
            s = strat_fn(df_15m, symbol)
            if s and s['confidence'] >= MIN_CONFIDENCE:
                s['symbol'] = symbol
                s['expiry_time'] = (datetime.now() + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
                results.append(s)
        
        s = strat_fibo(df_5m, df_1h, symbol)
        if s and s['confidence'] >= MIN_CONFIDENCE:
            s['symbol'] = symbol
            s['expiry_time'] = (datetime.now() + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
            results.append(s)
        
        return results
    
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
        logger.info("🔍 بحث...")
        signals = 0
        for symbol in SYMBOLS:
            try:
                for s in self.analyze(symbol):
                    self.db.save_signal(s)
                    self.send_signal(s)
                    signals += 1
                time.sleep(0.3)
            except:
                pass
        if signals > 0:
            logger.info(f"📊 {signals} إشارة")
    
    def send_signal(self, signal):
        emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        direction = "شراء" if signal['direction'] == 'BUY' else "بيع"
        msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
               f"💰 السعر: {signal['price']:.5f}\n"
               f"💪 الثقة: {signal['confidence']:.1%}\n"
               f"📊 {signal['strategy_name']}")
        try:
            self.tb.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
            logger.info(f"✅ {signal['symbol']} {signal['direction']} | {signal['strategy_name']}")
        except:
            pass
    
    def run(self):
        logger.info("🦅 Falcon Pro v5.1")
        
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
    bot = FalconPro()
    bot.run()
