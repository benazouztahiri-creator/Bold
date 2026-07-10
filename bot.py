#!/usr/bin/env python3
"""
Falcon AI v5.1 - Enhanced 4 Strategies
========================================
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

def safe_columns(df: pd.DataFrame) -> pd.DataFrame:
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
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    exit_price REAL, stop_loss REAL, take_profit REAL,
                    confidence REAL, score REAL,
                    strategy TEXT, strategy_name TEXT,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, exit_time DATETIME,
                    result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, pnl_pips REAL,
                    signal_hash TEXT UNIQUE
                );
                
                CREATE TABLE IF NOT EXISTS strategy_performance (
                    strategy TEXT PRIMARY KEY,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0.5
                );
            ''')
            
            for strat in ['divergence', 'breakout_retest', 'ema_cross', 'fibo_volume']:
                conn.execute('''
                    INSERT OR IGNORE INTO strategy_performance (strategy) VALUES (?)
                ''', (strat,))
            
            conn.commit()
    
    def save_signal(self, data: Dict) -> Optional[int]:
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO signals 
                    (symbol, direction, entry_price, stop_loss, take_profit,
                     confidence, score, strategy, strategy_name, expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data.get('stop_loss'), data.get('take_profit'),
                      data['confidence'], data.get('score', 0),
                      data.get('strategy', ''), data.get('strategy_name', ''),
                      data['expiry_time'], h))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except:
            return None
    
    def update_result(self, signal_id: int, exit_price: float, result: str, pnl: float, pips: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE signals SET exit_price=?, result=?, pnl_percent=?, pnl_pips=?,
                exit_time=datetime('now', 'localtime') WHERE id=?
            ''', (exit_price, result, pnl, pips, signal_id))
            
            strategy = conn.execute('SELECT strategy FROM signals WHERE id=?', (signal_id,)).fetchone()[0]
            if strategy:
                conn.execute('''
                    UPDATE strategy_performance 
                    SET total_trades = total_trades + 1,
                        wins = wins + ?,
                        total_pnl = total_pnl + ?
                    WHERE strategy = ?
                ''', (1 if result == 'WIN' else 0, pnl, strategy))
            
            conn.commit()
    
    def has_active_signal(self, symbol: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' 
                AND expiry_time > datetime('now', 'localtime')
            ''', (symbol,)).fetchone()[0]
            return c > 0
    
    def was_recent(self, symbol: str, minutes: int = 10) -> bool:
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?',
                           (symbol, cutoff)).fetchone()[0]
            return c > 0
    
    def get_expired_trades(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM signals WHERE result='PENDING' 
                AND expiry_time <= datetime('now', 'localtime')
            ''').fetchall()
            return [dict(r) for r in rows]
    
    def get_strategy_weights(self) -> Dict:
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
    def fetch(symbol: str, interval: str = '5min') -> Optional[pd.DataFrame]:
        key = f"{symbol}_{interval}"
        cached = data_cache.get(key)
        if cached is not None:
            return cached
        
        try:
            import yfinance as yf
            yf_symbol = symbol if '=X' in symbol else f"{symbol}=X"
            interval_map = {'5m': '5m', '15m': '15m', '1h': '1h', '1m': '1m'}
            yf_interval = interval_map.get(interval, '5m')
            
            df = yf.download(yf_symbol, period='5d', interval=yf_interval, progress=False)
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

def calculate_rsi_series(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    """سلسلة RSI كاملة"""
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

def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float:
    c = df['Close'].values
    delta = np.diff(c)
    gain = np.mean(delta[delta > 0]) if any(delta > 0) else 0
    loss = np.mean(-delta[delta < 0]) if any(delta < 0) else 0
    return round(100 - 100/(1 + gain/(loss+1e-8)), 1) if loss > 0 else 50

def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    h, l, c = df['High'].values, df['Low'].values, df['Close'].values
    tr = np.array([max(h[i+1]-l[i+1], abs(h[i+1]-c[i]), abs(l[i+1]-c[i])) for i in range(len(c)-1)])
    return round(float(np.mean(tr[-period:])), 5)

def calculate_ema(df: pd.DataFrame, period: int) -> float:
    return round(float(pd.Series(df['Close'].values).ewm(span=period, adjust=False).mean().values[-1]), 5)

def find_support_resistance(df: pd.DataFrame, lookback: int = 50) -> Tuple[float, float]:
    h = df['High'].values[-lookback:]
    l = df['Low'].values[-lookback:]
    highs_sorted = np.sort(h)[-5:]
    lows_sorted = np.sort(l)[:5]
    return float(np.mean(lows_sorted)), float(np.mean(highs_sorted))

def find_range(df: pd.DataFrame, lookback: int = 30) -> Tuple[float, float, bool]:
    h = df['High'].values[-lookback:]
    l = df['Low'].values[-lookback:]
    range_high, range_low = float(np.max(h)), float(np.min(l))
    range_size = (range_high - range_low) / range_low * 100
    return range_low, range_high, range_size < 0.3

def calculate_fibonacci(df: pd.DataFrame) -> Dict:
    h = df['High'].values[-50:]
    l = df['Low'].values[-50:]
    swing_high, swing_low = float(np.max(h)), float(np.min(l))
    diff = swing_high - swing_low
    
    return {
        'high': swing_high, 'low': swing_low,
        'fibo_382': round(swing_low + diff * 0.382, 5) if swing_high > swing_low else swing_low,
        'fibo_500': round(swing_low + diff * 0.500, 5) if swing_high > swing_low else swing_low,
        'fibo_618': round(swing_low + diff * 0.618, 5) if swing_high > swing_low else swing_low,
    }

def get_volume_ratio(df: pd.DataFrame) -> float:
    if 'Volume' not in df.columns:
        return 1.0
    v = df['Volume'].values
    if len(v) < 20:
        return 1.0
    return float(v[-1] / (np.mean(v[-20:]) + 1e-8))

# ============================================================================
# STRATEGY 1: RSI Divergence + Support/Resistance
# ============================================================================

def strategy_divergence(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    RSI Divergence مع الدعم والمقاومة
    - Bullish Divergence: السعر قاع أقل + RSI قاع أعلى → شراء
    - Bearish Divergence: السعر قمة أعلى + RSI قمة أقل → بيع
    """
    if len(df) < 50:
        return None
    
    c = df['Close'].values
    rsi_series = calculate_rsi_series(df)
    
    if len(rsi_series) < 20:
        return None
    
    price = float(c[-1])
    support, resistance = find_support_resistance(df)
    atr = calculate_atr(df)
    rsi_now = calculate_rsi(df)
    
    # ✅ Bullish Divergence (شراء)
    # السعر: قاع جديد أقل
    price_20_low = float(np.min(c[-20:]))
    price_40_low = float(np.min(c[-40:-20]))
    
    # RSI: قاع أعلى
    rsi_20_low = float(np.min(rsi_series[-20:]))
    rsi_40_low = float(np.min(rsi_series[-40:-20])) if len(rsi_series) >= 40 else rsi_20_low
    
    if price_20_low < price_40_low and rsi_20_low > rsi_40_low and rsi_now < 45:
        # السعر قريب من الدعم
        distance_to_support = (price - support) / support * 100
        if distance_to_support < 0.1:
            return {
                'direction': 'BUY', 'price': price,
                'stop_loss': round(support - atr * 0.5, 5),
                'take_profit': round(price + atr * 3.0, 5),
                'confidence': min(0.88, 0.5 + (45 - rsi_now) * 0.02),
                'strategy': 'divergence',
                'strategy_name': 'RSI انحراف صعودي + دعم',
                'score': 5
            }
    
    # ✅ Bearish Divergence (بيع)
    price_20_high = float(np.max(c[-20:]))
    price_40_high = float(np.max(c[-40:-20]))
    
    rsi_20_high = float(np.max(rsi_series[-20:]))
    rsi_40_high = float(np.max(rsi_series[-40:-20])) if len(rsi_series) >= 40 else rsi_20_high
    
    if price_20_high > price_40_high and rsi_20_high < rsi_40_high and rsi_now > 55:
        distance_to_resistance = (resistance - price) / price * 100
        if distance_to_resistance < 0.1:
            return {
                'direction': 'SELL', 'price': price,
                'stop_loss': round(resistance + atr * 0.5, 5),
                'take_profit': round(price - atr * 3.0, 5),
                'confidence': min(0.88, 0.5 + (rsi_now - 55) * 0.02),
                'strategy': 'divergence',
                'strategy_name': 'RSI انحراف هبوطي + مقاومة',
                'score': 5
            }
    
    return None

# ============================================================================
# STRATEGY 2: Breakout + Confirmed Retest (Candle Close)
# ============================================================================

def strategy_breakout_retest(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    اختراق مع Retest مؤكد
    - الشمعة تغلق خارج النطاق
    - الشمعة التالية تلامس الخط لكن تقفل داخله → دخول
    """
    if len(df) < 40:
        return None
    
    range_low, range_high, is_ranging = find_range(df)
    
    if not is_ranging:
        return None
    
    c = df['Close'].values
    h = df['High'].values
    l = df['Low'].values
    o = df['Open'].values
    
    price = float(c[-1])
    atr = calculate_atr(df)
    
    # ✅ اختراق علوي + Retest
    # الشمعة قبل الأخيرة: تغلق فوق المقاومة
    prev_close = float(c[-2])
    prev_high = float(h[-2])
    prev_low = float(l[-2])
    
    # الشمعة الحالية: تلامس الخط من فوق لكن تقفل تحته أو عنده
    current_low = float(l[-1])
    current_close = float(c[-1])
    
    if prev_close > range_high:
        # اختراق حقيقي - الشمعة قبل الأخيرة أغلقت فوق المقاومة
        if abs(current_low - range_high) / range_high < 0.015:
            # الـ Retest: الشمعة الحالية لمست الخط
            if current_close >= range_high:
                # أغلقت عند الخط أو فوقه → دخول شراء
                return {
                    'direction': 'BUY', 'price': price,
                    'stop_loss': round(range_low, 5),
                    'take_profit': round(price + atr * 3.5, 5),
                    'confidence': 0.80,
                    'strategy': 'breakout_retest',
                    'strategy_name': 'اختراق علوي + Retest مؤكد',
                    'score': 5
                }
    
    # ✅ اختراق سفلي + Retest
    if prev_close < range_low:
        current_high = float(h[-1])
        if abs(current_high - range_low) / range_low < 0.015:
            if current_close <= range_low:
                return {
                    'direction': 'SELL', 'price': price,
                    'stop_loss': round(range_high, 5),
                    'take_profit': round(price - atr * 3.5, 5),
                    'confidence': 0.80,
                    'strategy': 'breakout_retest',
                    'strategy_name': 'اختراق سفلي + Retest مؤكد',
                    'score': 5
                }
    
    return None

# ============================================================================
# STRATEGY 3: EMA 20/50 Crossover (Price Proximity)
# ============================================================================

def strategy_ema_cross(df: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    تقاطع EMA مع شرط القرب من المتوسطات
    - التقاطع صحيح فقط لو السعر قريب من EMA 20
    """
    if len(df) < 60:
        return None
    
    ema20 = calculate_ema(df, 20)
    ema50 = calculate_ema(df, 50)
    
    df_prev = df.iloc[:-2]
    ema20_prev = calculate_ema(df_prev, 20)
    ema50_prev = calculate_ema(df_prev, 50)
    
    price = float(df['Close'].iloc[-1])
    atr = calculate_atr(df)
    rsi = calculate_rsi(df)
    
    # ✅ المسافة بين السعر و EMA20
    distance_to_ema20 = abs(price - ema20) / price * 100
    
    # ✅ تقاطع صعودي (شراء)
    if ema20_prev <= ema50_prev and ema20_now > ema50_now:
        # السعر لازم يكون قريب من EMA20 (مش بعيد)
        if distance_to_ema20 < 0.15 and rsi > 40:
            return {
                'direction': 'BUY', 'price': price,
                'stop_loss': round(ema50 - atr, 5),
                'take_profit': round(price + atr * 3.0, 5),
                'confidence': 0.72,
                'strategy': 'ema_cross',
                'strategy_name': 'تقاطع EMA صعودي (قريب)',
                'score': 4
            }
    
    # ✅ تقاطع هبوطي (بيع)
    if ema20_prev >= ema50_prev and ema20_now < ema50_now:
        if distance_to_ema20 < 0.15 and rsi < 60:
            return {
                'direction': 'SELL', 'price': price,
                'stop_loss': round(ema50 + atr, 5),
                'take_profit': round(price - atr * 3.0, 5),
                'confidence': 0.72,
                'strategy': 'ema_cross',
                'strategy_name': 'تقاطع EMA هبوطي (قريب)',
                'score': 4
            }
    
    return None

# ============================================================================
# STRATEGY 4: Fibonacci + Volume Confirmation
# ============================================================================

def strategy_fibo_volume(df: pd.DataFrame, df_1h: pd.DataFrame, symbol: str) -> Optional[Dict]:
    """
    فيبوناتشي + تأكيد الحجم
    - التصحيح بحجم ضعيف
    - العودة بحجم قوي
    """
    if len(df) < 50 or len(df_1h) < 50:
        return None
    
    fibo = calculate_fibonacci(df_1h)
    price = float(df['Close'].iloc[-1])
    
    ema20_1h = calculate_ema(df_1h, 20)
    ema50_1h = calculate_ema(df_1h, 50)
    trend_up = ema20_1h > ema50_1h
    
    atr = calculate_atr(df)
    rsi = calculate_rsi(df)
    vol_ratio = get_volume_ratio(df)
    
    # ✅ ترند صاعد + تصحيح لـ 61.8% (شراء)
    if trend_up:
        distance_618 = abs(price - fibo['fibo_618']) / price * 100
        
        # التصحيح بحجم ضعيف + العودة بحجم قوي
        if distance_618 < 0.1 and rsi < 45 and vol_ratio > 1.2:
            return {
                'direction': 'BUY', 'price': price,
                'stop_loss': round(fibo['fibo_500'], 5),
                'take_profit': round(fibo['high'], 5),
                'confidence': min(0.85, 0.5 + vol_ratio * 0.2),
                'strategy': 'fibo_volume',
                'strategy_name': 'فيبوناتشي 61.8% + حجم عالي',
                'score': 5
            }
        
        distance_500 = abs(price - fibo['fibo_500']) / price * 100
        if distance_500 < 0.1 and rsi < 40 and vol_ratio > 1.3:
            return {
                'direction': 'BUY', 'price': price,
                'stop_loss': round(fibo['fibo_382'], 5),
                'take_profit': round(fibo['high'], 5),
                'confidence': min(0.88, 0.5 + vol_ratio * 0.2),
                'strategy': 'fibo_volume',
                'strategy_name': 'فيبوناتشي 50% + حجم قوي جداً',
                'score': 5
            }
    
    # ✅ ترند هابط + تصحيح لـ 61.8% (بيع)
    if not trend_up:
        distance_618 = abs(price - fibo['fibo_618']) / price * 100
        
        if distance_618 < 0.1 and rsi > 55 and vol_ratio > 1.2:
            return {
                'direction': 'SELL', 'price': price,
                'stop_loss': round(fibo['fibo_500'], 5),
                'take_profit': round(fibo['low'], 5),
                'confidence': min(0.85, 0.5 + vol_ratio * 0.2),
                'strategy': 'fibo_volume',
                'strategy_name': 'فيبوناتشي 61.8% + حجم عالي',
                'score': 5
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
            weights = self.db.get_strategy_weights()
            text = (f"🦅 **Falcon Pro v5.1**\n\n"
                   f"📊 4 استراتيجيات محسنة\n"
                   f"1️⃣ انحراف RSI: {weights.get('divergence', 0):.1%}\n"
                   f"2️⃣ اختراق+Retest: {weights.get('breakout_retest', 0):.1%}\n"
                   f"3️⃣ تقاطع EMA: {weights.get('ema_cross', 0):.1%}\n"
                   f"4️⃣ فيبو+حجم: {weights.get('fibo_volume', 0):.1%}")
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def analyze(self, symbol: str) -> List[Dict]:
        results = []
        
        df_5m = DataFetcher.fetch(symbol, '5m')
        df_15m = DataFetcher.fetch(symbol, '15m')
        df_1h = DataFetcher.fetch(symbol, '1h')
        
        if df_5m is None or df_15m is None or df_1h is None:
            return results
        
        now = datetime.utcnow()
        if now.weekday() >= 5:
            return results
        if self.db.has_active_signal(symbol):
            return results
        if self.db.was_recent(symbol):
            return results
        
        strategies = [
            ('divergence', strategy_divergence(df_15m, symbol)),
            ('breakout_retest', strategy_breakout_retest(df_15m, symbol)),
            ('ema_cross', strategy_ema_cross(df_15m, symbol)),
            ('fibo_volume', strategy_fibo_volume(df_5m, df_1h, symbol)),
        ]
        
        for strat_name, signal in strategies:
            if signal and signal['confidence'] >= MIN_CONFIDENCE:
                signal['symbol'] = symbol
                signal['expiry_time'] = (datetime.now() + timedelta(minutes=10)).strftime('%Y-%m-%d %H:%M:%S')
                results.append(signal)
        
        return results
    
    def check_trades(self):
        for trade in self.db.get_expired_trades():
            try:
                df = DataFetcher.fetch(trade['symbol'], '1m')
                if df is None: continue
                
                entry_time = datetime.strptime(trade['entry_time'], '%Y-%m-%d %H:%M:%S')
                expiry_time = datetime.strptime(trade['expiry_time'], '%Y-%m-%d %H:%M:%S')
                
                mask = (df.index >= entry_time) & (df.index <= expiry_time)
                period = df[mask]
                if period.empty: continue
                
                close_p = float(period['Close'].iloc[-1])
                entry = trade['entry_price']
                direction = trade['direction']
                
                is_jpy = "JPY" in trade['symbol']
                pip_value = 0.01 if is_jpy else 0.0001
                
                if direction == 'BUY':
                    pnl = (close_p - entry) / entry * 100
                    pips = (close_p - entry) / pip_value
                    result = 'WIN' if close_p > entry else 'LOSS'
                else:
                    pnl = (entry - close_p) / entry * 100
                    pips = (entry - close_p) / pip_value
                    result = 'WIN' if close_p < entry else 'LOSS'
                
                self.db.update_result(trade['id'], close_p, result, pnl, round(pips, 1))
            except:
                pass
    
    def hunt(self):
        logger.info("🔍 بحث...")
        
        signals_sent = 0
        
        for symbol in SYMBOLS:
            try:
                signals = self.analyze(symbol)
                for signal in signals:
                    self.db.save_signal(signal)
                    self.send_signal(signal)
                    signals_sent += 1
                time.sleep(0.3)
            except:
                pass
        
        if signals_sent > 0:
            logger.info(f"📊 {signals_sent} إشارة")
    
    def send_signal(self, signal: Dict):
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
        logger.info("🦅 Falcon Pro v5.1 - Enhanced Strategies")
        
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
