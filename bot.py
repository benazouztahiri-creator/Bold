#!/usr/bin/env python3
"""
Falcon AI v9.0 - Self-Evolving Trading Strategy
=================================================
The AI discovers its own winning strategy.
No hardcoded rules. Pure reinforcement learning.
"""

import os
import sys
import time
import logging
import sqlite3
import hashlib
import threading
import shutil
import json
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import deque

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from sklearn.preprocessing import RobustScaler
import xgboost as xgb

import telebot
import joblib

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    TELEGRAM_TOKEN: str = os.environ.get('TELEGRAM_TOKEN', '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '7553333305')
    
    SCAN_INTERVAL: int = 60
    
    SYMBOLS: List[str] = field(default_factory=lambda: ['EURUSD=X'])
    
    TRAINING_PERIOD: str = '2mo'
    MIN_TRAINING_SAMPLES: int = 500
    
    DB_PATH: str = 'falcon_v9.db'
    MODELS_DIR: str = 'models_v9'
    STRATEGY_FILE: str = 'evolved_strategy.json'
    
    # ✅ التعلم من النتائج
    LEARNING_RATE: float = 0.1  # سرعة التعلم
    MEMORY_SIZE: int = 100      # آخر 100 صفقة للتقييم
    MIN_WIN_RATE: float = 0.45  # لو نسبة النجاح أقل من كده، يغير الاستراتيجية

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FalconV9')

# ============================================================================
# SELF-EVOLVING STRATEGY
# ============================================================================

class EvolvingStrategy:
    """
    ✅ استراتيجية بتتعلم وتتطور لوحدها
    
    بتراقب نتائج الصفقات وتعدل:
    1. عتبة الثقة (threshold)
    2. مدة الصفقة (duration)
    3. أفضل وقت للتداول (time_filter)
    4. أي الأزواج بيربح أكثر (symbol_weights)
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.memory = deque(maxlen=config.MEMORY_SIZE)
        
        # ✅ قيم البداية
        self.confidence_threshold = 0.60
        self.base_duration = 7
        self.symbol_weights = {s: 1.0 for s in config.SYMBOLS}
        self.best_hours = list(range(24))  # كل الساعات في البداية
        self.avoid_hours = []
        
        # ✅ التعلم من التاريخ
        self.total_trades = 0
        self.total_wins = 0
        self.current_win_rate = 0.5
        self.evolution_generation = 1
        
        self.load()
    
    def add_result(self, trade_data: Dict):
        """✅ أضف نتيجة صفقة للذاكرة"""
        self.memory.append({
            'symbol': trade_data['symbol'],
            'direction': trade_data['direction'],
            'hour': trade_data.get('hour', datetime.now().hour),
            'confidence': trade_data.get('confidence', 0),
            'duration': trade_data.get('trade_duration', 7),
            'pnl': trade_data.get('pnl_percent', 0),
            'result': trade_data.get('result', 'LOSS'),
            'timestamp': datetime.now().isoformat()
        })
        
        self.total_trades += 1
        if trade_data.get('result') == 'WIN':
            self.total_wins += 1
        
        self.current_win_rate = self.total_wins / max(self.total_trades, 1)
        
        # ✅ تطور كل 20 صفقة
        if self.total_trades % 20 == 0 and len(self.memory) >= 10:
            self.evolve()
    
    def evolve(self):
        """
        ✅ طور الاستراتيجية بناءً على النتائج السابقة
        """
        logger.info(f"🧬 تطور الجيل {self.evolution_generation}...")
        
        recent = list(self.memory)[-50:]  # آخر 50 صفقة
        
        # ========== 1. تحسين عتبة الثقة ==========
        wins = [t for t in recent if t['result'] == 'WIN']
        losses = [t for t in recent if t['result'] == 'LOSS']
        
        if wins:
            avg_win_confidence = np.mean([t['confidence'] for t in wins])
            # لو الصفقات الرابحة ثقتها عالية، نرفع العتبة
            if avg_win_confidence > 0.65:
                self.confidence_threshold = min(0.75, self.confidence_threshold + 0.02)
                logger.info(f"  📈 رفع العتبة إلى {self.confidence_threshold:.0%}")
        
        if losses:
            avg_loss_confidence = np.mean([t['confidence'] for t in losses])
            # لو الصفقات الخاسرة ثقتها واطية، نخفض العتبة
            if avg_loss_confidence < 0.55:
                self.confidence_threshold = max(0.50, self.confidence_threshold - 0.02)
                logger.info(f"  📉 خفض العتبة إلى {self.confidence_threshold:.0%}")
        
        # ========== 2. تحسين مدة الصفقة ==========
        if wins:
            avg_win_duration = np.mean([t['duration'] for t in wins])
            self.base_duration = int(avg_win_duration)
            logger.info(f"  ⏱️ تعديل المدة إلى {self.base_duration} دقائق")
        
        # ========== 3. أفضل ساعات التداول ==========
        hour_performance = {}
        for t in recent:
            hour = t['hour']
            if hour not in hour_performance:
                hour_performance[hour] = {'wins': 0, 'total': 0}
            hour_performance[hour]['total'] += 1
            if t['result'] == 'WIN':
                hour_performance[hour]['wins'] += 1
        
        # أفضل 8 ساعات
        sorted_hours = sorted(hour_performance.items(), 
                             key=lambda x: x[1]['wins']/max(x[1]['total'],1), 
                             reverse=True)
        
        self.best_hours = [h for h, _ in sorted_hours[:8]]
        self.avoid_hours = [h for h, _ in sorted_hours[-4:] 
                           if hour_performance[h]['wins']/max(hour_performance[h]['total'],1) < 0.3]
        
        logger.info(f"  🕐 أفضل ساعات: {self.best_hours}")
        if self.avoid_hours:
            logger.info(f"  ⛔ تجنب: {self.avoid_hours}")
        
        # ========== 4. وزن كل زوج ==========
        symbol_performance = {}
        for t in recent:
            sym = t['symbol']
            if sym not in symbol_performance:
                symbol_performance[sym] = {'wins': 0, 'total': 0}
            symbol_performance[sym]['total'] += 1
            if t['result'] == 'WIN':
                symbol_performance[sym]['wins'] += 1
        
        for sym, perf in symbol_performance.items():
            win_rate = perf['wins'] / max(perf['total'], 1)
            self.symbol_weights[sym] = max(0.3, min(2.0, win_rate * 2))
        
        logger.info(f"  ⚖️ أوزان الأزواج: {self.symbol_weights}")
        
        self.evolution_generation += 1
        self.save()
        
        # ✅ إرسال تقرير التطور
        self.send_evolution_report()
    
    def send_evolution_report(self):
        """إرسال تقرير التطور للتليجرام"""
        try:
            import telebot
            tb = telebot.TeleBot(self.config.TELEGRAM_TOKEN)
            
            report = (f"🧬 **تطور الاستراتيجية - جيل #{self.evolution_generation}**\n\n"
                     f"📊 صفقات: {self.total_trades}\n"
                     f"📈 نجاح: {self.current_win_rate:.1%}\n"
                     f"🎯 العتبة: {self.confidence_threshold:.0%}\n"
                     f"⏱️ المدة: {self.base_duration} د\n"
                     f"🕐 الساعات: {self.best_hours[:6]}\n"
                     f"⭐ الأوزان: {dict(list(self.symbol_weights.items())[:4])}")
            
            tb.send_message(self.config.TELEGRAM_CHAT_ID, report, parse_mode='Markdown')
        except:
            pass
    
    def should_trade_now(self, symbol: str) -> Tuple[bool, str]:
        """✅ هل نتداول دلوقتي؟"""
        now = datetime.utcnow()
        
        # فلتر الساعات
        if now.hour in self.avoid_hours:
            return False, f"ساعة تجنب ({now.hour}:00)"
        
        # فلتر الويكند
        if now.weekday() >= 5:
            return False, "ويكند"
        
        # وزن الزوج
        if self.symbol_weights.get(symbol, 1.0) < 0.4:
            return False, "زوج ضعيف"
        
        return True, "مسموح"
    
    def get_dynamic_threshold(self, symbol: str) -> float:
        """✅ عتبة متغيرة حسب أداء الزوج"""
        weight = self.symbol_weights.get(symbol, 1.0)
        # الزوج القوي ← عتبة أقل (فرص أكثر)
        # الزوج الضعيف ← عتبة أعلى (حذر)
        return self.confidence_threshold * (2 - weight)
    
    def get_dynamic_duration(self, symbol: str) -> int:
        """✅ مدة متغيرة"""
        return self.base_duration
    
    def save(self):
        with open(self.config.STRATEGY_FILE, 'w') as f:
            json.dump({
                'confidence_threshold': self.confidence_threshold,
                'base_duration': self.base_duration,
                'symbol_weights': self.symbol_weights,
                'best_hours': self.best_hours,
                'avoid_hours': self.avoid_hours,
                'total_trades': self.total_trades,
                'total_wins': self.total_wins,
                'current_win_rate': self.current_win_rate,
                'evolution_generation': self.evolution_generation,
                'memory': list(self.memory)[-20:]  # آخر 20 صفقة
            }, f, indent=2)
    
    def load(self):
        if os.path.exists(self.config.STRATEGY_FILE):
            with open(self.config.STRATEGY_FILE, 'r') as f:
                data = json.load(f)
                self.confidence_threshold = data.get('confidence_threshold', 0.60)
                self.base_duration = data.get('base_duration', 7)
                self.symbol_weights = data.get('symbol_weights', {})
                self.best_hours = data.get('best_hours', list(range(24)))
                self.avoid_hours = data.get('avoid_hours', [])
                self.total_trades = data.get('total_trades', 0)
                self.total_wins = data.get('total_wins', 0)
                self.current_win_rate = data.get('current_win_rate', 0.5)
                self.evolution_generation = data.get('evolution_generation', 1)
                logger.info(f"📂 تحميل استراتيجية الجيل {self.evolution_generation}")

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with sqlite3.connect(db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    exit_price REAL, confidence REAL,
                    proba_buy REAL, proba_sell REAL,
                    trade_duration INTEGER, trade_hour INTEGER,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, exit_time DATETIME,
                    result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, pnl_pips REAL,
                    strategy_generation INTEGER,
                    signal_hash TEXT UNIQUE
                )
            ''')
            conn.commit()
    
    def save_signal(self, data: Dict) -> Optional[int]:
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO signals 
                    (symbol, direction, entry_price, confidence, proba_buy, proba_sell,
                     trade_duration, trade_hour, expiry_time, strategy_generation, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data.get('proba_buy', 0), data.get('proba_sell', 0),
                      data['trade_duration'], data.get('hour', datetime.now().hour),
                      data['expiry_time'], data.get('strategy_gen', 1), h))
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
            conn.commit()
    
    def has_active_signal(self, symbol: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' 
                AND expiry_time > datetime('now', 'localtime')
            ''', (symbol,)).fetchone()[0]
            return c > 0
    
    def was_recent(self, symbol: str, minutes: int = 5) -> bool:
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?
            ''', (symbol, cutoff)).fetchone()[0]
            return c > 0
    
    def get_expired_trades(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM signals WHERE result='PENDING' 
                AND expiry_time <= datetime('now', 'localtime')
            ''').fetchall()
            return [dict(r) for r in rows]

# ============================================================================
# FEATURES
# ============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l = df['Close'], df['High'], df['Low']
    
    for p in [1, 3, 5, 10]:
        f[f'ret_{p}'] = c.pct_change(p)
    
    for p in [5, 10, 20, 50]:
        f[f'sma_{p}'] = c.rolling(p).mean()
        f[f'dist_{p}'] = (c - f[f'sma_{p}']) / (f[f'sma_{p}'] + 1e-8)
    
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    f['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    f['macd'] = ema12 - ema26
    f['macd_s'] = f['macd'].ewm(span=9).mean()
    f['macd_h'] = f['macd'] - f['macd_s']
    
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f['bb'] = (c - sma20) / (2 * std20 + 1e-8)
    
    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    f['atr'] = tr.ewm(span=14).mean()
    f['atr_pct'] = f['atr'] / (c + 1e-8)
    
    atr14 = tr.ewm(span=14).mean()
    pdm = h.diff().clip(lower=0)
    ndm = (-l.diff()).clip(lower=0)
    pdi = 100 * (pdm.ewm(span=14).mean()) / (atr14 + 1e-8)
    ndi = 100 * (ndm.ewm(span=14).mean()) / (atr14 + 1e-8)
    dx = 100 * abs(pdi - ndi) / (pdi + ndi + 1e-8)
    f['adx'] = dx.ewm(span=14).mean()
    f['pdi'] = pdi
    f['ndi'] = ndi
    
    l14 = l.rolling(14).min()
    h14 = h.rolling(14).max()
    f['stoch'] = 100 * (c - l14) / (h14 - l14 + 1e-8)
    
    for p in [5, 10]:
        f[f'mom_{p}'] = c - c.shift(p)
    
    return f.fillna(0)

# ============================================================================
# TARGET
# ============================================================================

def create_target(df: pd.DataFrame, periods: int = 5) -> pd.Series:
    future = df['Close'].shift(-periods)
    current = df['Close']
    change = (future - current) / current * 100
    
    target = pd.Series(np.nan, index=df.index)
    target[change > 0.03] = 1
    target[change < -0.03] = 0
    
    return target

# ============================================================================
# MODEL
# ============================================================================

class TradingModel:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.model = None
        self.scaler = RobustScaler()
        self.features = []
        self.is_trained = False
    
    def train(self, df: pd.DataFrame) -> bool:
        try:
            from sklearn.feature_selection import mutual_info_classif
            from sklearn.metrics import accuracy_score
            
            X = calculate_features(df)
            y = create_target(df)
            
            valid = ~(X.isna().any(axis=1) | y.isna())
            X, y = X[valid], y[valid]
            
            if len(X) < 200:
                return False
            
            mi = mutual_info_classif(X, y, random_state=42)
            scores = sorted(zip(X.columns, mi), key=lambda x: x[1], reverse=True)
            self.features = [s[0] for s in scores[:20]]
            X = X[self.features]
            
            split = int(len(X) * 0.8)
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]
            
            X_train_s = self.scaler.fit_transform(X_train)
            
            self.model = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.03,
                random_state=42, n_jobs=1, verbosity=0, tree_method='hist'
            )
            self.model.fit(X_train_s, y_train)
            
            X_val_s = self.scaler.transform(X_val)
            val_pred = self.model.predict(X_val_s)
            acc = accuracy_score(y_val, val_pred)
            
            logger.info(f"✅ {self.symbol}: دقة={acc:.1%}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"❌ {self.symbol}: {e}")
            return False
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.60) -> Tuple[str, float, float, float]:
        if not self.is_trained:
            return "NEUTRAL", 0.0, 0.5, 0.5
        
        try:
            X = calculate_features(df).iloc[[-1]]
            available = [f for f in self.features if f in X.columns]
            
            if len(available) < 8:
                return "NEUTRAL", 0.0, 0.5, 0.5
            
            X = X[available].fillna(0)
            X_s = self.scaler.transform(X)
            
            probas = self.model.predict_proba(X_s)[0]
            proba_sell = float(probas[0])
            proba_buy = float(probas[1])
            
            if proba_buy > threshold:
                return "BUY", proba_buy, proba_buy, proba_sell
            elif proba_sell > threshold:
                return "SELL", proba_sell, proba_buy, proba_sell
            else:
                return "NEUTRAL", max(proba_buy, proba_sell), proba_buy, proba_sell
                
        except:
            return "NEUTRAL", 0.0, 0.5, 0.5
    
    def save(self):
        os.makedirs(f"models_v9/{self.symbol}", exist_ok=True)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'features': self.features
        }, f"models_v9/{self.symbol}/model.pkl")
    
    def load(self) -> bool:
        path = f"models_v9/{self.symbol}/model.pkl"
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.model = data['model']
        self.scaler = data['scaler']
        self.features = data['features']
        self.is_trained = True
        return True

# ============================================================================
# MAIN BOT
# ============================================================================

class FalconV9:
    def __init__(self, config: Config):
        self.config = config
        self.db = Database(config.DB_PATH)
        self.strategy = EvolvingStrategy(config)
        self.models = {}
        
        self.tb = telebot.TeleBot(config.TELEGRAM_TOKEN)
        self._setup_bot()
        
        if os.path.exists(config.MODELS_DIR):
            shutil.rmtree(config.MODELS_DIR)
        
        for symbol in config.SYMBOLS:
            df = self.fetch_data(symbol, '15m', config.TRAINING_PERIOD)
            if df is not None:
                model = TradingModel(symbol)
                if model.train(df):
                    model.save()
                    self.models[symbol] = model
    
    def _setup_bot(self):
        try:
            requests.get(f'https://api.telegram.org/bot{self.config.TELEGRAM_TOKEN}/deleteWebhook', timeout=3)
        except:
            pass
        
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            trained = sum(1 for m in self.models.values() if m.is_trained)
            text = (f"🦅 **Falcon V9.0**\n"
                   f"✅ نماذج: {trained}/{len(self.config.SYMBOLS)}\n"
                   f"🧬 الجيل: {self.strategy.evolution_generation}\n"
                   f"📊 صفقات: {self.strategy.total_trades}\n"
                   f"📈 نجاح: {self.strategy.current_win_rate:.1%}\n"
                   f"🎯 العتبة: {self.strategy.confidence_threshold:.0%}\n"
                   f"⏱️ المدة: {self.strategy.base_duration} د")
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def fetch_data(self, symbol: str, interval: str = '5m', period: str = '5d') -> Optional[pd.DataFrame]:
        for _ in range(3):
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                if not df.empty:
                    df.columns = [c.capitalize() for c in df.columns]
                    return df
            except:
                time.sleep(3)
        return None
    
    def analyze(self, symbol: str) -> Optional[Dict]:
        # ✅ الاستراتيجية تقرر
        should_trade, reason = self.strategy.should_trade_now(symbol)
        if not should_trade:
            logger.debug(f"⛔ {symbol}: {reason}")
            return None
        
        model = self.models.get(symbol)
        if not model or not model.is_trained:
            return None
        
        if self.db.has_active_signal(symbol):
            return None
        
        if self.db.was_recent(symbol):
            return None
        
        df_5m = self.fetch_data(symbol, '5m', '3d')
        df_15m = self.fetch_data(symbol, '15m', '5d')
        
        if df_5m is None or df_15m is None:
            return None
        
        # ✅ عتبة ديناميكية
        threshold = self.strategy.get_dynamic_threshold(symbol)
        
        dir_5m, conf_5m, pb_5m, ps_5m = model.predict(df_5m, threshold)
        dir_15m, conf_15m, pb_15m, ps_15m = model.predict(df_15m, threshold)
        
        if dir_5m != dir_15m or dir_5m == "NEUTRAL":
            return None
        
        confidence = (conf_5m + conf_15m) / 2
        
        if confidence < threshold:
            return None
        
        # ✅ مدة ديناميكية
        duration = self.strategy.get_dynamic_duration(symbol)
        entry = float(df_5m['Close'].iloc[-1])
        
        return {
            'symbol': symbol,
            'direction': dir_5m,
            'entry_price': entry,
            'confidence': confidence,
            'proba_buy': (pb_5m + pb_15m) / 2,
            'proba_sell': (ps_5m + ps_15m) / 2,
            'trade_duration': duration,
            'hour': datetime.now().hour,
            'strategy_gen': self.strategy.evolution_generation,
            'expiry_time': (datetime.now() + timedelta(minutes=duration)).strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def send_signal(self, signal: Dict):
        emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
        
        msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
               f"💰 {signal['entry_price']:.5f}\n"
               f"⏳ {signal['trade_duration']} د\n"
               f"💪 {signal['confidence']:.1%}\n"
               f"🧬 جيل {signal['strategy_gen']}\n\n"
               f"🤖 Falcon V9")
        
        try:
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
        except:
            pass
    
    def check_trades(self):
        for trade in self.db.get_expired_trades():
            try:
                df = self.fetch_data(trade['symbol'], '5m', '1d')
                if df is None:
                    continue
                
                current = float(df['Close'].iloc[-1])
                entry = trade['entry_price']
                direction = trade['direction']
                
                pip_value = 0.01 if 'JPY' in trade['symbol'] else 0.0001
                
                if direction == 'BUY':
                    pnl = (current - entry) / entry * 100
                    pips = (current - entry) / pip_value
                    result = 'WIN' if current > entry else 'LOSS'
                else:
                    pnl = (entry - current) / entry * 100
                    pips = (entry - current) / pip_value
                    result = 'WIN' if current < entry else 'LOSS'
                
                self.db.update_result(trade['id'], current, result, pnl, round(pips, 1))
                
                # ✅ الاستراتيجية تتعلم
                trade['pnl_percent'] = pnl
                trade['result'] = result
                self.strategy.add_result(trade)
                
            except:
                pass
    
    def scan(self):
        for symbol in self.config.SYMBOLS:
            try:
                signal = self.analyze(symbol)
                if signal and self.db.save_signal(signal):
                    self.send_signal(signal)
                time.sleep(1)
            except:
                pass
    
    def run(self):
        logger.info("🦅 Falcon V9.0 - استراتيجية ذاتية التطور")
        logger.info(f"🧬 الجيل الحالي: {self.strategy.evolution_generation}")
        logger.info(f"🎯 العتبة: {self.strategy.confidence_threshold:.0%}")
        
        def poll():
            while True:
                try:
                    self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except:
                    time.sleep(5)
        threading.Thread(target=poll, daemon=True).start()
        time.sleep(1)
        
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID,
                f"🦅 **Falcon V9.0**\n"
                f"✅ {trained}/{len(self.config.SYMBOLS)}\n"
                f"🧬 استراتيجية ذاتية التطور\n"
                f"⚡️ يعمل...",
                parse_mode='Markdown')
        except:
            pass
        
        while True:
            try:
                self.check_trades()
                self.scan()
                time.sleep(self.config.SCAN_INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"حلقة: {e}")
                time.sleep(10)

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    config = Config()
    bot = FalconV9(config)
    bot.run()
