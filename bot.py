#!/usr/bin/env python3
"""
Falcon AI v10 - Born to Learn
================================
Phase 1: Gather Data
Phase 2: Train Models
Phase 3: Backtest Strategies
Phase 4: Evolve Brain
Phase 5: Trade Smart
"""

import os
import sys
import time
import json
import logging
import sqlite3
import hashlib
import threading
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd
import requests

# ML
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

import telebot
import joblib

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FalconV10')

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    def __init__(self, db_path: str = 'falcon_v10.db'):
        self.db_path = db_path
        with sqlite3.connect(db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS market_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, timestamp DATETIME,
                    open REAL, high REAL, low REAL, close REAL,
                    UNIQUE(symbol, timestamp)
                );
                
                CREATE TABLE IF NOT EXISTS strategies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT, model_type TEXT, params TEXT,
                    win_rate REAL, total_trades INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS backtests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_id INTEGER, symbol TEXT,
                    total_trades INTEGER, wins INTEGER, losses INTEGER,
                    win_rate REAL, total_pnl REAL, sharpe_ratio REAL,
                    tested_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    exit_price REAL, confidence REAL,
                    strategy_id INTEGER, duration INTEGER,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME,
                    result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL
                );
                
                CREATE INDEX IF NOT EXISTS idx_market_data ON market_data(symbol, timestamp);
            ''')
            conn.commit()

# ============================================================================
# PHASE 1: DATA GATHERER
# ============================================================================

class DataGatherer:
    """يجمع البيانات من Alpha Vantage + Yahoo"""
    
    def __init__(self, db: Database, av_key: str = '5TFFWK21CUNA3P25'):
        self.db = db
        self.av_key = av_key
        self.symbols = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD',
            'USDCAD', 'NZDUSD', 'EURGBP', 'EURJPY',
            'GBPJPY', 'EURCHF', 'USDCHF', 'AUDJPY'
        ]
    
    def fetch_all(self):
        """جمع بيانات كل الأزواج"""
        logger.info("📥 Phase 1: جمع البيانات...")
        
        for symbol in self.symbols:
            self._fetch_symbol(symbol)
            time.sleep(12)  # Alpha Vantage حد 5/دقيقة
        
        logger.info("✅ Phase 1: اكتمل")
    
    def _fetch_symbol(self, symbol: str):
        """جلب بيانات زوج واحد"""
        from_curr = symbol[:3]
        to_curr = symbol[3:]
        
        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': from_curr,
            'to_symbol': to_curr,
            'interval': '15min',
            'outputsize': 'full',
            'apikey': self.av_key
        }
        
        try:
            r = requests.get('https://www.alphavantage.co/query', params=params, timeout=15)
            data = r.json()
            
            key = 'Time Series FX (15min)'
            if key not in data:
                return
            
            count = 0
            with sqlite3.connect(self.db.db_path) as conn:
                for ts, vals in data[key].items():
                    conn.execute('''
                        INSERT OR IGNORE INTO market_data (symbol, timestamp, open, high, low, close)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (symbol, ts, float(vals['1. open']), float(vals['2. high']),
                          float(vals['3. low']), float(vals['4. close'])))
                    count += 1
                conn.commit()
            
            logger.info(f"  📥 {symbol}: {count} صف")
            
        except Exception as e:
            logger.error(f"  ❌ {symbol}: {e}")

# ============================================================================
# PHASE 2: TRAINER
# ============================================================================

class Trainer:
    """يدرب نماذج مختلفة"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def get_data(self, symbol: str, limit: int = 5000) -> Optional[pd.DataFrame]:
        with sqlite3.connect(self.db.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM market_data WHERE symbol=?
                ORDER BY timestamp DESC LIMIT ?
            ''', conn, params=(symbol, limit))
        
        if df.empty:
            return None
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp').sort_index()
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        f = pd.DataFrame(index=df.index)
        c, h, l = df['close'], df['high'], df['low']
        
        for p in [1, 3, 5, 10]:
            f[f'ret_{p}'] = c.pct_change(p)
        for p in [5, 10, 20]:
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
        
        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()
        f['bb'] = (c - sma20) / (2 * std20 + 1e-8)
        
        tr1 = h - l
        tr2 = abs(h - c.shift())
        tr3 = abs(l - c.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        f['atr'] = tr.ewm(span=14).mean()
        
        return f.fillna(0)
    
    def create_target(self, df: pd.DataFrame, periods: int = 5) -> pd.Series:
        future = df['close'].shift(-periods)
        change = (future - df['close']) / df['close'] * 100
        target = pd.Series(np.nan, index=df.index)
        target[change > 0.03] = 1
        target[change < -0.03] = 0
        return target
    
    def train_models(self, symbol: str) -> List[Dict]:
        """تدريب عدة نماذج"""
        df = self.get_data(symbol)
        if df is None or len(df) < 500:
            return []
        
        X = self.create_features(df)
        y = self.create_target(df)
        
        valid = ~(X.isna().any(axis=1) | y.isna())
        X, y = X[valid], y[valid]
        
        if len(X) < 200:
            return []
        
        # تقسيم
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        
        models = []
        
        # 1. XGBoost
        xgb_model = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05,
                                        random_state=42, verbosity=0, tree_method='hist')
        xgb_model.fit(X_train_s, y_train)
        xgb_acc = (xgb_model.predict(X_test_s) == y_test).mean()
        models.append({'name': 'XGBoost', 'model': xgb_model, 'scaler': scaler, 
                       'features': list(X.columns), 'accuracy': xgb_acc})
        
        # 2. RandomForest
        rf_model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1)
        rf_model.fit(X_train_s, y_train)
        rf_acc = (rf_model.predict(X_test_s) == y_test).mean()
        models.append({'name': 'RandomForest', 'model': rf_model, 'scaler': scaler,
                       'features': list(X.columns), 'accuracy': rf_acc})
        
        # 3. GradientBoosting
        gb_model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
        gb_model.fit(X_train_s, y_train)
        gb_acc = (gb_model.predict(X_test_s) == y_test).mean()
        models.append({'name': 'GradientBoost', 'model': gb_model, 'scaler': scaler,
                       'features': list(X.columns), 'accuracy': gb_acc})
        
        logger.info(f"  🎓 {symbol}: XGB={xgb_acc:.1%} RF={rf_acc:.1%} GB={gb_acc:.1%}")
        
        return models

# ============================================================================
# PHASE 3: BACKTESTER
# ============================================================================

class Backtester:
    """يختبر الاستراتيجيات على بيانات تاريخية"""
    
    def __init__(self, db: Database):
        self.db = db
    
    def backtest_model(self, model_data: Dict, df: pd.DataFrame, 
                       duration_minutes: int = 7, threshold: float = 0.55) -> Dict:
        """اختبار نموذج على بيانات"""
        model = model_data['model']
        scaler = model_data['scaler']
        features = model_data['features']
        
        trainer = Trainer(self.db)
        X = trainer.create_features(df)
        X = X[features].fillna(0)
        
        results = []
        step = 10  # كل 10 شمعات
        
        for i in range(0, len(X) - 20, step):
            X_s = scaler.transform(X.iloc[[i]])
            proba = model.predict_proba(X_s)[0]
            
            proba_buy = proba[1]
            proba_sell = proba[0]
            
            direction = None
            if proba_buy > threshold:
                direction = 'BUY'
            elif proba_sell > threshold:
                direction = 'SELL'
            
            if direction:
                entry = float(df['close'].iloc[i])
                # بعد المدة
                exit_idx = min(i + duration_minutes * 3, len(df) - 1)  # 15min candles
                exit_price = float(df['close'].iloc[exit_idx])
                
                if direction == 'BUY':
                    pnl = (exit_price - entry) / entry * 100
                    win = exit_price > entry
                else:
                    pnl = (entry - exit_price) / entry * 100
                    win = exit_price < entry
                
                results.append({'win': win, 'pnl': pnl})
        
        if not results:
            return {'total': 0, 'wins': 0, 'win_rate': 0, 'total_pnl': 0}
        
        wins = sum(1 for r in results if r['win'])
        total = len(results)
        total_pnl = sum(r['pnl'] for r in results)
        
        return {
            'total': total, 'wins': wins, 'losses': total - wins,
            'win_rate': wins / total, 'total_pnl': total_pnl,
            'avg_pnl': total_pnl / total
        }

# ============================================================================
# PHASE 4: BRAIN
# ============================================================================

class Brain:
    """يتعلم ويختار أفضل استراتيجية"""
    
    def __init__(self, db: Database):
        self.db = db
        self.best_model = None
        self.best_threshold = 0.55
        self.best_duration = 7
    
    def evolve(self, trainer: Trainer, backtester: Backtester):
        """تعلم واختبر كل الاستراتيجيات"""
        logger.info("🧠 Phase 4: التعلم...")
        
        best_result = {'win_rate': 0}
        best_config = None
        
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY']
        
        for symbol in symbols:
            df = trainer.get_data(symbol)
            if df is None:
                continue
            
            models = trainer.train_models(symbol)
            
            for model_data in models:
                for duration in [5, 7, 10]:
                    for threshold in [0.52, 0.55, 0.58, 0.60]:
                        result = backtester.backtest_model(
                            model_data, df, duration, threshold
                        )
                        
                        if result['total'] >= 20 and result['win_rate'] > best_result['win_rate']:
                            best_result = result
                            best_config = {
                                'model': model_data,
                                'duration': duration,
                                'threshold': threshold,
                                'symbol': symbol
                            }
                            
                            logger.info(f"  ⭐ {symbol} {model_data['name']}: "
                                      f"{result['win_rate']:.1%} | {duration}د | {threshold:.0%}")
        
        if best_config:
            self.best_model = best_config['model']
            self.best_duration = best_config['duration']
            self.best_threshold = best_config['threshold']
            
            logger.info(f"🏆 أفضل استراتيجية: {best_config['symbol']} "
                       f"{best_config['model']['name']} "
                       f"نسبة={best_result['win_rate']:.1%}")
        
        return best_config is not None

# ============================================================================
# PHASE 5: TRADER
# ============================================================================

class Trader:
    """يرسل الإشارات"""
    
    def __init__(self, db: Database, brain: Brain, token: str, chat_id: str):
        self.db = db
        self.brain = brain
        self.tb = telebot.TeleBot(token)
        self.chat_id = chat_id
        self._setup_bot()
    
    def _setup_bot(self):
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != self.chat_id:
                return
            
            if self.brain.best_model:
                text = (f"🦅 **Falcon V10**\n\n"
                       f"🧠 العقل: نشط\n"
                       f"⭐ النموذج: {self.brain.best_model['name']}\n"
                       f"🎯 العتبة: {self.brain.best_threshold:.0%}\n"
                       f"⏱️ المدة: {self.brain.best_duration} د")
            else:
                text = "🦅 **Falcon V10**\n\n🧠 العقل: يحتاج تدريب\n📥 اجمع البيانات أولاً"
            
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def analyze(self, symbol: str) -> Optional[Dict]:
        if self.brain.best_model is None:
            return None
        
        # جلب بيانات حية
        from_curr = symbol[:3]
        to_curr = symbol[3:]
        
        params = {
            'function': 'FX_INTRADAY',
            'from_symbol': from_curr,
            'to_symbol': to_curr,
            'interval': '5min',
            'outputsize': 'compact',
            'apikey': '5TFFWK21CUNA3P25'
        }
        
        try:
            r = requests.get('https://www.alphavantage.co/query', params=params, timeout=10)
            data = r.json()
            
            key = 'Time Series FX (5min)'
            if key not in data:
                return None
            
            records = []
            for ts, vals in data[key].items():
                records.append({
                    'timestamp': ts, 'open': float(vals['1. open']),
                    'high': float(vals['2. high']), 'low': float(vals['3. low']),
                    'close': float(vals['4. close'])
                })
            
            df = pd.DataFrame(records)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp').sort_index()
            
            # تحليل
            trainer = Trainer(self.db)
            X = trainer.create_features(df).iloc[[-1]]
            features = self.brain.best_model['features']
            available = [f for f in features if f in X.columns]
            
            if len(available) < 5:
                return None
            
            X = X[available].fillna(0)
            X_s = self.brain.best_model['scaler'].transform(X)
            
            proba = self.brain.best_model['model'].predict_proba(X_s)[0]
            
            if proba[1] > self.brain.best_threshold:
                direction = 'BUY'
                confidence = proba[1]
            elif proba[0] > self.brain.best_threshold:
                direction = 'SELL'
                confidence = proba[0]
            else:
                return None
            
            entry = float(df['close'].iloc[-1])
            
            return {
                'symbol': symbol, 'direction': direction,
                'entry_price': entry, 'confidence': confidence,
                'duration': self.brain.best_duration,
                'expiry_time': (datetime.now() + timedelta(minutes=self.brain.best_duration)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except:
            return None
    
    def send_signal(self, signal: Dict):
        emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
        
        msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
               f"💰 {signal['entry_price']:.5f}\n"
               f"⏳ {signal['duration']} د\n"
               f"💪 {signal['confidence']:.1%}\n\n"
               f"🤖 Falcon V10")
        
        try:
            self.tb.send_message(self.chat_id, msg, parse_mode='Markdown')
            logger.info(f"✅ {signal['symbol']} {signal['direction']}")
        except:
            pass
    
    def scan(self, symbols: List[str]):
        for symbol in symbols:
            try:
                signal = self.analyze(symbol)
                if signal:
                    self.send_signal(signal)
                time.sleep(12)
            except:
                pass
    
    def run_polling(self):
        def poll():
            while True:
                try:
                    self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except:
                    time.sleep(5)
        threading.Thread(target=poll, daemon=True).start()

# ============================================================================
# MAIN
# ============================================================================

class FalconV10:
    def __init__(self):
        self.db = Database()
        self.gatherer = DataGatherer(self.db)
        self.trainer = Trainer(self.db)
        self.backtester = Backtester(self.db)
        self.brain = Brain(self.db)
        self.trader = Trader(
            self.db, self.brain,
            '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk',
            '7553333305'
        )
    
    def run_phases(self):
        """تشغيل كل المراحل"""
        
        # Phase 1: جمع البيانات
        logger.info("=" * 40)
        logger.info("🦅 Falcon V10 - Born to Learn")
        logger.info("=" * 40)
        
        self.gatherer.fetch_all()
        
        # Phase 2+3+4: تدريب واختبار وتعلم
        logger.info("🧠 Phase 2-4: تدريب واختبار...")
        success = self.brain.evolve(self.trainer, self.backtester)
        
        if success:
            logger.info("🏆 العقل جاهز للتداول!")
        else:
            logger.warning("⚠️ العقل محتاج بيانات أكتر")
        
        # Phase 5: بدء التداول
        logger.info("📡 Phase 5: بدء المراقبة...")
        self.trader.run_polling()
        
        symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        try:
            self.trader.tb.send_message(
                self.trader.chat_id,
                f"🦅 **Falcon V10**\n\n"
                f"📥 البيانات: جاهزة\n"
                f"🧠 العقل: {'نشط' if self.brain.best_model else 'يحتاج تدريب'}\n"
                f"⚡️ المراقبة بدأت...",
                parse_mode='Markdown'
            )
        except:
            pass
        
        while True:
            try:
                self.trader.scan(symbols)
                time.sleep(60)
            except KeyboardInterrupt:
                break
            except:
                time.sleep(30)

if __name__ == "__main__":
    bot = FalconV10()
    bot.run_phases()
