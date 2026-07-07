#!/usr/bin/env python3
"""
Falcon AI Pro v4.0 - No LightGBM
=================================
XGBoost + CatBoost + RandomForest + GradientBoosting
"""

import os
import sys
import time
import json
import logging
import sqlite3
import hashlib
import warnings
import threading
import gc
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.max_open_warning'] = 0

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from catboost import CatBoostClassifier

import telebot
import joblib

warnings.filterwarnings('ignore')
os.environ['OMP_NUM_THREADS'] = '2'

# ============================================================================
# CONFIG
# ============================================================================

@dataclass
class Config:
    TELEGRAM_TOKEN: str = os.environ.get('TELEGRAM_TOKEN', '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk')
    TELEGRAM_CHAT_ID: str = os.environ.get('TELEGRAM_CHAT_ID', '7553333305')
    
    TRADE_DURATION_MINUTES: int = 10
    SCAN_INTERVAL_MINUTES: int = 3
    
    SYMBOLS: List[str] = field(default_factory=lambda: [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
        'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X'
    ])
    
    TRAINING_PERIOD_1H: str = '6mo'
    TRAINING_PERIOD_15M: str = '1mo'
    
    WALK_FORWARD_WINDOWS: int = 3
    MIN_TRAINING_SAMPLES: int = 500
    
    CONFIDENCE_THRESHOLD: float = 0.60
    DYNAMIC_THRESHOLD_MIN: float = 0.55
    DYNAMIC_THRESHOLD_MAX: float = 0.75
    
    RETRAINING_INTERVAL_HOURS: int = 24
    MAX_FEATURES: int = 60
    FORECAST_PERIODS: int = 5
    
    DB_PATH: str = 'falcon_trading.db'
    MODELS_DIR: str = 'models'
    
    MAX_RETRIES: int = 5
    RETRY_DELAY: int = 10
    MAX_WORKERS: int = 2
    SIGNAL_COOLDOWN_MINUTES: int = 10
    
    LOG_FILE: str = 'falcon_bot.log'

# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(config: Config) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger('FalconPro')

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    def __init__(self, db_path: str, logger: logging.Logger):
        self.db_path = db_path
        self.logger = logger
        self._init()
    
    def _init(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    confidence REAL, regime TEXT, entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, meta_proba REAL, signal_hash TEXT UNIQUE
                );
                CREATE TABLE IF NOT EXISTS performance (
                    symbol TEXT PRIMARY KEY,
                    last_50_wins INTEGER DEFAULT 0,
                    last_50_total INTEGER DEFAULT 0,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            conn.commit()
    
    def save_signal(self, data: Dict) -> Optional[int]:
        try:
            hash_str = f"{data['symbol']}_{data['direction']}_{datetime.now().timestamp()}"
            signal_hash = hashlib.md5(hash_str.encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO signals 
                    (symbol, direction, entry_price, confidence, regime, expiry_time, meta_proba, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data.get('regime', ''), data['expiry_time'],
                      data.get('meta_proba', 0), signal_hash))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except:
            return None
    
    def update_result(self, signal_id: int, exit_price: float, result: str, pnl: float, symbol: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('UPDATE signals SET exit_price=?, result=?, pnl_percent=? WHERE id=?',
                        (exit_price, result, pnl, signal_id))
            conn.execute('''
                INSERT INTO performance (symbol, last_50_wins, last_50_total)
                VALUES (?, ?, 1)
                ON CONFLICT(symbol) DO UPDATE SET
                last_50_wins = last_50_wins + ?,
                last_50_total = last_50_total + 1
            ''', (symbol, 1 if result == 'WIN' else 0, 1 if result == 'WIN' else 0))
            conn.commit()
    
    def get_dynamic_threshold(self, symbol: str) -> float:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute('SELECT last_50_wins, last_50_total FROM performance WHERE symbol=?',
                              (symbol,)).fetchone()
            if not row or row[1] < 10:
                return 0.65
            win_rate = row[0] / row[1]
            if win_rate > 0.70: return 0.58
            elif win_rate > 0.60: return 0.62
            elif win_rate > 0.50: return 0.68
            else: return 0.75
    
    def check_active_signal(self, symbol: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' 
                AND expiry_time > datetime('now', 'localtime')
            ''', (symbol,)).fetchone()[0]
            return count > 0
    
    def check_recent_signal(self, symbol: str, minutes: int) -> bool:
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            count = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?
            ''', (symbol, cutoff)).fetchone()[0]
            return count > 0
    
    def get_pending_trades(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM signals WHERE result='PENDING' 
                AND expiry_time <= datetime('now', 'localtime')
            ''').fetchall()
            return [dict(r) for r in rows]
    
    def get_stats(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM signals WHERE result!='PENDING'").fetchone()[0]
            wins = conn.execute("SELECT COUNT(*) FROM signals WHERE result='WIN'").fetchone()[0]
            return {'total': total, 'wins': wins, 'losses': total-wins,
                    'win_rate': wins/total if total > 0 else 0}

# ============================================================================
# MARKET REGIME
# ============================================================================

class MarketRegime:
    @staticmethod
    def detect(df: pd.DataFrame) -> Dict:
        if len(df) < 50:
            return {'regime': 'UNKNOWN', 'adx': 0, 'is_trend': False}
        
        c, h, l = df['Close'], df['High'], df['Low']
        
        tr1 = h - l
        tr2 = abs(h - c.shift())
        tr3 = abs(l - c.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=14).mean()
        
        plus_dm = h.diff().clip(lower=0)
        minus_dm = (-l.diff()).clip(lower=0)
        plus_di = 100 * (plus_dm.ewm(span=14).mean()) / (atr + 1e-8)
        minus_di = 100 * (minus_dm.ewm(span=14).mean()) / (atr + 1e-8)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
        adx = float(dx.ewm(span=14).mean().iloc[-1])
        
        current_vol = c.pct_change().rolling(20).std().iloc[-1]
        historical_vol = c.pct_change().rolling(100).std().iloc[-1] if len(c) >= 100 else current_vol
        vol_ratio = current_vol / (historical_vol + 1e-8)
        
        if adx > 25:
            regime = 'TREND_HIGH_VOL' if vol_ratio > 1.3 else 'TREND_LOW_VOL'
        else:
            regime = 'RANGE_HIGH_VOL' if vol_ratio > 1.3 else 'RANGE_LOW_VOL'
        
        return {'regime': regime, 'adx': round(adx, 1), 'is_trend': adx > 25}

# ============================================================================
# FEATURES (60+)
# ============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l, o = df['Close'], df['High'], df['Low'], df['Open']
    v = df.get('Volume', pd.Series(1, index=df.index))
    
    for p in [1, 3, 5, 10, 20]:
        f[f'ret_{p}'] = c.pct_change(p)
    
    for p in [5, 10, 20, 50, 100, 200]:
        if len(df) >= p:
            f[f'sma_{p}'] = c.rolling(p).mean()
            f[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
            f[f'dist_sma_{p}'] = (c - f[f'sma_{p}']) / (f[f'sma_{p}'] + 1e-8)
    
    for p in [7, 14, 21]:
        delta = c.diff()
        gain = delta.where(delta > 0, 0.0).rolling(p).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(p).mean()
        f[f'rsi_{p}'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    f['macd'] = ema12 - ema26
    f['macd_signal'] = f['macd'].ewm(span=9).mean()
    f['macd_hist'] = f['macd'] - f['macd_signal']
    
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f['bb_pos'] = (c - sma20) / (2 * std20 + 1e-8)
    f['bb_width'] = (4 * std20) / (sma20 + 1e-8)
    
    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    f['atr'] = tr.ewm(span=14).mean()
    f['atr_pct'] = f['atr'] / (c + 1e-8)
    
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    f['stoch_k'] = 100 * (c - low14) / (high14 - low14 + 1e-8)
    f['stoch_d'] = f['stoch_k'].rolling(3).mean()
    
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    f['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-8)
    
    hh14 = h.rolling(14).max()
    ll14 = l.rolling(14).min()
    f['williams_r'] = -100 * (hh14 - c) / (hh14 - ll14 + 1e-8)
    
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    atr14 = tr.ewm(span=14).mean()
    plus_di = 100 * (plus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    minus_di = 100 * (minus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    f['adx'] = dx.ewm(span=14).mean()
    f['di_plus'] = plus_di
    f['di_minus'] = minus_di
    
    for p in [5, 10, 20]:
        f[f'roc_{p}'] = (c - c.shift(p)) / (c.shift(p) + 1e-8) * 100
        f[f'mom_{p}'] = c - c.shift(p)
    
    for p in [5, 10, 20]:
        f[f'vol_{p}'] = c.pct_change().rolling(p).std()
    
    f['dc_upper'] = h.rolling(20).max()
    f['dc_lower'] = l.rolling(20).min()
    f['dc_pos'] = (c - f['dc_lower']) / (f['dc_upper'] - f['dc_lower'] + 1e-8)
    
    f['body_size'] = abs(c - o) / (h - l + 1e-8)
    f['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-8)
    f['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-8)
    
    f['trend_str'] = c.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    
    f['vol_ratio'] = v / (v.rolling(20).mean() + 1e-8)
    f['vol_trend'] = v.rolling(5).mean() / (v.rolling(20).mean() + 1e-8)
    
    return f.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# ============================================================================
# SMART TARGET
# ============================================================================

def create_target(df: pd.DataFrame, periods: int) -> pd.Series:
    atr = (df['High'] - df['Low']).rolling(14).mean()
    future = df['Close'].shift(-periods)
    change = future - df['Close']
    threshold = atr * 0.5
    
    target = pd.Series(np.nan, index=df.index)
    target[change > threshold] = 1
    target[change < -threshold] = 0
    return target

# ============================================================================
# ENSEMBLE MODEL (3 models + Meta)
# ============================================================================

class EnsembleModel:
    def __init__(self, symbol: str, config: Config, logger: logging.Logger):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        
        self.base_models = {}
        self.meta_model = None
        self.calibrators = {}
        self.scaler = RobustScaler()
        self.selected_features = []
        self.is_trained = False
        self.version = None
        self.wf_score = 0
    
    def _init_models(self):
        self.base_models = {
            'xgboost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=5,
                                          random_state=42, n_jobs=2, verbosity=0, tree_method='hist'),
            'catboost': CatBoostClassifier(iterations=200, learning_rate=0.03, depth=5,
                                            random_seed=42, verbose=False, thread_count=2, allow_writing_files=False),
            'randomforest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=2),
            'gradient_boost': GradientBoostingClassifier(n_estimators=200, learning_rate=0.03, max_depth=5, random_state=42)
        }
    
    def _walk_forward_train(self, X: pd.DataFrame, y: pd.Series) -> float:
        tscv = TimeSeriesSplit(n_splits=self.config.WALK_FORWARD_WINDOWS)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            X_train_s = self.scaler.fit_transform(X_train)
            X_val_s = self.scaler.transform(X_val)
            
            base_preds = np.zeros((len(X_val), len(self.base_models)))
            
            for i, (name, model) in enumerate(self.base_models.items()):
                try:
                    if name == 'catboost':
                        model.fit(X_train_s, y_train, verbose=False)
                    else:
                        model.fit(X_train_s, y_train)
                    base_preds[:, i] = model.predict_proba(X_val_s)[:, 1]
                except:
                    base_preds[:, i] = 0.5
            
            meta = LogisticRegression()
            meta.fit(base_preds, y_val)
            scores.append(accuracy_score(y_val, meta.predict(base_preds)))
        
        return np.mean(scores)
    
    def train(self, df: pd.DataFrame) -> bool:
        try:
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                return False
            
            self.logger.info(f"🎓 {self.symbol}: {len(df)} عينة...")
            
            features = calculate_features(df)
            target = create_target(df, self.config.FORECAST_PERIODS)
            
            valid = ~(features.isna().any(axis=1) | target.isna())
            X = features[valid]
            y = target[valid]
            
            if len(X) < 200:
                return False
            
            mi = mutual_info_classif(X, y, random_state=42)
            scores = sorted(zip(X.columns, mi), key=lambda x: x[1], reverse=True)
            self.selected_features = [s[0] for s in scores[:self.config.MAX_FEATURES]]
            X = X[self.selected_features]
            
            self._init_models()
            
            # Walk-Forward
            self.wf_score = self._walk_forward_train(X, y)
            self.logger.info(f"📈 {self.symbol}: WF Score = {self.wf_score:.3f}")
            
            # Final train
            X_s = self.scaler.fit_transform(X)
            base_preds_all = np.zeros((len(X), len(self.base_models)))
            
            for i, (name, model) in enumerate(self.base_models.items()):
                try:
                    if name == 'catboost':
                        model.fit(X_s, y, verbose=False)
                    else:
                        model.fit(X_s, y)
                    base_preds_all[:, i] = model.predict_proba(X_s)[:, 1]
                    
                    self.calibrators[name] = CalibratedClassifierCV(model, cv=3, method='isotonic')
                    self.calibrators[name].fit(X_s, y)
                except:
                    pass
            
            self.meta_model = LogisticRegression()
            self.meta_model.fit(base_preds_all, y)
            
            self.is_trained = True
            self.version = datetime.now().strftime('v%Y%m%d_%H%M%S')
            
            acc = accuracy_score(y, self.meta_model.predict(base_preds_all))
            self.logger.info(f"✅ {self.symbol}: دقة={acc:.1%}, ميزات={len(self.selected_features)}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.symbol}: {e}")
            return False
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.65) -> Tuple[str, float, Dict]:
        if not self.is_trained:
            return "NEUTRAL", 0.0, {}
        
        try:
            features = calculate_features(df).iloc[[-1]]
            available = [f for f in self.selected_features if f in features.columns]
            
            if len(available) < 10:
                return "NEUTRAL", 0.0, {}
            
            X = features[available].fillna(0)
            X_s = self.scaler.transform(X)
            
            base_probas = []
            for name, cal in self.calibrators.items():
                try:
                    base_probas.append(float(cal.predict_proba(X_s)[0, 1]))
                except:
                    base_probas.append(0.5)
            
            meta_proba = float(self.meta_model.predict_proba(np.array([base_probas]))[0, 1])
            
            if meta_proba > threshold:
                return "BUY", meta_proba, {'meta_proba': meta_proba}
            elif meta_proba < (1 - threshold):
                return "SELL", 1 - meta_proba, {'meta_proba': meta_proba}
            return "NEUTRAL", max(meta_proba, 1 - meta_proba), {'meta_proba': meta_proba}
            
        except:
            return "NEUTRAL", 0.0, {}
    
    def save(self):
        path = os.path.join(self.config.MODELS_DIR, self.symbol)
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'base_models': self.base_models,
            'meta_model': self.meta_model,
            'calibrators': self.calibrators,
            'scaler': self.scaler,
            'features': self.selected_features,
            'version': self.version,
            'wf_score': self.wf_score
        }, os.path.join(path, 'pro_model.pkl'))
    
    def load(self) -> bool:
        path = os.path.join(self.config.MODELS_DIR, self.symbol, 'pro_model.pkl')
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.base_models = data['base_models']
        self.meta_model = data['meta_model']
        self.calibrators = data['calibrators']
        self.scaler = data['scaler']
        self.selected_features = data['features']
        self.version = data['version']
        self.wf_score = data.get('wf_score', 0)
        self.is_trained = True
        return True

# ============================================================================
# MAIN BOT
# ============================================================================

class FalconProBot:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config)
        self.db = Database(config.DB_PATH, self.logger)
        self.models = {}
        self.executor = ThreadPoolExecutor(max_workers=config.MAX_WORKERS)
        
        self.tb = telebot.TeleBot(config.TELEGRAM_TOKEN)
        self._setup_commands()
        
        for symbol in config.SYMBOLS:
            model = EnsembleModel(symbol, config, self.logger)
            loaded = model.load()
            self.logger.info(f"{'📂' if loaded else '🆕'} {symbol}")
            self.models[symbol] = model
        
        self.running = False
        self.last_retrain = None
    
    def _setup_commands(self):
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            trained = sum(1 for m in self.models.values() if m.is_trained)
            stats = self.db.get_stats()
            text = f"🦅 **Falcon Pro**\n✅ نماذج: {trained}/{len(self.models)}\n📊 صفقات: {stats['total']}\n📈 نجاح: {stats['win_rate']:.1%}"
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['stats'])
        def stats_cmd(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            s = self.db.get_stats()
            self.tb.reply_to(msg, f"📊 {s['total']} | ✅ {s['wins']} | 📈 {s['win_rate']:.1%}")
    
    def fetch_data(self, symbol: str, interval: str = '5m', period: str = '5d') -> Optional[pd.DataFrame]:
        for attempt in range(self.config.MAX_RETRIES):
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                if not df.empty:
                    df.columns = [c.capitalize() for c in df.columns]
                    return df
            except:
                if attempt < self.config.MAX_RETRIES - 1:
                    time.sleep(self.config.RETRY_DELAY)
        return None
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        try:
            model = self.models.get(symbol)
            if not model or not model.is_trained:
                return None
            
            if self.db.check_active_signal(symbol):
                return None
            
            if self.db.check_recent_signal(symbol, self.config.SIGNAL_COOLDOWN_MINUTES):
                return None
            
            df_5m = self.fetch_data(symbol, '5m', '3d')
            df_15m = self.fetch_data(symbol, '15m', '5d')
            
            if df_5m is None or df_15m is None:
                return None
            
            threshold = self.db.get_dynamic_threshold(symbol)
            
            dir_5m, conf_5m, info_5m = model.predict(df_5m, threshold)
            dir_15m, conf_15m, info_15m = model.predict(df_15m, threshold)
            
            if dir_5m != dir_15m or dir_5m == "NEUTRAL":
                return None
            
            regime = MarketRegime.detect(df_15m)
            
            if not regime['is_trend'] and conf_5m < 0.70:
                return None
            
            confidence = (conf_5m + conf_15m) / 2
            
            if confidence < threshold:
                return None
            
            self.logger.info(f"🎯 {symbol}: {dir_5m} | Meta={info_5m.get('meta_proba', 0):.2%} | {regime['regime']}")
            
            return {
                'symbol': symbol,
                'direction': dir_5m,
                'entry_price': float(df_5m['Close'].iloc[-1]),
                'confidence': confidence,
                'regime': regime['regime'],
                'meta_proba': info_5m.get('meta_proba', 0),
                'expiry_time': (datetime.now() + timedelta(minutes=self.config.TRADE_DURATION_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Analyze {symbol}: {e}")
            return None
    
    def send_signal(self, signal: Dict):
        try:
            emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
            direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
            
            msg = f"{emoji} **{signal['symbol']}** - {direction}\n\n💰 {signal['entry_price']:.5f}\n⏳ {self.config.TRADE_DURATION_MINUTES} د\n💪 {signal['confidence']:.1%}\n📊 {signal['regime']}\n\n🤖 Falcon Pro"
            
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
            self.logger.info(f"✅ إشارة: {signal['symbol']} {signal['direction']}")
        except:
            pass
    
    def check_trades(self):
        for trade in self.db.get_pending_trades():
            try:
                df = self.fetch_data(trade['symbol'], '5m', '1d')
                if df is None:
                    continue
                
                current = float(df['Close'].iloc[-1])
                entry = trade['entry_price']
                
                if trade['direction'] == 'BUY':
                    pnl = (current - entry) / entry * 100
                    result = 'WIN' if current > entry else 'LOSS'
                else:
                    pnl = (entry - current) / entry * 100
                    result = 'WIN' if current < entry else 'LOSS'
                
                self.db.update_result(trade['id'], current, result, pnl, trade['symbol'])
            except:
                pass
    
    def scan_markets(self):
        futures = {self.executor.submit(self.analyze_symbol, s): s for s in self.config.SYMBOLS}
        signals = 0
        for future in as_completed(futures, timeout=60):
            try:
                signal = future.result(timeout=20)
                if signal and self.db.save_signal(signal):
                    self.send_signal(signal)
                    signals += 1
            except:
                pass
        return signals
    
    def train_all_models(self):
        self.logger.info("🎓 تدريب...")
        
        for symbol in self.config.SYMBOLS:
            try:
                df = None
                for interval, period in [('1h', self.config.TRAINING_PERIOD_1H), 
                                          ('15m', self.config.TRAINING_PERIOD_15M)]:
                    df = self.fetch_data(symbol, interval, period)
                    if df is not None and len(df) >= self.config.MIN_TRAINING_SAMPLES:
                        break
                    time.sleep(3)
                
                if df is not None:
                    model = EnsembleModel(symbol, self.config, self.logger)
                    if model.train(df):
                        model.save()
                        self.models[symbol] = model
                
                time.sleep(5)
            except Exception as e:
                self.logger.error(f"Train {symbol}: {e}")
        
        self.last_retrain = datetime.now()
        
        trained = sum(1 for m in self.models.values() if m.is_trained)
        try:
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID,
                f"🎓 **تدريب مكتمل**\n✅ {trained}/{len(self.config.SYMBOLS)}",
                parse_mode='Markdown')
        except:
            pass
    
    def start_telegram(self):
        def poll():
            while True:
                try:
                    self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except:
                    time.sleep(10)
        threading.Thread(target=poll, daemon=True).start()
    
    def run(self):
        self.running = True
        
        self.logger.info("=" * 40)
        self.logger.info("🦅 Falcon AI Pro v4")
        self.logger.info(f"🤖 XGBoost + CatBoost + RF + GB")
        self.logger.info(f"🧠 Meta Model Active")
        self.logger.info("=" * 40)
        
        self.start_telegram()
        time.sleep(2)
        
        if not any(m.is_trained for m in self.models.values()):
            self.train_all_models()
        
        self.last_retrain = datetime.now()
        
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID,
                f"🦅 **Falcon Pro**\n✅ {trained}/{len(self.config.SYMBOLS)}\n⚡️ Scanning...",
                parse_mode='Markdown')
        except:
            pass
        
        while self.running:
            try:
                self.check_trades()
                self.scan_markets()
                
                if (datetime.now() - self.last_retrain).total_seconds() > 86400:
                    self.train_all_models()
                
                time.sleep(self.config.SCAN_INTERVAL_MINUTES * 60)
            except KeyboardInterrupt:
                break
            except Exception as e:
                self.logger.error(f"Loop: {e}")
                time.sleep(30)
        
        self.executor.shutdown(wait=True)

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    config = Config()
    bot = FalconProBot(config)
    bot.run()
