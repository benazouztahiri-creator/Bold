#!/usr/bin/env python3
"""
Falcon AI Pro v4.1 - Balanced Signals + Accurate Confidence
============================================================
Fixed: Buy/Sell balance + Proper confidence scoring
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
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
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
    
    # ✅ عتبات منفصلة للشراء والبيع
    CONFIDENCE_THRESHOLD_BUY: float = 0.60
    CONFIDENCE_THRESHOLD_SELL: float = 0.60
    
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
                    confidence REAL, proba_buy REAL, proba_sell REAL,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, signal_hash TEXT UNIQUE
                );
                CREATE TABLE IF NOT EXISTS performance (
                    symbol TEXT PRIMARY KEY,
                    buy_wins INTEGER DEFAULT 0, buy_total INTEGER DEFAULT 0,
                    sell_wins INTEGER DEFAULT 0, sell_total INTEGER DEFAULT 0,
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
                    (symbol, direction, entry_price, confidence, proba_buy, proba_sell, expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data.get('proba_buy', 0), data.get('proba_sell', 0),
                      data['expiry_time'], signal_hash))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except:
            return None
    
    def update_result(self, signal_id: int, exit_price: float, result: str, pnl: float, 
                      symbol: str, direction: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('UPDATE signals SET exit_price=?, result=?, pnl_percent=? WHERE id=?',
                        (exit_price, result, pnl, signal_id))
            
            # ✅ تحديث منفصل للشراء والبيع
            if direction == 'BUY':
                conn.execute('''
                    INSERT INTO performance (symbol, buy_wins, buy_total)
                    VALUES (?, ?, 1)
                    ON CONFLICT(symbol) DO UPDATE SET
                    buy_wins = buy_wins + ?,
                    buy_total = buy_total + 1
                ''', (symbol, 1 if result == 'WIN' else 0, 1 if result == 'WIN' else 0))
            else:
                conn.execute('''
                    INSERT INTO performance (symbol, sell_wins, sell_total)
                    VALUES (?, ?, 1)
                    ON CONFLICT(symbol) DO UPDATE SET
                    sell_wins = sell_wins + ?,
                    sell_total = sell_total + 1
                ''', (symbol, 1 if result == 'WIN' else 0, 1 if result == 'WIN' else 0))
            conn.commit()
    
    def get_dynamic_threshold(self, symbol: str) -> Tuple[float, float]:
        """✅ يرجع عتبة منفصلة للشراء والبيع"""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute('''
                SELECT buy_wins, buy_total, sell_wins, sell_total 
                FROM performance WHERE symbol=?
            ''', (symbol,)).fetchone()
            
            if not row or (row[1] or 0) < 5:
                buy_threshold = 0.60
            else:
                buy_rate = row[0] / max(row[1], 1)
                if buy_rate > 0.65: buy_threshold = 0.55
                elif buy_rate > 0.55: buy_threshold = 0.60
                else: buy_threshold = 0.68
            
            if not row or (row[3] or 0) < 5:
                sell_threshold = 0.60
            else:
                sell_rate = row[2] / max(row[3], 1)
                if sell_rate > 0.65: sell_threshold = 0.55
                elif sell_rate > 0.55: sell_threshold = 0.60
                else: sell_threshold = 0.68
            
            return buy_threshold, sell_threshold
    
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
            buys = conn.execute("SELECT COUNT(*) FROM signals WHERE result!='PENDING' AND direction='BUY'").fetchone()[0]
            sells = conn.execute("SELECT COUNT(*) FROM signals WHERE result!='PENDING' AND direction='SELL'").fetchone()[0]
            return {
                'total': total, 'wins': wins, 'losses': total-wins,
                'win_rate': wins/total if total > 0 else 0,
                'buy_count': buys, 'sell_count': sells
            }

# ============================================================================
# FEATURES
# ============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l, o = df['Close'], df['High'], df['Low'], df['Open']
    
    for p in [1, 3, 5, 10, 20]:
        f[f'ret_{p}'] = c.pct_change(p)
    
    for p in [5, 10, 20, 50]:
        f[f'sma_{p}'] = c.rolling(p).mean()
        f[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
        f[f'dist_sma_{p}'] = (c - f[f'sma_{p}']) / (f[f'sma_{p}'] + 1e-8)
    
    for p in [7, 14]:
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
    
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    f['cci'] = (tp - sma_tp) / (0.015 * mad + 1e-8)
    
    for p in [5, 10]:
        f[f'roc_{p}'] = (c - c.shift(p)) / (c.shift(p) + 1e-8) * 100
    
    for p in [5, 10, 20]:
        f[f'vol_{p}'] = c.pct_change().rolling(p).std()
    
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    atr14 = tr.ewm(span=14).mean()
    plus_di = 100 * (plus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    minus_di = 100 * (minus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    f['adx'] = dx.ewm(span=14).mean()
    
    f['body_size'] = abs(c - o) / (h - l + 1e-8)
    f['upper_wick'] = (h - np.maximum(c, o)) / (h - l + 1e-8)
    f['lower_wick'] = (np.minimum(c, o) - l) / (h - l + 1e-8)
    
    f['trend_str'] = c.rolling(20).apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
    
    return f.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# ============================================================================
# BALANCED TARGET
# ============================================================================

def create_balanced_target(df: pd.DataFrame, periods: int) -> pd.Series:
    """
    ✅ هدف متوازن:
    - BUY (1): السعر بيرتفع بمقدار meaningful
    - SELL (0): السعر بينخفض بمقدار meaningful
    - متأكدينش إن عدد BUY ≈ عدد SELL
    """
    atr = (df['High'] - df['Low']).rolling(14).mean()
    future = df['Close'].shift(-periods)
    change = future - df['Close']
    threshold = atr * 0.5
    
    # عد الإشارتين
    buy_count = (change > threshold).sum()
    sell_count = (change < -threshold).sum()
    
    target = pd.Series(np.nan, index=df.index)
    
    # ✅ لو في فرق كبير، نزود العتبة للجهة الأكثر
    if buy_count > sell_count * 1.5:
        # BUY كتير → نرفع عتبة الشراء
        buy_threshold = threshold * 1.3
        sell_threshold = threshold
    elif sell_count > buy_count * 1.5:
        # SELL كتير → نرفع عتبة البيع
        buy_threshold = threshold
        sell_threshold = threshold * 1.3
    else:
        buy_threshold = threshold
        sell_threshold = threshold
    
    target[change > buy_threshold] = 1   # BUY
    target[change < -sell_threshold] = 0  # SELL
    
    return target

# ============================================================================
# ENSEMBLE MODEL WITH CALIBRATED CONFIDENCE
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
        self.buy_sell_ratio = 1.0  # ✅ نسبة الشراء للبيع في التدريب
    
    def _init_models(self):
        self.base_models = {
            'xgboost': xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=5,
                                          random_state=42, n_jobs=2, verbosity=0, tree_method='hist'),
            'catboost': CatBoostClassifier(iterations=200, learning_rate=0.03, depth=5,
                                            random_seed=42, verbose=False, thread_count=2, 
                                            allow_writing_files=False, auto_class_weights='Balanced'),
            'randomforest': RandomForestClassifier(n_estimators=200, max_depth=10, 
                                                    random_state=42, n_jobs=2, class_weight='balanced'),
            'gradient_boost': GradientBoostingClassifier(n_estimators=200, learning_rate=0.03, 
                                                          max_depth=5, random_state=42)
        }
    
    def train(self, df: pd.DataFrame) -> bool:
        try:
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                return False
            
            self.logger.info(f"🎓 {self.symbol}: {len(df)} عينة...")
            
            features = calculate_features(df)
            target = create_balanced_target(df, self.config.FORECAST_PERIODS)
            
            valid = ~(features.isna().any(axis=1) | target.isna())
            X = features[valid]
            y = target[valid]
            
            # ✅ حساب نسبة الشراء/البيع
            buy_pct = y.mean()
            self.buy_sell_ratio = buy_pct / (1 - buy_pct) if buy_pct > 0 and buy_pct < 1 else 1.0
            self.logger.info(f"📊 {self.symbol}: BUY={buy_pct:.1%}, SELL={1-buy_pct:.1%}, Ratio={self.buy_sell_ratio:.2f}")
            
            if len(X) < 200:
                return False
            
            mi = mutual_info_classif(X, y, random_state=42)
            scores = sorted(zip(X.columns, mi), key=lambda x: x[1], reverse=True)
            self.selected_features = [s[0] for s in scores[:self.config.MAX_FEATURES]]
            X = X[self.selected_features]
            
            self._init_models()
            
            X_s = self.scaler.fit_transform(X)
            
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X_s[:split_idx], X_s[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            base_preds = np.zeros((len(X_val), len(self.base_models)))
            
            for i, (name, model) in enumerate(self.base_models.items()):
                try:
                    if name == 'catboost':
                        model.fit(X_train, y_train, verbose=False)
                    else:
                        model.fit(X_train, y_train)
                    base_preds[:, i] = model.predict_proba(X_val)[:, 1]
                    
                    # ✅ Probability Calibration
                    self.calibrators[name] = CalibratedClassifierCV(
                        model, cv=3, method='isotonic'
                    )
                    self.calibrators[name].fit(X_train, y_train)
                except:
                    base_preds[:, i] = 0.5
            
            self.meta_model = LogisticRegression(class_weight='balanced')
            self.meta_model.fit(base_preds, y_val)
            
            self.is_trained = True
            self.version = datetime.now().strftime('v%Y%m%d_%H%M%S')
            
            meta_pred = self.meta_model.predict(base_preds)
            acc = accuracy_score(y_val, meta_pred)
            f1 = f1_score(y_val, meta_pred, zero_division=0)
            
            # ✅ حساب Brier Score (مؤشر جودة الاحتمالات)
            meta_proba = self.meta_model.predict_proba(base_preds)[:, 1]
            brier = brier_score_loss(y_val, meta_proba)
            
            self.logger.info(f"✅ {self.symbol}: دقة={acc:.1%}, F1={f1:.3f}, Brier={brier:.4f}, ميزات={len(self.selected_features)}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ {self.symbol}: {e}", exc_info=True)
            return False
    
    def predict(self, df: pd.DataFrame, threshold_buy: float = 0.60, threshold_sell: float = 0.60) -> Tuple[str, float, float, float]:
        """
        ✅ يرجع: (الاتجاه, الثقة, احتمالية الشراء, احتمالية البيع)
        """
        if not self.is_trained:
            return "NEUTRAL", 0.0, 0.0, 0.0
        
        try:
            features = calculate_features(df).iloc[[-1]]
            available = [f for f in self.selected_features if f in features.columns]
            
            if len(available) < 10:
                return "NEUTRAL", 0.0, 0.0, 0.0
            
            X = features[available].fillna(0)
            X_s = self.scaler.transform(X)
            
            # ✅ احتمالات منفصلة من كل نموذج
            base_probas = []
            for name, cal in self.calibrators.items():
                try:
                    base_probas.append(float(cal.predict_proba(X_s)[0, 1]))
                except:
                    base_probas.append(0.5)
            
            # ✅ Meta model - احتمالية BUY
            proba_buy = float(self.meta_model.predict_proba(np.array([base_probas]))[0, 1])
            proba_sell = 1 - proba_buy
            
            # ✅ تعديل الاحتمالات حسب نسبة البيانات الأصلية
            # لو البيانات الأصلية كان فيها BUY أكتر، نخفض احتمالية BUY شوية
            if self.buy_sell_ratio > 1.3:
                proba_buy = proba_buy * 0.95  # نخفض BUY لو كان أكتر في التدريب
            elif self.buy_sell_ratio < 0.7:
                proba_buy = proba_buy * 1.05  # نزود BUY لو كان أقل في التدريب
            
            proba_buy = np.clip(proba_buy, 0.05, 0.95)
            proba_sell = 1 - proba_buy
            
            # ✅ قرار منفصل للشراء والبيع
            if proba_buy > threshold_buy:
                direction = "BUY"
                confidence = proba_buy
            elif proba_sell > threshold_sell:
                direction = "SELL"
                confidence = proba_sell
            else:
                direction = "NEUTRAL"
                confidence = max(proba_buy, proba_sell)
            
            return direction, confidence, proba_buy, proba_sell
            
        except Exception as e:
            return "NEUTRAL", 0.0, 0.0, 0.0
    
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
            'buy_sell_ratio': self.buy_sell_ratio
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
        self.buy_sell_ratio = data.get('buy_sell_ratio', 1.0)
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
        self._remove_webhook()
        self._setup_commands()
        
        for symbol in config.SYMBOLS:
            model = EnsembleModel(symbol, config, self.logger)
            loaded = model.load()
            self.logger.info(f"{'📂' if loaded else '🆕'} {symbol}" + 
                           (f" (Ratio={model.buy_sell_ratio:.2f})" if loaded else ""))
            self.models[symbol] = model
        
        self.running = False
        self.last_retrain = None
    
    def _remove_webhook(self):
        try:
            self.tb.remove_webhook()
            time.sleep(1)
        except:
            pass
    
    def _setup_commands(self):
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            trained = sum(1 for m in self.models.values() if m.is_trained)
            stats = self.db.get_stats()
            text = (f"🦅 **Falcon Pro v4.1**\n"
                   f"✅ نماذج: {trained}/{len(self.models)}\n"
                   f"📊 صفقات: {stats['total']}\n"
                   f"🟢 شراء: {stats.get('buy_count', 0)} | 🔴 بيع: {stats.get('sell_count', 0)}\n"
                   f"📈 نجاح: {stats['win_rate']:.1%}")
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['stats'])
        def stats_cmd(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            s = self.db.get_stats()
            self.tb.reply_to(msg, 
                f"📊 {s['total']} | ✅ {s['wins']} | 📈 {s['win_rate']:.1%}\n"
                f"🟢 شراء: {s.get('buy_count', 0)} | 🔴 بيع: {s.get('sell_count', 0)}")
    
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
            
            # ✅ عتبات ديناميكية منفصلة
            threshold_buy, threshold_sell = self.db.get_dynamic_threshold(symbol)
            
            dir_5m, conf_5m, proba_buy_5m, proba_sell_5m = model.predict(df_5m, threshold_buy, threshold_sell)
            dir_15m, conf_15m, proba_buy_15m, proba_sell_15m = model.predict(df_15m, threshold_buy, threshold_sell)
            
            if dir_5m != dir_15m or dir_5m == "NEUTRAL":
                return None
            
            confidence = (conf_5m + conf_15m) / 2
            proba_buy = (proba_buy_5m + proba_buy_15m) / 2
            proba_sell = (proba_sell_5m + proba_sell_15m) / 2
            
            # ✅ استخدام العتبة المناسبة للاتجاه
            if dir_5m == 'BUY' and confidence < threshold_buy:
                return None
            if dir_5m == 'SELL' and confidence < threshold_sell:
                return None
            
            self.logger.info(f"🎯 {symbol}: {dir_5m} | ثقة={confidence:.1%} | "
                           f"P(BUY)={proba_buy:.2%} | P(SELL)={proba_sell:.2%}")
            
            return {
                'symbol': symbol,
                'direction': dir_5m,
                'entry_price': float(df_5m['Close'].iloc[-1]),
                'confidence': confidence,
                'proba_buy': proba_buy,
                'proba_sell': proba_sell,
                'expiry_time': (datetime.now() + timedelta(minutes=self.config.TRADE_DURATION_MINUTES)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Analyze {symbol}: {e}")
            return None
    
    def send_signal(self, signal: Dict):
        try:
            emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
            direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
            
            msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
                   f"💰 الدخول: {signal['entry_price']:.5f}\n"
                   f"⏳ المدة: {self.config.TRADE_DURATION_MINUTES} د\n"
                   f"💪 الثقة: {signal['confidence']:.1%}\n"
                   f"📊 P(BUY): {signal['proba_buy']:.1%} | P(SELL): {signal['proba_sell']:.1%}\n\n"
                   f"🤖 Falcon Pro v4.1")
            
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
                
                self.db.update_result(trade['id'], current, result, pnl, trade['symbol'], trade['direction'])
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
        self.logger.info("🎓 تدريب النماذج...")
        
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
                f"🎓 **تدريب مكتمل**\n✅ {trained}/{len(self.config.SYMBOLS)}\n⚖️ Balanced Target",
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
        self.logger.info("🦅 Falcon AI Pro v4.1")
        self.logger.info("⚖️ Balanced Buy/Sell Target")
        self.logger.info("📊 Calibrated Confidence (Brier Score)")
        self.logger.info("🎯 Separate Buy/Sell Thresholds")
        self.logger.info("=" * 40)
        
        self.start_telegram()
        time.sleep(2)
        
        if not any(m.is_trained for m in self.models.values()):
            self.train_all_models()
        
        self.last_retrain = datetime.now()
        
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID,
                f"🦅 **Falcon Pro v4.1**\n✅ {trained}/{len(self.config.SYMBOLS)}\n⚖️ Balanced\n⚡️ Scanning...",
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
