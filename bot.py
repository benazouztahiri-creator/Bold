#!/usr/bin/env python3
"""
Falcon AI Pro v6.1 - Scientific Trade Timing
==============================================
Duration based on: Momentum, Cycles, Reversal Points, Volume
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
import signal
import traceback
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import requests
from scipy import stats
from scipy.signal import find_peaks

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
    
    SCAN_INTERVAL_SECONDS: int = 30
    
    # ✅ المدة بقت ذكية بالكامل
    MIN_TRADE_DURATION: int = 2
    MAX_TRADE_DURATION: int = 20
    
    SYMBOLS: List[str] = field(default_factory=lambda: [
        'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
        'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X'
    ])
    
    TRAINING_PERIOD_1H: str = '6mo'
    TRAINING_PERIOD_15M: str = '1mo'
    
    CONFIDENCE_THRESHOLD: float = 0.60
    ENSEMBLE_AGREEMENT: int = 2
    MIN_TRAINING_SAMPLES: int = 500
    MAX_FEATURES: int = 50
    FORECAST_PERIODS: int = 5
    
    DB_PATH: str = 'falcon_trading.db'
    MODELS_DIR: str = 'models'
    
    MAX_RETRIES: int = 3
    RETRY_DELAY: int = 3
    MAX_WORKERS: int = 4
    SIGNAL_COOLDOWN_MINUTES: int = 8
    
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
# TELEGRAM MANAGER
# ============================================================================

class TelegramManager:
    def __init__(self, token: str):
        self.token = token
        self.bot = None
        self._cleanup()
    
    def _cleanup(self):
        try:
            requests.get(f'https://api.telegram.org/bot{self.token}/deleteWebhook')
            time.sleep(0.5)
        except:
            pass
    
    def get_bot(self) -> telebot.TeleBot:
        if self.bot is None:
            self.bot = telebot.TeleBot(self.token)
        return self.bot
    
    def start_polling(self):
        bot = self.get_bot()
        def poll_worker():
            while True:
                try:
                    bot.infinity_polling(timeout=10, long_polling_timeout=5)
                except:
                    time.sleep(5)
        threading.Thread(target=poll_worker, daemon=True).start()

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
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    confidence REAL, trade_duration INTEGER,
                    duration_reason TEXT, entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, signal_hash TEXT UNIQUE
                )
            ''')
            conn.commit()
    
    def save_signal(self, data: Dict) -> Optional[int]:
        try:
            hash_str = f"{data['symbol']}_{data['direction']}_{datetime.now().timestamp()}"
            signal_hash = hashlib.md5(hash_str.encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO signals 
                    (symbol, direction, entry_price, confidence, trade_duration,
                     duration_reason, expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data['trade_duration'],
                      data.get('duration_reason', ''), data['expiry_time'], signal_hash))
                conn.commit()
                return conn.execute('SELECT last_insert_rowid()').fetchone()[0]
        except:
            return None
    
    def update_result(self, signal_id: int, exit_price: float, result: str, pnl: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                UPDATE signals SET exit_price=?, result=?, pnl_percent=?,
                exit_time=datetime('now', 'localtime') WHERE id=?
            ''', (exit_price, result, pnl, signal_id))
            conn.commit()
    
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
    
    def get_expired_trades(self) -> List[Dict]:
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
            total_pnl = conn.execute("SELECT SUM(pnl_percent) FROM signals WHERE result!='PENDING'").fetchone()[0] or 0
            return {
                'total': total, 'wins': wins, 'losses': total-wins,
                'win_rate': wins/total if total > 0 else 0,
                'total_pnl': total_pnl
            }

# ============================================================================
# SCIENTIFIC DURATION CALCULATOR
# ============================================================================

class ScientificDuration:
    """
    ✅ حساب المدة المثالية للصفقة بناءً على:
    1. زخم السعر (Momentum) - متى بينتهي الزخم؟
    2. دورات السوق (Cycles) - فين الدورة الحالية؟
    3. نقاط الانعكاس (Reversal Points) - أقرب دعم/مقاومة
    4. حجم التداول (Volume) - تأكيد الحركة
    """
    
    @staticmethod
    def calculate(df_5m: pd.DataFrame, df_15m: pd.DataFrame, direction: str) -> Tuple[int, str]:
        """
        ✅ حساب المدة المثالية علمياً
        
        Returns:
            (المدة بالدقائق, سبب المدة)
        """
        reasons = []
        durations = []
        
        # ========== 1. تحليل الزخم ==========
        momentum_duration, momentum_reason = ScientificDuration._momentum_analysis(df_5m, direction)
        durations.append(momentum_duration)
        reasons.append(momentum_reason)
        
        # ========== 2. تحليل الدورات ==========
        cycle_duration, cycle_reason = ScientificDuration._cycle_analysis(df_15m)
        durations.append(cycle_duration)
        reasons.append(cycle_reason)
        
        # ========== 3. تحليل نقاط الانعكاس ==========
        reversal_duration, reversal_reason = ScientificDuration._reversal_analysis(df_15m, direction)
        durations.append(reversal_duration)
        reasons.append(reversal_reason)
        
        # ========== 4. تحليل حجم التداول ==========
        volume_duration, volume_reason = ScientificDuration._volume_analysis(df_5m)
        durations.append(volume_duration)
        reasons.append(volume_reason)
        
        # ========== 5. المتوسط المرجح ==========
        # الزخم له وزن أكبر لأنه الأهم
        weights = [0.35, 0.25, 0.25, 0.15]
        final_duration = sum(d * w for d, w in zip(durations, weights))
        
        # تقريب لأقرب دقيقة
        final_duration = max(2, min(20, round(final_duration)))
        
        # سبب واحد مختصر
        main_reason = reasons[0]  # الزخم هو السبب الرئيسي
        
        return final_duration, main_reason
    
    @staticmethod
    def _momentum_analysis(df: pd.DataFrame, direction: str) -> Tuple[int, str]:
        """
        ✅ تحليل الزخم: متى بينتهي الزخم الحالي؟
        """
        closes = df['Close'].values
        
        if len(closes) < 20:
            return 8, "زخم غير واضح"
        
        # حساب معدل التغير (ROC) لفترات مختلفة
        roc_3 = (closes[-1] - closes[-4]) / closes[-4] * 100
        roc_5 = (closes[-1] - closes[-6]) / closes[-6] * 100
        roc_10 = (closes[-1] - closes[-11]) / closes[-11] * 100
        
        # حساب تسارع الزخم (هل الزخم بيزيد ولا بيقل؟)
        momentum_acceleration = roc_3 - roc_10
        
        # حساب RSI للزخم
        delta = np.diff(closes)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.mean(gain[-14:]) if len(gain) >= 14 else np.mean(gain)
        avg_loss = np.mean(loss[-14:]) if len(loss) >= 14 else np.mean(loss)
        
        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))
        
        # تحديد المدة حسب الزخم
        if abs(roc_3) > 0.15 and abs(momentum_acceleration) > 0.05:
            # زخم قوي ومتسارع
            if direction == 'BUY' and roc_3 > 0:
                return 4, "زخم قوي متسارع"
            elif direction == 'SELL' and roc_3 < 0:
                return 4, "زخم قوي متسارع"
            else:
                return 6, "زخم عكسي"
        
        elif abs(roc_5) > 0.1:
            # زخم متوسط
            if rsi > 70 or rsi < 30:
                return 3, "زخم متوسط + تشبع"
            else:
                return 7, "زخم متوسط"
        
        else:
            # زخم ضعيف
            return 10, "زخم ضعيف"
    
    @staticmethod
    def _cycle_analysis(df: pd.DataFrame) -> Tuple[int, str]:
        """
        ✅ تحليل دورات السوق: فين احنا في الدورة؟
        """
        closes = df['Close'].values
        
        if len(closes) < 50:
            return 8, "دورة غير واضحة"
        
        # استخدام autocorrelation لاكتشاف الدورة
        try:
            # تبسيط: حساب المسافة بين القمم المحلية
            peaks, _ = find_peaks(closes[-50:], distance=5)
            
            if len(peaks) >= 2:
                # متوسط المسافة بين القمم
                avg_cycle = np.mean(np.diff(peaks))
                
                # تحديد موقعنا في الدورة
                last_peak_idx = peaks[-1] if len(peaks) > 0 else 0
                position_in_cycle = len(closes[-50:]) - last_peak_idx
                
                if position_in_cycle < avg_cycle * 0.3:
                    return 5, "بداية دورة"
                elif position_in_cycle < avg_cycle * 0.7:
                    return 8, "وسط دورة"
                else:
                    return 4, "نهاية دورة"
            
        except:
            pass
        
        return 7, "دورة طبيعية"
    
    @staticmethod
    def _reversal_analysis(df: pd.DataFrame, direction: str) -> Tuple[int, str]:
        """
        ✅ تحليل نقاط الانعكاس: أقرب دعم/مقاومة
        """
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        
        if len(closes) < 20:
            return 8, "مستويات غير واضحة"
        
        current_price = closes[-1]
        
        # حساب أقرب مقاومة (للشراء) أو دعم (للبيع)
        if direction == 'BUY':
            # البحث عن أقرب مقاومة
            recent_highs = highs[-20:]
            resistance_levels = []
            
            for i in range(len(recent_highs) - 1):
                if recent_highs[i] > current_price:
                    resistance_levels.append(recent_highs[i])
            
            if resistance_levels:
                nearest_resistance = min(resistance_levels)
                distance_pct = (nearest_resistance - current_price) / current_price * 100
                
                if distance_pct < 0.05:
                    return 3, "مقاومة قريبة جداً"
                elif distance_pct < 0.1:
                    return 6, "مقاومة قريبة"
                else:
                    return 10, "مقاومة بعيدة"
        else:
            # البحث عن أقرب دعم
            recent_lows = lows[-20:]
            support_levels = []
            
            for i in range(len(recent_lows) - 1):
                if recent_lows[i] < current_price:
                    support_levels.append(recent_lows[i])
            
            if support_levels:
                nearest_support = max(support_levels)
                distance_pct = (current_price - nearest_support) / current_price * 100
                
                if distance_pct < 0.05:
                    return 3, "دعم قريب جداً"
                elif distance_pct < 0.1:
                    return 6, "دعم قريب"
                else:
                    return 10, "دعم بعيد"
        
        return 8, "بدون مستويات قريبة"
    
    @staticmethod
    def _volume_analysis(df: pd.DataFrame) -> Tuple[int, str]:
        """
        ✅ تحليل حجم التداول: تأكيد الحركة
        """
        if 'Volume' not in df.columns or df['Volume'].sum() == 0:
            return 8, "حجم تداول غير متاح"
        
        volumes = df['Volume'].values
        
        if len(volumes) < 20:
            return 8, "حجم غير كافي"
        
        current_vol = volumes[-1]
        avg_vol = np.mean(volumes[-20:])
        
        vol_ratio = current_vol / (avg_vol + 1e-8)
        
        if vol_ratio > 2.0:
            return 5, "حجم تداول عالي جداً"
        elif vol_ratio > 1.5:
            return 7, "حجم تداول عالي"
        elif vol_ratio > 1.0:
            return 9, "حجم تداول طبيعي"
        else:
            return 12, "حجم تداول منخفض"

# ============================================================================
# MARKET FILTER
# ============================================================================

class MarketFilter:
    @staticmethod
    def can_trade() -> Tuple[bool, str]:
        now = datetime.utcnow()
        day = now.weekday()
        hour = now.hour
        
        if day >= 5:
            return False, "ويكند"
        if day == 4 and hour >= 20:
            return False, "إغلاق جمعة"
        
        return True, "مسموح"

# ============================================================================
# FEATURES
# ============================================================================

def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    c, h, l, o = df['Close'], df['High'], df['Low'], df['Open']
    
    for p in [1, 3, 5, 10]:
        f[f'ret_{p}'] = c.pct_change(p)
    
    for p in [5, 10, 20, 50]:
        f[f'sma_{p}'] = c.rolling(p).mean()
        f[f'ema_{p}'] = c.ewm(span=p, adjust=False).mean()
    
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    f['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-8)))
    
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    f['macd'] = ema12 - ema26
    f['macd_signal'] = f['macd'].ewm(span=9).mean()
    
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    f['bb_pos'] = (c - sma20) / (2 * std20 + 1e-8)
    
    tr1 = h - l
    tr2 = abs(h - c.shift())
    tr3 = abs(l - c.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    f['atr'] = tr.ewm(span=14).mean()
    
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    f['stoch_k'] = 100 * (c - low14) / (high14 - low14 + 1e-8)
    
    plus_dm = h.diff().clip(lower=0)
    minus_dm = (-l.diff()).clip(lower=0)
    atr14 = tr.ewm(span=14).mean()
    plus_di = 100 * (plus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    minus_di = 100 * (minus_dm.ewm(span=14).mean()) / (atr14 + 1e-8)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    f['adx'] = dx.ewm(span=14).mean()
    
    for p in [5, 10]:
        f[f'roc_{p}'] = (c - c.shift(p)) / (c.shift(p) + 1e-8) * 100
    
    return f.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)

# ============================================================================
# TARGET
# ============================================================================

def create_balanced_target(df: pd.DataFrame, periods: int) -> pd.Series:
    atr = (df['High'] - df['Low']).rolling(14).mean()
    future = df['Close'].shift(-periods)
    change = future - df['Close']
    threshold = atr * 0.5
    
    target = pd.Series(np.nan, index=df.index)
    
    buy_count = (change > threshold).sum()
    sell_count = (change < -threshold).sum()
    
    if buy_count > sell_count * 1.5:
        target[change > threshold * 1.3] = 1
        target[change < -threshold] = 0
    elif sell_count > buy_count * 1.5:
        target[change > threshold] = 1
        target[change < -threshold * 1.3] = 0
    else:
        target[change > threshold] = 1
        target[change < -threshold] = 0
    
    return target

# ============================================================================
# MODEL
# ============================================================================

class EnsembleModel:
    def __init__(self, symbol: str, config: Config, logger: logging.Logger):
        self.symbol = symbol
        self.config = config
        self.logger = logger
        self.models = {}
        self.meta_model = None
        self.calibrators = {}
        self.scaler = RobustScaler()
        self.selected_features = []
        self.is_trained = False
        self.version = None
    
    def _init_models(self):
        self.models = {
            'xgb': xgb.XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=4,
                                      random_state=42, n_jobs=2, verbosity=0, tree_method='hist'),
            'cat': CatBoostClassifier(iterations=150, learning_rate=0.03, depth=4,
                                       random_seed=42, verbose=False, thread_count=2,
                                       allow_writing_files=False),
            'rf': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42, n_jobs=2),
            'gb': GradientBoostingClassifier(n_estimators=150, learning_rate=0.03, max_depth=4, random_state=42)
        }
    
    def train(self, df: pd.DataFrame) -> bool:
        try:
            if len(df) < self.config.MIN_TRAINING_SAMPLES:
                return False
            
            features = calculate_features(df)
            target = create_balanced_target(df, self.config.FORECAST_PERIODS)
            
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
            X_s = self.scaler.fit_transform(X)
            
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X_s[:split_idx], X_s[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            base_preds = np.zeros((len(X_val), len(self.models)))
            
            for i, (name, model) in enumerate(self.models.items()):
                try:
                    if name == 'cat':
                        model.fit(X_train, y_train, verbose=False)
                    else:
                        model.fit(X_train, y_train)
                    base_preds[:, i] = model.predict_proba(X_val)[:, 1]
                    
                    self.calibrators[name] = CalibratedClassifierCV(model, cv=3, method='isotonic')
                    self.calibrators[name].fit(X_train, y_train)
                except:
                    base_preds[:, i] = 0.5
            
            self.meta_model = LogisticRegression()
            self.meta_model.fit(base_preds, y_val)
            
            self.is_trained = True
            self.version = datetime.now().strftime('v%Y%m%d_%H%M%S')
            return True
            
        except:
            return False
    
    def predict(self, df: pd.DataFrame, threshold: float = 0.60) -> Tuple[str, float]:
        if not self.is_trained:
            return "NEUTRAL", 0.0
        
        try:
            features = calculate_features(df).iloc[[-1]]
            available = [f for f in self.selected_features if f in features.columns]
            
            if len(available) < 10:
                return "NEUTRAL", 0.0
            
            X = features[available].fillna(0)
            X_s = self.scaler.transform(X)
            
            base_probas = []
            votes = 0
            
            for name, cal in self.calibrators.items():
                try:
                    proba = float(cal.predict_proba(X_s)[0, 1])
                    base_probas.append(proba)
                    if proba > 0.5:
                        votes += 1
                except:
                    base_probas.append(0.5)
            
            proba_buy = float(self.meta_model.predict_proba(np.array([base_probas]))[0, 1])
            
            if proba_buy > threshold and votes >= self.config.ENSEMBLE_AGREEMENT:
                return "BUY", proba_buy
            elif (1 - proba_buy) > threshold and (len(self.calibrators) - votes) >= self.config.ENSEMBLE_AGREEMENT:
                return "SELL", 1 - proba_buy
            
            return "NEUTRAL", max(proba_buy, 1 - proba_buy)
            
        except:
            return "NEUTRAL", 0.0
    
    def save(self):
        path = os.path.join(self.config.MODELS_DIR, self.symbol)
        os.makedirs(path, exist_ok=True)
        joblib.dump({
            'models': self.models,
            'meta_model': self.meta_model,
            'calibrators': self.calibrators,
            'scaler': self.scaler,
            'features': self.selected_features,
            'version': self.version
        }, os.path.join(path, 'model.pkl'))
    
    def load(self) -> bool:
        path = os.path.join(self.config.MODELS_DIR, self.symbol, 'model.pkl')
        if not os.path.exists(path):
            return False
        data = joblib.load(path)
        self.models = data['models']
        self.meta_model = data['meta_model']
        self.calibrators = data['calibrators']
        self.scaler = data['scaler']
        self.selected_features = data['features']
        self.version = data['version']
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
        
        self.tg = TelegramManager(config.TELEGRAM_TOKEN)
        self.tb = self.tg.get_bot()
        self._setup_commands()
        
        for symbol in config.SYMBOLS:
            model = EnsembleModel(symbol, config, self.logger)
            if model.load():
                self.logger.info(f"📂 {symbol}")
            else:
                self.logger.info(f"🆕 {symbol}")
            self.models[symbol] = model
        
        self.running = False
        self.last_retrain = None
        self.last_scan_time = {}
    
    def _setup_commands(self):
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != self.config.TELEGRAM_CHAT_ID:
                return
            trained = sum(1 for m in self.models.values() if m.is_trained)
            stats = self.db.get_stats()
            can_trade, reason = MarketFilter.can_trade()
            text = (f"🦅 **Falcon Pro v6.1**\n"
                   f"✅ نماذج: {trained}/{len(self.models)}\n"
                   f"📊 صفقات: {stats['total']}\n"
                   f"📈 نجاح: {stats['win_rate']:.1%}\n"
                   f"💰 ربح: {stats['total_pnl']:.2f}%\n"
                   f"🧠 مدة علمية\n"
                   f"🚦 {'✅' if can_trade else '⛔ '+reason}")
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
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
            can_trade, _ = MarketFilter.can_trade()
            if not can_trade:
                return None
            
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
            
            dir_5m, conf_5m = model.predict(df_5m, self.config.CONFIDENCE_THRESHOLD)
            dir_15m, conf_15m = model.predict(df_15m, self.config.CONFIDENCE_THRESHOLD)
            
            if dir_5m != dir_15m or dir_5m == "NEUTRAL":
                return None
            
            confidence = (conf_5m + conf_15m) / 2
            
            if confidence < self.config.CONFIDENCE_THRESHOLD:
                return None
            
            # ✅ حساب المدة العلمية
            trade_duration, duration_reason = ScientificDuration.calculate(df_5m, df_15m, dir_5m)
            
            entry_price = float(df_5m['Close'].iloc[-1])
            
            self.logger.info(f"🎯 {symbol}: {dir_5m} | {confidence:.1%} | {trade_duration}د | {duration_reason}")
            
            return {
                'symbol': symbol,
                'direction': dir_5m,
                'entry_price': entry_price,
                'confidence': confidence,
                'trade_duration': trade_duration,
                'duration_reason': duration_reason,
                'expiry_time': (datetime.now() + timedelta(minutes=trade_duration)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except:
            return None
    
    def send_signal(self, signal: Dict):
        try:
            emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
            direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
            
            msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
                   f"💰 {signal['entry_price']:.5f}\n"
                   f"⏳ {signal['trade_duration']} د\n"
                   f"💪 {signal['confidence']:.1%}\n\n"
                   f"🤖 Falcon Pro")
            
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
            self.logger.info(f"✅ {signal['symbol']} {signal['direction']} | {signal['trade_duration']}د | {signal.get('duration_reason', '')}")
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
                
                if direction == 'BUY':
                    pnl = (current - entry) / entry * 100
                    result = 'WIN' if current > entry else 'LOSS'
                else:
                    pnl = (entry - current) / entry * 100
                    result = 'WIN' if current < entry else 'LOSS'
                
                self.db.update_result(trade['id'], current, result, pnl)
            except:
                pass
    
    def scan_all_symbols(self):
        futures = {self.executor.submit(self.analyze_symbol, s): s for s in self.config.SYMBOLS}
        signals = 0
        for future in as_completed(futures, timeout=30):
            try:
                signal = future.result(timeout=15)
                if signal and self.db.save_signal(signal):
                    self.send_signal(signal)
                    signals += 1
            except:
                pass
        return signals
    
    def train_all_models(self):
        for symbol in self.config.SYMBOLS:
            try:
                df = None
                for interval, period in [('1h', self.config.TRAINING_PERIOD_1H), 
                                          ('15m', self.config.TRAINING_PERIOD_15M)]:
                    df = self.fetch_data(symbol, interval, period)
                    if df is not None and len(df) >= self.config.MIN_TRAINING_SAMPLES:
                        break
                    time.sleep(2)
                
                if df is not None:
                    model = EnsembleModel(symbol, self.config, self.logger)
                    if model.train(df):
                        model.save()
                        self.models[symbol] = model
                
                time.sleep(2)
            except:
                pass
        
        self.last_retrain = datetime.now()
    
    def run(self):
        self.running = True
        
        self.logger.info("🦅 Falcon Pro v6.1 - Scientific Timing")
        
        self.tg.start_polling()
        time.sleep(1)
        
        if not any(m.is_trained for m in self.models.values()):
            self.train_all_models()
        
        self.last_retrain = datetime.now()
        
        try:
            trained = sum(1 for m in self.models.values() if m.is_trained)
            self.tb.send_message(self.config.TELEGRAM_CHAT_ID,
                f"🦅 **Falcon Pro v6.1**\n✅ {trained}/{len(self.config.SYMBOLS)}\n🧠 توقيت علمي\n⚡️ يعمل...",
                parse_mode='Markdown')
        except:
            pass
        
        while self.running:
            try:
                self.check_trades()
                signals = self.scan_all_symbols()
                
                if (datetime.now() - self.last_retrain).total_seconds() > 86400:
                    self.train_all_models()
                
                time.sleep(self.config.SCAN_INTERVAL_SECONDS)
                
            except KeyboardInterrupt:
                break
            except:
                time.sleep(5)
        
        self.executor.shutdown(wait=True)

def main():
    config = Config()
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    while True:
        try:
            bot = FalconProBot(config)
            bot.run()
        except KeyboardInterrupt:
            break
        except:
            time.sleep(5)

if __name__ == "__main__":
    main()
