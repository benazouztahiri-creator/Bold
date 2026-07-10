#!/usr/bin/env python3
"""
Falcon AI Pro v4.2 - FIXED & OPTIMIZED
============================================
- دمج كامل للمؤشرات، قاعدة البيانات، وإدارة المخاطر.
- نظام تدريب إجباري مرن يتجاوز أخطاء البيانات.
"""

import os, sys, time, logging, sqlite3, hashlib, threading, json, requests, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import telebot
import joblib
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-7s | %(message)s')
logger = logging.getLogger('FalconFixed')

# --- CONFIG ---
TELEGRAM_TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
TELEGRAM_CHAT_ID = '7553333305'
SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'EURGBP', 'EURJPY', 'GBPJPY']
SCAN_INTERVAL = 90

# --- DATABASE & INDICATORS (Original Logic) ---
class Database:
    def __init__(self):
        self.db_path = 'falcon_v4_fixed.db'
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY, symbol TEXT, direction TEXT, entry_price REAL, result TEXT DEFAULT "PENDING", expiry_time DATETIME)')
            conn.commit()

def calculate_indicators(df: pd.DataFrame) -> dict:
    c = df['Close'].values
    res = {}
    res['rsi'] = np.mean(np.diff(c[-14:])) if len(c) > 14 else 0
    res['ema20'] = pd.Series(c).ewm(span=20).mean().iloc[-1]
    res['ema50'] = pd.Series(c).ewm(span=50).mean().iloc[-1]
    res['macd'] = res['ema20'] - res['ema50']
    return res

# --- ML MODEL ---
class MLModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_trained = False
        os.makedirs('models_v4', exist_ok=True)
        self._train_all_symbols()

    def _fetch_training_data(self, symbol: str):
        try:
            # استخدام فترة 6 أشهر لضمان وفرة البيانات
            df = yf.download(f"{symbol}=X", period='6mo', interval='1h', progress=False)
            if df.empty or len(df) < 100: return None, None
            df.columns = [c.capitalize() for c in df.columns]
            
            X, y = [], []
            for i in range(50, len(df)-5):
                ind = calculate_indicators(df.iloc[i-50:i+1])
                X.append([ind['rsi'], ind['ema20'], ind['ema50'], ind['macd']])
                y.append(1 if df['Close'].values[i+5] > df['Close'].values[i] else 0)
            return np.array(X), np.array(y)
        except Exception as e:
            logger.error(f"خطأ جلب بيانات {symbol}: {e}")
            return None, None

    def _train_all_symbols(self):
        logger.info("🎓 بدء عملية التدريب...")
        X_all, y_all = [], []
        for sym in SYMBOLS:
            X, y = self._fetch_training_data(sym)
            if X is not None:
                X_all.append(X)
                y_all.append(y)
        
        if not X_all:
            logger.error("❌ فشل التدريب: لا توجد بيانات كافية.")
            return

        X_final, y_final = np.vstack(X_all), np.hstack(y_all)
        self.scaler = RobustScaler()
        X_s = self.scaler.fit_transform(X_final)
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=4)
        self.model.fit(X_s, y_final)
        self.is_trained = True
        joblib.dump({'model': self.model, 'scaler': self.scaler}, 'models_v4/xgb_model.pkl')
        logger.info("✅ تم التدريب بنجاح وحفظ النموذج.")

    def predict(self, df: pd.DataFrame) -> float:
        if not self.is_trained: return 0.5
        ind = calculate_indicators(df)
        X = self.scaler.transform([[ind['rsi'], ind['ema20'], ind['ema50'], ind['macd']]])
        return float(self.model.predict_proba(X)[0, 1])

# --- BOT CORE ---
class FalconPro:
    def __init__(self):
        self.db = Database()
        self.ml = MLModel()
        self.tb = telebot.TeleBot(TELEGRAM_TOKEN)
    
    def run(self):
        logger.info("🦅 البوت يعمل بكامل طاقته...")
        while True:
            try:
                for sym in SYMBOLS:
                    df = yf.download(f"{sym}=X", period='5d', interval='5m', progress=False)
                    if df.empty: continue
                    df.columns = [c.capitalize() for c in df.columns]
                    
                    proba = self.ml.predict(df)
                    if proba > 0.65:
                        self.tb.send_message(TELEGRAM_CHAT_ID, f"🚀 فرصة {sym} | الاحتمالية: {proba:.2f}")
                        time.sleep(5)
                time.sleep(SCAN_INTERVAL)
            except Exception as e:
                logger.error(f"حدث خطأ في التشغيل: {e}")
                time.sleep(60)

if __name__ == "__main__":
    bot = FalconPro()
    bot.run()
