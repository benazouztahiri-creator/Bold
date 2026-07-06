import telebot
import time
import pandas as pd
import numpy as np
import xgboost as xgb
import yfinance as yf
from datetime import datetime

# =============================================
# ⚙️ الإعدادات الأساسية
# =============================================
TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
CHAT_ID = '7553333305'
bot = telebot.TeleBot(TOKEN)

# =============================================
# 🧠 محرك الذكاء الاصطناعي (AI Engine)
# =============================================
class FalconAI:
    def __init__(self):
        self.model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=6)
        self.is_trained = False

    def prepare_features(self, df):
        df = df.copy()
        df['rsi'] = 100 - (100 / (1 + df['close'].diff().rolling(14).mean() / df['close'].diff().rolling(14).mean().abs()))
        df['ema20'] = df['close'].ewm(span=20).mean()
        df['ema50'] = df['close'].ewm(span=50).mean()
        df['volatility'] = df['close'].rolling(20).std()
        df['momentum'] = df['close'].pct_change(5)
        return df.fillna(0)

    def train(self, df):
        df = self.prepare_features(df)
        df['target'] = (df['close'].shift(-3) > df['close']).astype(int)
        df = df.dropna()
        X = df[['rsi', 'ema20', 'ema50', 'volatility', 'momentum']]
        y = df['target']
        self.model.fit(X, y)
        self.is_trained = True

    def get_signal(self, df):
        if not self.is_trained: return "NEUTRAL", 0
        features = self.prepare_features(df).iloc[[-1]][['rsi', 'ema20', 'ema50', 'volatility', 'momentum']]
        prob = self.model.predict_proba(features)[0][1]
        if prob > 0.65: return "BUY", prob
        if prob < 0.35: return "SELL", prob
        return "NEUTRAL", prob

ai_engine = FalconAI()

# =============================================
# 📤 الدوال المساعدة
# =============================================
def send_startup_message():
    msg = f"""
🦅 **تم تشغيل الصقر v5.0 (AI Powered)**
⏰ التوقيت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📡 البوت الآن يقوم بتحليل الأسواق...
"""
    bot.send_message(CHAT_ID, msg, parse_mode='Markdown')

def send_signal(symbol, signal, confidence, price):
    emoji = "🟢" if signal == "BUY" else "🔴"
    direction = "صعود ▲" if signal == "BUY" else "هبوط ▼"
    
    msg = f"""
╔══════════════════════╗
   🎯 إشارة الصقر (AI)
╚══════════════════════╝

{emoji} **الاتجاه:** {direction}
💰 **سعر الدخول:** {price:.5f}
📊 **نسبة الثقة:** {confidence:.2%}

⚠️ للتعليم فقط | إدارة رأس المال أولاً
"""
    bot.send_message(CHAT_ID, msg, parse_mode='Markdown')

# =============================================
# 🚀 الحلقة الرئيسية
# =============================================
def main_loop():
    # إرسال رسالة التفعيل فوراً
    send_startup_message()
    
    symbols = ['EURUSD=X', 'BTC-USD', 'GC=F']
    print("🦅 الصقر v5.0 (AI Hybrid) يعمل...")
    
    while True:
        for symbol in symbols:
            try:
                df = yf.Ticker(symbol).history(period='10d', interval='5m')
                df = df.rename(columns={'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close'})
                
                if not ai_engine.is_trained:
                    ai_engine.train(df)
                
                signal, conf = ai_engine.get_signal(df)
                
                if signal != "NEUTRAL":
                    send_signal(symbol, signal, conf, df['close'].iloc[-1])
                    print(f"✅ إشارة {signal} لـ {symbol} بنسبة ثقة {conf:.2%}")
                    
                time.sleep(30)
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)

if __name__ == "__main__":
    main_loop()
