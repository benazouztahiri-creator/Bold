#!/usr/bin/env python3
"""
Falcon AI + FinBERT - Free Pre-trained Model
==============================================
Uses FinBERT for market sentiment analysis.
No training needed. Works immediately.
"""

import os
import sys
import time
import logging
import sqlite3
import hashlib
import threading
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import requests

import telebot

# ============================================================================
# CONFIG
# ============================================================================

TELEGRAM_TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
TELEGRAM_CHAT_ID = '7553333305'

SYMBOLS = ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X']

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FalconFinBERT')

# ============================================================================
# FinBERT SENTIMENT ANALYZER
# ============================================================================

class FinBERTSentiment:
    """
    ✅ FinBERT - نموذج تحليل المشاعر المالية
    مجاني 100% - مفتوح المصدر
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """تحميل FinBERT من HuggingFace"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            logger.info("📥 تحميل FinBERT...")
            
            model_name = "ProsusAI/finbert"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            logger.info("✅ FinBERT جاهز")
            
        except ImportError:
            logger.warning("⚠️ تثبيت المكتبات المطلوبة...")
            os.system("pip install transformers torch -q")
            self._load_model()
        
        except Exception as e:
            logger.error(f"❌ فشل تحميل FinBERT: {e}")
            self.model = None
    
    def analyze(self, text: str) -> Dict:
        """تحليل مشاعر النص"""
        if self.model is None:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        try:
            import torch
            
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # FinBERT: [negative, neutral, positive]
            neg = float(probabilities[0][0])
            neu = float(probabilities[0][1])
            pos = float(probabilities[0][2])
            
            if pos > neg and pos > neu:
                return {'sentiment': 'positive', 'confidence': pos, 'score': pos - neg}
            elif neg > pos and neg > neu:
                return {'sentiment': 'negative', 'confidence': neg, 'score': neg - pos}
            else:
                return {'sentiment': 'neutral', 'confidence': neu, 'score': 0}
                
        except:
            return {'sentiment': 'neutral', 'confidence': 0.5}

# ============================================================================
# RSI + MACD INDICATORS
# ============================================================================

def calculate_technical_score(df: pd.DataFrame) -> float:
    """
    ✅ تحليل فني سريع
    +1 = BUY قوي
    -1 = SELL قوي
    """
    if len(df) < 20:
        return 0
    
    c = df['Close'].values
    
    score = 0
    
    # RSI
    delta = np.diff(c)
    gain = np.mean(delta[delta > 0]) if any(delta > 0) else 0
    loss = np.mean(-delta[delta < 0]) if any(delta < 0) else 0
    rsi = 100 - (100 / (1 + gain / (loss + 1e-8))) if loss > 0 else 50
    
    if rsi < 30:
        score += 0.5  # oversold = BUY
    elif rsi > 70:
        score -= 0.5  # overbought = SELL
    
    # MACD
    ema12 = pd.Series(c).ewm(span=12).mean().values
    ema26 = pd.Series(c).ewm(span=26).mean().values
    macd = ema12[-1] - ema26[-1]
    macd_signal = pd.Series(ema12 - ema26).ewm(span=9).mean().values[-1]
    
    if macd > macd_signal:
        score += 0.5
    else:
        score -= 0.5
    
    # EMA Cross
    ema20 = pd.Series(c).ewm(span=20).mean().values[-1]
    ema50 = pd.Series(c).ewm(span=50).mean().values[-1] if len(c) >= 50 else ema20
    
    if ema20 > ema50:
        score += 0.3
    else:
        score -= 0.3
    
    return max(-1, min(1, score))

# ============================================================================
# MAIN BOT
# ============================================================================

class FalconFinBERT:
    def __init__(self):
        self.sentiment = FinBERTSentiment()
        self.tb = telebot.TeleBot(TELEGRAM_TOKEN)
        self._setup_bot()
    
    def _setup_bot(self):
        try:
            requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=3)
        except:
            pass
        
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != TELEGRAM_CHAT_ID:
                return
            text = "🦅 **Falcon + FinBERT**\n\n✅ النموذج جاهز\n🧠 تحليل مشاعر + تحليل فني\n⚡️ يعمل..."
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(func=lambda msg: True)
        def analyze(msg):
            if str(msg.chat.id) != TELEGRAM_CHAT_ID:
                return
            
            symbol = msg.text.strip().upper()
            if '=X' not in symbol:
                symbol = f"{symbol}=X"
            
            self.tb.reply_to(msg, f"🔍 تحليل {symbol}...")
            
            result = self._analyze_symbol(symbol)
            
            if result is None:
                self.tb.reply_to(msg, "❌ لا بيانات")
                return
            
            emoji = "🟢" if result['direction'] == 'BUY' else ("🔴" if result['direction'] == 'SELL' else "⏳")
            
            text = (f"📊 **{symbol}**\n\n"
                   f"{emoji} {result['direction']}\n"
                   f"💰 {result['price']:.5f}\n"
                   f"📈 فني: {result['technical']:+.1f}\n"
                   f"🧠 مشاعر: {result['sentiment']}\n"
                   f"💪 ثقة: {result['confidence']:.1%}")
            
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """تحليل زوج"""
        try:
            # 1. بيانات السعر
            df = yf.download(symbol, period='5d', interval='5m', progress=False)
            
            if df.empty:
                return None
            
            df.columns = [c.capitalize() for c in df.columns]
            price = float(df['Close'].iloc[-1])
            
            # 2. تحليل فني
            technical_score = calculate_technical_score(df)
            
            # 3. تحليل المشاعر (عنوان مالي عام)
            sentiment_result = self.sentiment.analyze(
                f"The {symbol} forex pair is showing "
                f"{'upward' if technical_score > 0 else 'downward'} momentum "
                f"with RSI at current levels"
            )
            
            # 4. دمج النتائج
            sentiment_score = sentiment_result.get('score', 0)
            
            # 70% فني + 30% مشاعر
            final_score = technical_score * 0.7 + sentiment_score * 0.3
            
            if final_score > 0.3:
                direction = 'BUY'
                confidence = abs(final_score)
            elif final_score < -0.3:
                direction = 'SELL'
                confidence = abs(final_score)
            else:
                direction = 'NEUTRAL'
                confidence = 0.5
            
            return {
                'symbol': symbol,
                'direction': direction,
                'price': price,
                'technical': technical_score,
                'sentiment': sentiment_result['sentiment'],
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"تحليل {symbol}: {e}")
            return None
    
    def run(self):
        logger.info("🦅 Falcon + FinBERT - جاهز")
        
        def poll():
            while True:
                try:
                    self.tb.infinity_polling(timeout=10, long_polling_timeout=5)
                except:
                    time.sleep(5)
        threading.Thread(target=poll, daemon=True).start()
        time.sleep(1)
        
        try:
            self.tb.send_message(TELEGRAM_CHAT_ID, 
                "🦅 **Falcon + FinBERT**\n✅ جاهز\n🧠 تحليل ذكي\n⚡️ يعمل...", 
                parse_mode='Markdown')
        except:
            pass
        
        while True:
            try:
                time.sleep(60)
            except KeyboardInterrupt:
                break
            except:
                time.sleep(30)

if __name__ == "__main__":
    bot = FalconFinBERT()
    bot.run()
