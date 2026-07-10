#!/usr/bin/env python3
"""
Falcon AI + FinBERT - Auto Opportunity Hunter
===============================================
Scans ALL pairs automatically.
Finds the BEST trade and sends it.
No need to ask - it hunts for you.
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

# ✅ 12 زوج للبحث التلقائي
SYMBOLS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
    'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X',
    'GBPJPY=X', 'EURCHF=X', 'USDCHF=X', 'AUDJPY=X'
]

SCAN_INTERVAL = 60  # فحص كل 60 ثانية
MIN_CONFIDENCE = 0.55  # الحد الأدنى للثقة

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FalconHunter')

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    def __init__(self):
        self.db_path = 'falcon_hunter.db'
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    confidence REAL, score REAL,
                    entry_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    expiry_time DATETIME, result TEXT DEFAULT 'PENDING',
                    pnl_percent REAL, signal_hash TEXT UNIQUE
                )
            ''')
            conn.commit()
    
    def save(self, data: Dict) -> bool:
        try:
            h = hashlib.md5(f"{data['symbol']}_{data['direction']}_{time.time()}".encode()).hexdigest()
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR IGNORE INTO signals 
                    (symbol, direction, entry_price, confidence, score, expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data['confidence'], data.get('score', 0), data['expiry_time'], h))
                conn.commit()
            return True
        except:
            return False
    
    def has_active(self, symbol: str) -> bool:
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND result='PENDING' 
                AND expiry_time > datetime('now', 'localtime')
            ''', (symbol,)).fetchone()[0]
            return c > 0
    
    def was_recent(self, symbol: str, minutes: int = 8) -> bool:
        cutoff = (datetime.now() - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
        with sqlite3.connect(self.db_path) as conn:
            c = conn.execute('''
                SELECT COUNT(*) FROM signals WHERE symbol=? AND entry_time > ?
            ''', (symbol, cutoff)).fetchone()[0]
            return c > 0

# ============================================================================
# FinBERT SENTIMENT
# ============================================================================

class FinBERTSentiment:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            logger.info("📥 تحميل FinBERT...")
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info("✅ FinBERT جاهز")
        except Exception as e:
            logger.error(f"❌ FinBERT: {e}")
            self.model = None
    
    def analyze(self, text: str) -> Dict:
        if self.model is None:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'score': 0}
        
        try:
            import torch
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
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
            return {'sentiment': 'neutral', 'confidence': 0.5, 'score': 0}

# ============================================================================
# TECHNICAL ANALYZER
# ============================================================================

def calculate_technical_score(df: pd.DataFrame) -> Tuple[float, str]:
    """تحليل فني - يرجع score وسبب"""
    if len(df) < 20:
        return 0, "بيانات غير كافية"
    
    c = df['Close'].values
    score = 0
    reasons = []
    
    # RSI
    delta = np.diff(c)
    gain = np.mean(delta[delta > 0]) if any(delta > 0) else 0
    loss = np.mean(-delta[delta < 0]) if any(delta < 0) else 0
    rsi = 100 - (100 / (1 + gain / (loss + 1e-8))) if loss > 0 else 50
    
    if rsi < 30:
        score += 1.5
        reasons.append(f"RSI={rsi:.0f}")
    elif rsi > 70:
        score -= 1.5
        reasons.append(f"RSI={rsi:.0f}")
    
    # MACD
    ema12 = pd.Series(c).ewm(span=12).mean().values
    ema26 = pd.Series(c).ewm(span=26).mean().values
    macd_line = ema12[-1] - ema26[-1]
    macd_signal = pd.Series(ema12 - ema26).ewm(span=9).mean().values[-1]
    
    if macd_line > macd_signal:
        score += 1
        reasons.append("MACD+")
    else:
        score -= 1
        reasons.append("MACD-")
    
    # EMA
    ema20 = pd.Series(c).ewm(span=20).mean().values[-1]
    ema50 = pd.Series(c).ewm(span=50).mean().values[-1] if len(c) >= 50 else ema20
    price = c[-1]
    
    if price > ema20:
        score += 0.5
    else:
        score -= 0.5
    
    if ema20 > ema50:
        score += 0.5
    
    # Bollinger
    sma20 = np.mean(c[-20:])
    std20 = np.std(c[-20:])
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    
    if price < bb_lower:
        score += 1.5
        reasons.append("BB↑")
    elif price > bb_upper:
        score -= 1.5
        reasons.append("BB↓")
    
    return score, ", ".join(reasons[:3]) if reasons else "محايد"

# ============================================================================
# MAIN HUNTER
# ============================================================================

class FalconHunter:
    def __init__(self):
        self.sentiment = FinBERTSentiment()
        self.db = Database()
        self.tb = telebot.TeleBot(TELEGRAM_TOKEN)
        self._setup_bot()
        self.best_signal = None
    
    def _setup_bot(self):
        try:
            requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=3)
        except:
            pass
        
        @self.tb.message_handler(commands=['start', 'status'])
        def status(msg):
            if str(msg.chat.id) != TELEGRAM_CHAT_ID:
                return
            text = "🦅 **Falcon Hunter**\n\n🔍 يبحث عن أفضل فرصة\n📊 12 زوج\n⚡️ تلقائي"
            self.tb.reply_to(msg, text, parse_mode='Markdown')
        
        @self.tb.message_handler(commands=['best'])
        def best(msg):
            if str(msg.chat.id) != TELEGRAM_CHAT_ID:
                return
            if self.best_signal:
                s = self.best_signal
                emoji = "🟢" if s['direction'] == 'BUY' else "🔴"
                text = (f"{emoji} **{s['symbol']}** - {s['direction']}\n\n"
                       f"💰 {s['price']:.5f}\n"
                       f"💪 {s['confidence']:.1%}\n"
                       f"📊 Score: {s['score']:.1f}")
                self.tb.reply_to(msg, text, parse_mode='Markdown')
            else:
                self.tb.reply_to(msg, "⏳ لسه ببحث...")
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """تحليل زوج واحد"""
        try:
            if self.db.has_active(symbol):
                return None
            
            if self.db.was_recent(symbol):
                return None
            
            df = yf.download(symbol, period='3d', interval='5m', progress=False)
            
            if df.empty or len(df) < 30:
                return None
            
            df.columns = [c.capitalize() for c in df.columns]
            price = float(df['Close'].iloc[-1])
            
            # تحليل فني
            tech_score, tech_reason = calculate_technical_score(df)
            
            # تحليل مشاعر
            sentiment_result = self.sentiment.analyze(
                f"The {symbol} forex pair shows "
                f"{'bullish' if tech_score > 0 else 'bearish'} signals "
                f"with current momentum"
            )
            
            sentiment_score = sentiment_result.get('score', 0)
            
            # Score نهائي
            final_score = tech_score * 0.7 + sentiment_score * 0.3
            
            if final_score > 0.8:
                direction = 'BUY'
                confidence = min(0.95, 0.5 + abs(final_score) * 0.15)
            elif final_score < -0.8:
                direction = 'SELL'
                confidence = min(0.95, 0.5 + abs(final_score) * 0.15)
            else:
                return None
            
            if confidence < MIN_CONFIDENCE:
                return None
            
            return {
                'symbol': symbol,
                'direction': direction,
                'price': price,
                'confidence': confidence,
                'score': final_score,
                'reason': tech_reason,
                'expiry_time': (datetime.now() + timedelta(minutes=7)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return None
    
    def hunt(self):
        """✅ البحث عن أفضل فرصة بين كل الأزواج"""
        logger.info("🔍 البحث عن أفضل فرصة...")
        
        best = None
        best_score = 0
        
        for symbol in SYMBOLS:
            try:
                result = self.analyze_symbol(symbol)
                
                if result:
                    abs_score = abs(result['score'])
                    
                    if abs_score > best_score:
                        best_score = abs_score
                        best = result
                        
                        logger.info(f"  ⭐ {symbol}: {result['direction']} | "
                                  f"Score={result['score']:.1f} | {result['reason']}")
                
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"  ❌ {symbol}: {e}")
        
        if best:
            self.best_signal = best
            self.db.save(best)
            self.send_signal(best)
            logger.info(f"🏆 أفضل فرصة: {best['symbol']} {best['direction']}")
        else:
            logger.info("⏳ لا توجد فرص قوية حالياً")
    
    def send_signal(self, signal: Dict):
        """إرسال الإشارة"""
        emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
        
        msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
               f"💰 {signal['price']:.5f}\n"
               f"⏳ 7 د\n"
               f"💪 {signal['confidence']:.1%}\n"
               f"📊 {signal['reason']}\n\n"
               f"🤖 Falcon Hunter")
        
        try:
            self.tb.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
            logger.info(f"✅ تم الإرسال: {signal['symbol']} {signal['direction']}")
        except:
            pass
    
    def run(self):
        logger.info("🦅 Falcon Hunter - يبحث عن أفضل فرصة")
        
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
                "🦅 **Falcon Hunter**\n\n🔍 يبحث عن أفضل فرصة\n📊 12 زوج\n⚡️ تلقائي...",
                parse_mode='Markdown')
        except:
            pass
        
        while True:
            try:
                self.hunt()
                time.sleep(SCAN_INTERVAL)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"خطأ: {e}")
                time.sleep(30)

if __name__ == "__main__":
    bot = FalconHunter()
    bot.run()
