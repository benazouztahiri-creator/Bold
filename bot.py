#!/usr/bin/env python3
"""
Falcon Hunter Pro - ADX + ATR + Real News
===========================================
Professional upgrades:
1. ADX filter (no trades in sideways market)
2. ATR-based dynamic SL/TP
3. Real Yahoo Finance news + FinBERT analysis
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
import json

import telebot

# ============================================================================
# CONFIG
# ============================================================================

TELEGRAM_TOKEN = '8773849578:AAH9a6-8hU5YFYTad2EA5jQyfffIoeL8npk'
TELEGRAM_CHAT_ID = '7553333305'

SYMBOLS = [
    'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X',
    'USDCAD=X', 'NZDUSD=X', 'EURGBP=X', 'EURJPY=X',
    'GBPJPY=X', 'EURCHF=X', 'USDCHF=X', 'AUDJPY=X'
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
logger = logging.getLogger('FalconPro')

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    def __init__(self):
        self.db_path = 'falcon_pro.db'
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT, direction TEXT, entry_price REAL,
                    stop_loss REAL, take_profit REAL,
                    confidence REAL, score REAL,
                    adx REAL, atr REAL,
                    news_sentiment TEXT, news_confidence REAL,
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
                    (symbol, direction, entry_price, stop_loss, take_profit,
                     confidence, score, adx, atr, news_sentiment, news_confidence,
                     expiry_time, signal_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (data['symbol'], data['direction'], data['entry_price'],
                      data.get('stop_loss'), data.get('take_profit'),
                      data['confidence'], data.get('score', 0),
                      data.get('adx', 0), data.get('atr', 0),
                      data.get('news_sentiment', ''), data.get('news_confidence', 0),
                      data['expiry_time'], h))
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
            
            logger.info("📥 FinBERT...")
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info("✅ FinBERT")
        except:
            self.model = None
    
    def analyze(self, text: str) -> Dict:
        if self.model is None:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'score': 0}
        
        try:
            import torch
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            neg, neu, pos = float(probs[0][0]), float(probs[0][1]), float(probs[0][2])
            
            if pos > neg and pos > neu:
                return {'sentiment': 'positive', 'confidence': pos, 'score': pos - neg}
            elif neg > pos and neg > neu:
                return {'sentiment': 'negative', 'confidence': neg, 'score': neg - pos}
            return {'sentiment': 'neutral', 'confidence': neu, 'score': 0}
        except:
            return {'sentiment': 'neutral', 'confidence': 0.5, 'score': 0}

# ============================================================================
# 1. REAL NEWS FETCHER
# ============================================================================

class NewsFetcher:
    """✅ يجيب أخبار حقيقية من Yahoo Finance"""
    
    @staticmethod
    def get_forex_news() -> List[Dict]:
        news_list = []
        
        try:
            # Yahoo Finance RSS للفوركس
            url = "https://feeds.finance.yahoo.com/rss/2.0/headline?s=EURUSD=X,GBPUSD=X,USDJPY=X&region=US&lang=en-US"
            
            import feedparser
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:5]:  # أول 5 أخبار
                news_list.append({
                    'title': entry.title,
                    'summary': entry.get('summary', ''),
                    'published': entry.get('published', ''),
                    'link': entry.get('link', '')
                })
        except:
            pass
        
        # ✅ أخبار احتياطية لو RSS فشل
        if not news_list:
            try:
                # Yahoo Finance API
                symbols = "EURUSD=X,GBPUSD=X,USDJPY=X"
                url = f"https://query2.finance.yahoo.com/v1/finance/news?symbols={symbols}"
                headers = {'User-Agent': 'Mozilla/5.0'}
                r = requests.get(url, headers=headers, timeout=10)
                data = r.json()
                
                for item in data.get('news', [])[:5]:
                    news_list.append({
                        'title': item.get('title', ''),
                        'summary': item.get('summary', ''),
                        'published': item.get('published', ''),
                        'link': item.get('link', '')
                    })
            except:
                pass
        
        return news_list
    
    @staticmethod
    def get_currency_news(symbol: str) -> str:
        """يجيب أخبار خاصة بزوج معين"""
        currency = symbol.replace('=X', '')
        
        try:
            url = f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US"
            import feedparser
            feed = feedparser.parse(url)
            
            if feed.entries:
                return feed.entries[0].title
        except:
            pass
        
        return f"Forex market update for {currency}"

# ============================================================================
# 2. ADX CALCULATOR
# ============================================================================

def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """✅ حساب ADX"""
    h = df['High'].values
    l = df['Low'].values
    c = df['Close'].values
    
    tr1 = h[1:] - l[1:]
    tr2 = np.abs(h[1:] - c[:-1])
    tr3 = np.abs(l[1:] - c[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    
    up = h[1:] - h[:-1]
    down = l[:-1] - l[1:]
    
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / (atr + 1e-8)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / (atr + 1e-8)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-8)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    
    return float(adx[-1])

# ============================================================================
# 3. ATR-BASED RISK MANAGEMENT
# ============================================================================

def calculate_atr_sl_tp(df: pd.DataFrame, entry_price: float, direction: str) -> Tuple[float, float]:
    """✅ ATR للوقف والهدف"""
    h = df['High'].values
    l = df['Low'].values
    c = df['Close'].values
    
    tr1 = h[-14:] - l[-14:]
    tr2 = np.abs(h[-14:] - np.roll(c[-14:], 1))
    tr3 = np.abs(l[-14:] - np.roll(c[-14:], 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.mean(tr)
    
    pip_value = 0.01 if 'JPY' in df.name if hasattr(df, 'name') else False else 0.0001
    
    # ✅ ديناميكي: SL = 1.5x ATR, TP = 3x ATR
    sl_distance = atr * 1.5
    tp_distance = atr * 3.0
    
    if direction == 'BUY':
        stop_loss = entry_price - sl_distance
        take_profit = entry_price + tp_distance
    else:
        stop_loss = entry_price + sl_distance
        take_profit = entry_price - tp_distance
    
    return round(stop_loss, 5), round(take_profit, 5), round(atr, 5)

# ============================================================================
# TECHNICAL ANALYZER
# ============================================================================

def calculate_technical_score(df: pd.DataFrame) -> Tuple[float, str, float, float]:
    """تحليل فني + ADX + ATR"""
    if len(df) < 30:
        return 0, "بيانات غير كافية", 0, 0
    
    c = df['Close'].values
    score = 0
    reasons = []
    
    # ✅ ADX
    adx = calculate_adx(df)
    
    if adx < 20:
        return 0, f"ADX={adx:.0f} سوق عرضي", adx, 0  # مفيش تداول
    
    # ✅ ATR
    h = df['High'].values
    l = df['Low'].values
    tr1 = h[-14:] - l[-14:]
    tr2 = np.abs(h[-14:] - np.roll(c[-14:], 1))
    tr3 = np.abs(l[-14:] - np.roll(c[-14:], 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = np.mean(tr)
    
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
    
    if price > ema20: score += 0.5
    else: score -= 0.5
    if ema20 > ema50: score += 0.5
    
    # Bollinger
    sma20 = np.mean(c[-20:])
    std20 = np.std(c[-20:])
    bb_lower = sma20 - 2 * std20
    bb_upper = sma20 + 2 * std20
    
    if price < bb_lower:
        score += 1.5
        reasons.append("BB↑")
    elif price > bb_upper:
        score -= 1.5
        reasons.append("BB↓")
    
    return score, ", ".join(reasons[:3]) if reasons else "محايد", adx, atr

# ============================================================================
# MAIN HUNTER
# ============================================================================

class FalconPro:
    def __init__(self):
        self.sentiment = FinBERTSentiment()
        self.db = Database()
        self.tb = telebot.TeleBot(TELEGRAM_TOKEN)
        self._setup_bot()
    
    def _setup_bot(self):
        try:
            requests.get(f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/deleteWebhook', timeout=3)
        except:
            pass
        
        @self.tb.message_handler(commands=['start'])
        def start(msg):
            text = "🦅 **Falcon Pro**\n\n✅ ADX + ATR + أخبار\n🔍 يبحث تلقائياً"
            self.tb.reply_to(msg, text, parse_mode='Markdown')
    
    def analyze_symbol(self, symbol: str) -> Optional[Dict]:
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
            
            # ✅ تحليل فني + ADX + ATR
            tech_score, tech_reason, adx, atr = calculate_technical_score(df)
            
            # ✅ ADX فلتر
            if adx < 20:
                logger.info(f"⛔ {symbol}: ADX={adx:.0f} سوق عرضي")
                return None
            
            # ✅ أخبار حقيقية
            news_text = NewsFetcher.get_currency_news(symbol)
            news_sentiment = self.sentiment.analyze(news_text[:200])
            
            # ✅ Score نهائي
            sentiment_score = news_sentiment.get('score', 0)
            final_score = tech_score * 0.6 + sentiment_score * 0.4
            
            if final_score > 0.8:
                direction = 'BUY'
            elif final_score < -0.8:
                direction = 'SELL'
            else:
                return None
            
            confidence = min(0.95, 0.5 + abs(final_score) * 0.15)
            
            if confidence < MIN_CONFIDENCE:
                return None
            
            # ✅ ATR للوقف والهدف
            stop_loss, take_profit, atr_val = calculate_atr_sl_tp(df, price, direction)
            
            return {
                'symbol': symbol,
                'direction': direction,
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'confidence': confidence,
                'score': final_score,
                'adx': round(adx, 1),
                'atr': round(atr_val, 5),
                'reason': tech_reason,
                'news_sentiment': news_sentiment['sentiment'],
                'news_confidence': news_sentiment['confidence'],
                'expiry_time': (datetime.now() + timedelta(minutes=7)).strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            return None
    
    def hunt(self):
        logger.info("🔍 بحث...")
        
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
                                  f"ADX={result['adx']} | {result['reason']}")
                
                time.sleep(2)
            except:
                pass
        
        if best:
            self.db.save(best)
            self.send_signal(best)
            logger.info(f"🏆 {best['symbol']} {best['direction']}")
        else:
            logger.info("⏳ لا فرص")
    
    def send_signal(self, signal: Dict):
        emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        direction = "شراء ▲" if signal['direction'] == 'BUY' else "بيع ▼"
        
        msg = (f"{emoji} **{signal['symbol']}** - {direction}\n\n"
               f"💰 دخول: {signal['price']:.5f}\n"
               f"🛑 SL: {signal['stop_loss']:.5f}\n"
               f"🎯 TP: {signal['take_profit']:.5f}\n"
               f"⏳ 7 د\n"
               f"💪 {signal['confidence']:.1%}\n"
               f"📊 ADX: {signal['adx']}\n"
               f"🧠 {signal['news_sentiment']}\n\n"
               f"🤖 Falcon Pro")
        
        try:
            self.tb.send_message(TELEGRAM_CHAT_ID, msg, parse_mode='Markdown')
            logger.info(f"✅ {signal['symbol']} {signal['direction']}")
        except:
            pass
    
    def run(self):
        logger.info("🦅 Falcon Pro - ADX + ATR + أخبار")
        
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
                "🦅 **Falcon Pro**\n\n✅ ADX + ATR + أخبار\n🔍 يبحث...",
                parse_mode='Markdown')
        except:
            pass
        
        while True:
            try:
                self.hunt()
                time.sleep(SCAN_INTERVAL)
            except KeyboardInterrupt:
                break
            except:
                time.sleep(30)

if __name__ == "__main__":
    bot = FalconPro()
    bot.run()
