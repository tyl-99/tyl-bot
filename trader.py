#!/usr/bin/env python3
"""
Lightweight trader orchestrator

Flow per pair (mirrors ctrader.py style, simplified):
1) Fetch latest M30 candles via cTrader OpenAPI
2) Pass DataFrame to mapped strategy (same mapping as ctrader.py)
3) If signal -> scrape ForexFactory calendar (via Selenium) and parse events
4) Call Gemini 2.5 Flash with strategy metrics + ForexFactory news for recommendation
5) Send Pushover notification with SL/TP, EMA/RSI, news, and LLM recommendation

Environment variables required:
- CTRADER_CLIENT_ID
- CTRADER_CLIENT_SECRET
- CTRADER_ACCOUNT_ID
- CTRADER_ACCESS_TOKEN (if required by your setup)
- GEMINI_APIKEY
- PUSHOVER_APP_TOKEN (optional, falls back to hardcoded)
- PUSHOVER_USER_KEY (optional, falls back to hardcoded)
"""

import os
import sys
import time
import json
import logging
import datetime
import argparse
from typing import Dict, Any, Optional, List

import requests
import pandas as pd
from dotenv import load_dotenv

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

from ctrader_open_api import Client, Protobuf, TcpProtocol, EndPoints
from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAGetTrendbarsReq
from ctrader_open_api.messages.OpenApiMessages_pb2 import ProtoOAApplicationAuthReq, ProtoOAAccountAuthReq
from ctrader_open_api.messages.OpenApiCommonMessages_pb2 import *
from ctrader_open_api.messages.OpenApiModelMessages_pb2 import *
from twisted.internet import reactor

# Strategy imports (mirror ctrader.py, but allow missing files gracefully)
try:
    from strategy.eurusd_strategy import EURUSDSTRATEGY
except Exception:
    EURUSDSTRATEGY = None
try:
    from strategy.gbpusd_strategy import GBPUSDStrategy as GBPUSDSTRATEGY
except Exception:
    GBPUSDSTRATEGY = None
try:
    from strategy.eurgbp_strategy import EURGBPStrategy as EURGBPSTRATEGY
except Exception:
    EURGBPSTRATEGY = None
try:
    from strategy.usdjpy_strategy import USDJPYStrategy as USDJPYSTRATEGY
except Exception:
    USDJPYSTRATEGY = None
try:
    from strategy.gbpjpy_strategy import GBPJPYStrategy as GBPJPYSTRATEGY
except Exception:
    GBPJPYSTRATEGY = None
try:
    from strategy.eurjpy_strategy import EURJPYStrategy as EURJPYSTRATEGY
except Exception:
    EURJPYSTRATEGY = None

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FOREX_SYMBOLS = {
    "EUR/USD": 1,
    "GBP/USD": 2,
    "EUR/JPY": 3,
    "USD/JPY": 4,
    "GBP/JPY": 7,
    "EUR/GBP": 9,
}

PAIR_TIMEFRAMES = {
    "EUR/USD": "M30",
    "GBP/USD": "M30",
    "EUR/JPY": "M30",
    "EUR/GBP": "M30",
    "USD/JPY": "M30",
    "GBP/JPY": "M30",
}


class SimpleTrader:
    def __init__(self):
        self.client_id = os.getenv("CTRADER_CLIENT_ID")
        self.client_secret = os.getenv("CTRADER_CLIENT_SECRET")
        self.account_id = os.getenv("CTRADER_ACCOUNT_ID")
        if self.account_id:
            try:
                self.account_id = int(self.account_id)
            except Exception:
                raise ValueError("CTRADER_ACCOUNT_ID must be integer")
        if not all([self.client_id, self.client_secret, self.account_id]):
            raise ValueError("Missing cTrader credentials envs")

        self.host = EndPoints.PROTOBUF_DEMO_HOST
        self.client = Client(self.host, EndPoints.PROTOBUF_PORT, TcpProtocol)

        # Strategy instances mapping
        self.strategies: Dict[str, Any] = {
            "EUR/USD": EURUSDSTRATEGY() if EURUSDSTRATEGY else None,
            "GBP/USD": GBPUSDSTRATEGY() if GBPUSDSTRATEGY else None,
            "EUR/JPY": EURJPYSTRATEGY() if EURJPYSTRATEGY else None,
            "EUR/GBP": EURGBPSTRATEGY() if EURGBPSTRATEGY else None,
            "USD/JPY": USDJPYSTRATEGY() if USDJPYSTRATEGY else None,
            "GBP/JPY": GBPJPYSTRATEGY() if GBPJPYSTRATEGY else None,
        }

        self.gemini_api_key = os.getenv("GEMINI_APIKEY")
        self.pushover_app_token = os.getenv("PUSHOVER_APP_TOKEN")
        self.pushover_user_key = os.getenv("PUSHOVER_USER_KEY")

        # Testing flags
        self.force_signal: Optional[str] = None  # BUY/SELL
        self.force_entry: Optional[float] = None
        self.force_sl: Optional[float] = None
        self.force_tp: Optional[float] = None
        self.dry_run_notify: bool = False

        # Pair iteration state
        self.pairs = [p for p in PAIR_TIMEFRAMES.keys() if self.strategies.get(p)]
        self.current_pair: Optional[str] = None
        self.pair_index: int = 0
        self.trendbar: pd.DataFrame = pd.DataFrame()
        
        # News cache - scraped once at startup
        self.all_news_events: List[Dict[str, Any]] = []

    def connect(self) -> None:
        self.client.setConnectedCallback(self.connected)
        self.client.setDisconnectedCallback(self.disconnected)
        self.client.setMessageReceivedCallback(self.on_message)
        self.client.startService()
        reactor.run()

    def connected(self, client):
        logger.info("Connected to cTrader")
        self.authenticate_app()

    def disconnected(self, client, reason):
        logger.info("Disconnected: %s", reason)

    def on_message(self, client, message):
        pass

    def authenticate_app(self):
        req = ProtoOAApplicationAuthReq()
        req.clientId = self.client_id
        req.clientSecret = self.client_secret
        d = self.client.send(req)
        d.addCallbacks(self.on_app_auth_success, self.on_error)

    def on_app_auth_success(self, response):
        access = os.getenv("CTRADER_ACCESS_TOKEN") or ""
        req = ProtoOAAccountAuthReq()
        req.ctidTraderAccountId = self.account_id
        if access:
            req.accessToken = access
        d = self.client.send(req)
        d.addCallbacks(self.on_user_auth_success, self.on_error)

    def on_user_auth_success(self, response):
        logger.info("User authenticated")
        if not self.pairs:
            logger.error("No pairs configured with strategies; stopping")
            reactor.stop()
            return
        # Scrape news once at startup
        logger.info("Scraping ForexFactory news...")
        self.scrape_all_news()
        self.pair_index = 0
        self.process_current_pair()

    def _period_to_proto(self, tf: str):
        # Mirror ctrader.py usage: use enum Value lookup by string name
        try:
            return ProtoOATrendbarPeriod.Value(tf)
        except Exception:
            return ProtoOATrendbarPeriod.Value("M30")

    def fetch_m30_trendbars(self, pair: str, weeks: int = 6) -> pd.DataFrame:
        # This path is now event-driven via callbacks. Keep method for compatibility tests.
        symbol_id = FOREX_SYMBOLS.get(pair)
        if not symbol_id:
            return pd.DataFrame()
        timeframe = PAIR_TIMEFRAMES.get(pair, "M30")
        req = ProtoOAGetTrendbarsReq()
        req.ctidTraderAccountId = self.account_id
        req.period = ProtoOATrendbarPeriod.Value(timeframe)
        req.symbolId = symbol_id
        now = datetime.datetime.utcnow()
        since = now - datetime.timedelta(weeks=weeks)
        req.fromTimestamp = int(since.timestamp() * 1000)
        req.toTimestamp = int(now.timestamp() * 1000)
        d = self.client.send(req)
        d.addCallbacks(lambda resp: self._on_trendbars_response(pair, resp), self.on_error)
        return pd.DataFrame()  # actual data handled async

    def analyze_strategy(self, pair: str, df: pd.DataFrame) -> Dict[str, Any]:
        strat = self.strategies.get(pair)
        if not strat or df is None or df.empty:
            return {"decision": "NO TRADE", "reason": "No data or strategy"}
        try:
            return strat.analyze_trade_signal(df, pair)
        except Exception as e:
            return {"decision": "NO TRADE", "reason": str(e)}

    def _gemini_generate(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "x-goog-api-key": self.gemini_api_key,
            "Content-Type": "application/json"
        }
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        return r.json()

    def scrape_all_news(self) -> None:
        """Scrape ForexFactory calendar once at startup and cache all events"""
        try:
            today_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
            
            # Initialize Selenium
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            # Railway/deployment environment settings
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--disable-software-rasterizer')
            
            # Set Chrome binary location for Railway/Nixpacks
            chrome_bin = os.getenv('GOOGLE_CHROME_BIN') or os.getenv('CHROMIUM_BIN') or '/nix/store/*/chromium-*/bin/chromium'
            if chrome_bin and chrome_bin != '/nix/store/*/chromium-*/bin/chromium':
                chrome_options.binary_location = chrome_bin
            
            # Try webdriver-manager first (auto-downloads ChromeDriver), then fallback to system
            try:
                from webdriver_manager.chrome import ChromeDriverManager
                from selenium.webdriver.chrome.service import Service
                service = Service(ChromeDriverManager().install())
                driver = webdriver.Chrome(service=service, options=chrome_options)
            except Exception:
                try:
                    # Fallback: try with system chromedriver
                    from selenium.webdriver.chrome.service import Service
                    chromedriver_path = os.getenv('CHROMEDRIVER_PATH') or '/usr/bin/chromedriver'
                    service = Service(executable_path=chromedriver_path)
                    driver = webdriver.Chrome(service=service, options=chrome_options)
                except Exception:
                    # Final fallback: let Selenium find it automatically
                    driver = webdriver.Chrome(options=chrome_options)
            driver.implicitly_wait(10)
            
            # Format date for ForexFactory
            dt = datetime.datetime.strptime(today_str, "%Y-%m-%d")
            month_abbr = dt.strftime("%b").lower()
            ff_date = f"{month_abbr}{dt.day}.{dt.year}"
            url = f"https://www.forexfactory.com/calendar?day={ff_date}"
            
            logger.info(f"Loading ForexFactory: {url}")
            driver.get(url)
            
            # Wait for calendar table
            wait = WebDriverWait(driver, 20)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "calendar__table")))
            
            events = []
            rows = driver.find_elements(By.CLASS_NAME, "calendar__row")
            
            for row in rows:
                try:
                    # Get time
                    time_cells = row.find_elements(By.CLASS_NAME, "calendar__time")
                    if not time_cells:
                        continue
                    time_str = time_cells[0].text.strip()
                    
                    # Get currency
                    currency_cells = row.find_elements(By.CLASS_NAME, "calendar__currency")
                    currency = currency_cells[0].text.strip() if currency_cells else ""
                    
                    # Get impact
                    impact = "low"
                    impact_cells = row.find_elements(By.CLASS_NAME, "calendar__impact")
                    if impact_cells:
                        impact_spans = impact_cells[0].find_elements(By.TAG_NAME, "span")
                        if impact_spans:
                            classes = impact_spans[0].get_attribute("class")
                            if "icon--ff-impact-red" in classes:
                                impact = "high"
                            elif "icon--ff-impact-ora" in classes or "icon--ff-impact-yel" in classes:
                                impact = "medium"
                    
                    # Get event title
                    event_cells = row.find_elements(By.CLASS_NAME, "calendar__event")
                    if not event_cells:
                        continue
                    title_spans = event_cells[0].find_elements(By.CLASS_NAME, "calendar__event-title")
                    title = title_spans[0].text.strip() if title_spans else ""
                    
                    if not title:
                        continue
                    
                    # Get values
                    actual_cells = row.find_elements(By.CLASS_NAME, "calendar__actual")
                    actual = actual_cells[0].text.strip() if actual_cells else None
                    
                    forecast_cells = row.find_elements(By.CLASS_NAME, "calendar__forecast")
                    forecast = forecast_cells[0].text.strip() if forecast_cells else None
                    
                    previous_cells = row.find_elements(By.CLASS_NAME, "calendar__previous")
                    previous = previous_cells[0].text.strip() if previous_cells else None
                    
                    event = {
                        "date": today_str,
                        "time": time_str,
                        "time_utc": time_str,
                        "currency": currency,
                        "impact": impact,
                        "title": title,
                        "actual": actual if actual else None,
                        "forecast": forecast if forecast else None,
                        "previous": previous if previous else None
                    }
                    
                    events.append(event)
                    
                except Exception:
                    continue
            
            driver.quit()
            self.all_news_events = events
            logger.info(f"✅ Scraped {len(events)} total events from ForexFactory")
            
        except Exception as e:
            logger.error(f"Error scraping ForexFactory with Selenium: {e}")
            self.all_news_events = []

    def get_news_for_pair(self, pair: str) -> Dict[str, Any]:
        """Filter cached news events for a specific pair"""
        try:
            # If news hasn't been scraped yet, scrape it now (fallback for direct method calls)
            if not self.all_news_events:
                logger.warning("News not scraped yet, scraping now...")
                self.scrape_all_news()
            
            # Extract currencies from pair
            try:
                base, quote = pair.split("/")
            except Exception:
                base, quote = pair, ""
            
            currencies = [base, quote]
            
            # Filter by currencies and impact (high/medium only)
            filtered_events = []
            for event in self.all_news_events:
                event_currency = event.get("currency", "")
                event_impact = event.get("impact", "")
                if event_currency in currencies and event_impact in ["high", "medium"]:
                    filtered_events.append(event)
            
            logger.info(f"✅ Found {len(filtered_events)} relevant events for {pair}")
            
            summary = ""
            if filtered_events:
                summary = f"Found {len(filtered_events)} relevant events for {pair} today."
            else:
                summary = f"No high/medium impact events for {pair} today."
            
            return {
                "events": filtered_events[:10],
                "summary": summary
            }
            
        except Exception as e:
            logger.error(f"Error filtering news for {pair}: {e}")
            return {"events": [], "summary": f"Error filtering news: {e}"}

    def llm_recommendation(self, pair: str, strategy_signal: Dict[str, Any], indicators: Dict[str, Any], news: Dict[str, Any]) -> str:
        """Ask Gemini 2.5 Flash for a trade recommendation given strategy signal + indicators + news."""
        try:
            # Enhance signal with risk metrics
            enhanced_signal = strategy_signal.copy()
            entry = strategy_signal.get("entry_price")
            sl = strategy_signal.get("stop_loss")
            tp = strategy_signal.get("take_profit")
            current_price = indicators.get("current_price", entry)
            
            if entry and sl and tp:
                # Calculate risk metrics
                if enhanced_signal.get("decision") == "BUY":
                    risk_pips = abs(entry - sl)
                    reward_pips = abs(tp - entry)
                else:  # SELL
                    risk_pips = abs(sl - entry)
                    reward_pips = abs(entry - tp)
                
                enhanced_signal["risk_pips"] = round(risk_pips, 5)
                enhanced_signal["reward_pips"] = round(reward_pips, 5)
                if risk_pips > 0:
                    enhanced_signal["risk_reward_ratio"] = round(reward_pips / risk_pips, 2)
                enhanced_signal["distance_to_entry_pips"] = round(abs(current_price - entry), 5) if current_price else 0
            
            prompt = (
                "You are an expert forex trading assistant. Analyze the Supply & Demand strategy signal, technical indicators, and news events. "
                "Consider: trend alignment, RSI conditions, volatility (ATR), session timing, and upcoming news impact. "
                "Provide a concise recommendation (<= 8 lines) with: action (BUY/SELL/HOLD), confidence level (low/medium/high), "
                "and clear rationale combining technical analysis and news context. "
                "If conflicting signals exist, explain the conflict and provide balanced assessment."
            )
            model = "gemini-2.5-flash"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
            payload = {
                "contents": [
                    {"parts": [
                        {"text": prompt},
                        {"text": f"PAIR: {pair}"},
                        {"text": f"SIGNAL: {json.dumps(enhanced_signal)}"},
                        {"text": f"INDICATORS: {json.dumps(indicators)}"},
                        {"text": f"NEWS: {json.dumps(news)[:4000]}"}
                    ]}
                ],
                "generationConfig": {
                    "temperature": 0.4,
                }
            }
            resp = self._gemini_generate(url, payload)
            text = ""
            try:
                text = resp["candidates"][0]["content"]["parts"][0]["text"].strip()
            except Exception:
                text = ""
            return text or "No recommendation available."
        except Exception as e:
            return f"Error getting recommendation: {e}"

    def compute_indicators_snapshot(self, df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        try:
            close = df["close"]
            current_price = float(close.iloc[-1])
            out["current_price"] = current_price
            
            # EMAs
            ema_values = {}
            for p in [20, 50, 200]:
                if len(close) >= p:
                    ema_val = float(pd.Series.ewm(close, span=p, adjust=False).mean().iloc[-1])
                    ema_values[f"ema_{p}"] = ema_val
                    out[f"ema_{p}"] = ema_val
                    # Price position relative to EMA
                    out[f"price_vs_ema_{p}"] = "above" if current_price > ema_val else "below"
            
            # Trend alignment (bullish if price > EMA20 > EMA50 > EMA200)
            if len(ema_values) >= 3:
                if current_price > ema_values.get("ema_20", 0) > ema_values.get("ema_50", 0) > ema_values.get("ema_200", 0):
                    out["trend_alignment"] = "bullish"
                elif current_price < ema_values.get("ema_20", 0) < ema_values.get("ema_50", 0) < ema_values.get("ema_200", 0):
                    out["trend_alignment"] = "bearish"
                else:
                    out["trend_alignment"] = "mixed"
            
            # RSI 14
            if len(close) >= 15:
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi_val = float(rsi.iloc[-1])
                out["rsi_14"] = rsi_val
                out["rsi_status"] = "overbought" if rsi_val > 70 else ("oversold" if rsi_val < 30 else "neutral")
            
            # ATR 14 (Average True Range)
            if len(df) >= 15:
                high = df["high"]
                low = df["low"]
                prev_close = close.shift(1)
                # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
                tr1 = high - low
                tr2 = (high - prev_close).abs()
                tr3 = (low - prev_close).abs()
                true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
                # ATR = Simple Moving Average of True Range (14 period)
                atr = true_range.rolling(window=14).mean()
                atr_val = float(atr.iloc[-1])
                out["atr_14"] = atr_val
                # Volatility context
                out["volatility_vs_atr"] = "high" if (high.iloc[-1] - low.iloc[-1]) > atr_val * 1.5 else ("low" if (high.iloc[-1] - low.iloc[-1]) < atr_val * 0.5 else "normal")
            
            # Recent price action (last 5 candles)
            if len(df) >= 5:
                recent_high = float(df["high"].iloc[-5:].max())
                recent_low = float(df["low"].iloc[-5:].min())
                out["recent_high_5"] = recent_high
                out["recent_low_5"] = recent_low
            
            # Volume context (if available)
            if "volume" in df.columns and len(df) >= 20:
                recent_volume = float(df["volume"].iloc[-1])
                avg_volume = float(df["volume"].iloc[-20:].mean())
                out["volume"] = recent_volume
                out["volume_vs_avg"] = "above" if recent_volume > avg_volume * 1.2 else ("below" if recent_volume < avg_volume * 0.8 else "normal")
            
            # Time context
            if "timestamp" in df.columns and len(df) > 0:
                last_timestamp = df["timestamp"].iloc[-1]
                if isinstance(last_timestamp, pd.Timestamp):
                    out["timestamp"] = last_timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")
                    hour = last_timestamp.hour
                    if 0 <= hour < 8:
                        out["session"] = "Asian"
                    elif 8 <= hour < 16:
                        out["session"] = "European"
                    else:
                        out["session"] = "American"
        except Exception:
            pass
        return out

    def send_pushover(self, title: str, message: str) -> None:
        app_token = self.pushover_app_token or "ah7dehvsrm6j3pmwg9se5h7svwj333"
        user_key = self.pushover_user_key or "u4ipwwnphbcs2j8iiosg3gqvompfs2"
        if not self.pushover_app_token or not self.pushover_user_key:
            logger.warning("Pushover env missing; using hardcoded fallback like ctrader.py")
        payload = {
            "token": app_token,
            "user": user_key,
            "message": message,
            "title": title,
            "priority": 1,
            "sound": "cashregister",
        }
        try:
            r = requests.post("https://api.pushover.net/1/messages.json", data=payload, timeout=10)
            if r.status_code != 200:
                logger.error("Pushover failed: %s", r.text)
        except Exception as e:
            logger.error("Pushover error: %s", e)

    def format_notification(self, pair: str, signal: Dict[str, Any], indicators: Dict[str, Any], news: Dict[str, Any], reco: str) -> str:
        lines = []
        action = signal.get("decision") or signal.get("action") or "UNKNOWN"
        entry = signal.get("entry_price")
        sl = signal.get("stop_loss")
        tp = signal.get("take_profit")
        rr = signal.get("risk_reward_ratio")
        lines.append(f"🎯 {action} SIGNAL")
        lines.append(f"💱 Pair: {pair}")
        lines.append("")
        lines.append("📊 TRADE LEVELS:")
        if entry is not None:
            lines.append(f"Entry: {entry:.5f}")
        if sl is not None:
            lines.append(f"Stop Loss: {sl:.5f}")
        if tp is not None:
            lines.append(f"Take Profit: {tp:.5f}")
        if rr is not None:
            lines.append(f"R:R: {rr}")
        lines.append("")
        lines.append("📈 INDICATORS:")
        for k in ["ema_20", "ema_50", "ema_200", "rsi_14", "atr_14"]:
            if k in indicators:
                val = indicators[k]
                fmt = f"{val:.2f}" if isinstance(val, float) else str(val)
                lines.append(f"{k.upper()}: {fmt}")
        # Add context indicators
        if "trend_alignment" in indicators:
            lines.append(f"Trend: {indicators['trend_alignment'].upper()}")
        if "rsi_status" in indicators:
            lines.append(f"RSI Status: {indicators['rsi_status'].upper()}")
        if "session" in indicators:
            lines.append(f"Session: {indicators['session']}")
        lines.append("")
        lines.append("📰 FOREX FACTORY NEWS:")
        if isinstance(news, dict):
            events = news.get("events") or []
            for ev in events[:3]:
                t = ev.get("time_utc", "?")
                cur = ev.get("currency", "?")
                imp = ev.get("impact", "?")
                title_ev = ev.get("title", "?")
                lines.append(f"- [{t}] {cur} {imp}: {title_ev}")
            summary = news.get("summary")
            if summary:
                lines.append(f"Summary: {summary[:180]}")
        lines.append("")
        lines.append("🤖 LLM RECOMMENDATION:")
        lines.append(reco[:600])
        lines.append("")
        lines.append(f"⏰ {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        return "\n".join(lines)

    def run_once_for_pair(self, pair: str) -> None:
        try:
            df = self.fetch_m30_trendbars(pair, weeks=6)
            # Forced signal mode for testing the full downstream pipeline
            if self.force_signal:
                # Fallbacks based on last close and candle range
                last_close = float(df['close'].iloc[-1]) if df is not None and not df.empty else 0.0
                last_range = float(df['candle_range'].iloc[-1]) if df is not None and not df.empty else 0.0
                
                # If no data, use reasonable defaults for testing
                if last_close == 0.0:
                    # Default prices for common pairs
                    default_prices = {
                        "EUR/USD": 1.08000,
                        "GBP/USD": 1.29000,
                        "USD/JPY": 149.500,
                        "EUR/JPY": 161.500,
                        "GBP/JPY": 192.500,
                        "EUR/GBP": 0.84000,
                    }
                    last_close = default_prices.get(pair, 1.0)
                    last_range = last_close * 0.002  # 0.2% range
                
                entry = self.force_entry if self.force_entry is not None else last_close
                if self.force_sl is not None and self.force_tp is not None:
                    sl = self.force_sl
                    tp = self.force_tp
                else:
                    # Heuristic SL/TP using last candle range and 1:3 RR
                    rng = last_range if last_range > 0 else (entry * 0.001)
                    if self.force_signal.upper() == 'BUY':
                        sl = entry - rng
                        tp = entry + (rng * 3.0)
                    else:
                        sl = entry + rng
                        tp = entry - (rng * 3.0)
                signal = {
                    "decision": self.force_signal.upper(),
                    "entry_price": entry,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "risk_reward_ratio": 3.0,
                    "reason": "Forced test signal"
                }
            else:
                signal = self.analyze_strategy(pair, df)
            action = signal.get("decision") or signal.get("action")
            if action in (None, "NO TRADE", "HOLD", "NONE"):
                logger.info("No trade for %s: %s", pair, signal.get("reason", ""))
                return
            indicators = self.compute_indicators_snapshot(df)
            news = self.get_news_for_pair(pair)
            reco = self.llm_recommendation(pair, signal, indicators, news)
            message = self.format_notification(pair, signal, indicators, news, reco)
            title = f"{action} Signal - {pair}"
            if self.dry_run_notify:
                logger.info("[Dry Run] Would send Pushover: %s\n%s", title, message)
            else:
                self.send_pushover(title, message)
        except Exception as e:
            logger.error("Error running pair %s: %s", pair, e)

    def run(self) -> None:
        # Lifecycle now managed by Twisted reactor in connect()
        self.connect()

    def on_error(self, failure):
        logger.error("API error: %s", failure)
        self.move_next_pair()

    def _on_trendbars_response(self, pair: str, response):
        try:
            parsed = Protobuf.extract(response)
            trendbars = getattr(parsed, 'trendbar', None)
            if not trendbars:
                logger.warning("No trendbars for %s", pair)
                # If forced signal mode, continue with empty DataFrame
                if self.force_signal:
                    self.trendbar = pd.DataFrame()
                    self._after_trendbars_ready()
                else:
                    self.move_next_pair()
                return
            rows = []
            for tb in trendbars:
                rows.append({
                    'timestamp': datetime.datetime.utcfromtimestamp(tb.utcTimestampInMinutes * 60),
                    'open': (tb.low + tb.deltaOpen) / 1e5,
                    'high': (tb.low + tb.deltaHigh) / 1e5,
                    'low': tb.low / 1e5,
                    'close': (tb.low + tb.deltaClose) / 1e5,
                    'volume': tb.volume,
                })
            df = pd.DataFrame(rows)
            df.sort_values('timestamp', inplace=True)
            df['candle_range'] = df['high'] - df['low']
            self.trendbar = df
            # Continue pipeline now that df is ready
            self._after_trendbars_ready()
        except Exception as e:
            logger.error("Trendbar parse error for %s: %s", pair, e)
            self.move_next_pair()

    def process_current_pair(self):
        self.current_pair = self.pairs[self.pair_index]
        logger.info("Processing %s", self.current_pair)
        # request trendbars asynchronously
        self.fetch_m30_trendbars(self.current_pair, weeks=6)

    def _after_trendbars_ready(self):
        pair = self.current_pair
        df = self.trendbar
        # Use forced or real strategy
        if self.force_signal:
            last_close = float(df['close'].iloc[-1]) if df is not None and not df.empty else 0.0
            last_range = float(df['candle_range'].iloc[-1]) if df is not None and not df.empty else 0.0
            
            # If no data, use reasonable defaults for testing
            if last_close == 0.0:
                # Default prices for common pairs
                default_prices = {
                    "EUR/USD": 1.08000,
                    "GBP/USD": 1.29000,
                    "USD/JPY": 149.500,
                    "EUR/JPY": 161.500,
                    "GBP/JPY": 192.500,
                    "EUR/GBP": 0.84000,
                }
                last_close = default_prices.get(pair, 1.0)
                last_range = last_close * 0.002  # 0.2% range
            
            entry = self.force_entry if self.force_entry is not None else last_close
            if self.force_sl is not None and self.force_tp is not None:
                sl = self.force_sl
                tp = self.force_tp
            else:
                rng = last_range if last_range > 0 else (entry * 0.001)
                if self.force_signal.upper() == 'BUY':
                    sl = entry - rng
                    tp = entry + (rng * 3.0)
                else:
                    sl = entry + rng
                    tp = entry - (rng * 3.0)
            signal = {
                "decision": self.force_signal.upper(),
                "entry_price": entry,
                "stop_loss": sl,
                "take_profit": tp,
                "risk_reward_ratio": 3.0,
                "reason": "Forced test signal"
            }
        else:
            signal = self.analyze_strategy(pair, df)
        action = signal.get("decision") or signal.get("action")
        if action in (None, "NO TRADE", "HOLD", "NONE"):
            logger.info("No trade for %s: %s", pair, signal.get("reason", ""))
            self.move_next_pair()
            return
        indicators = self.compute_indicators_snapshot(df)
        news = self.get_news_for_pair(pair)
        reco = self.llm_recommendation(pair, signal, indicators, news)
        message = self.format_notification(pair, signal, indicators, news, reco)
        title = f"{action} Signal - {pair}"
        if self.dry_run_notify:
            logger.info("[Dry Run] Would send Pushover: %s\n%s", title, message)
        else:
            self.send_pushover(title, message)
        self.move_next_pair()

    def move_next_pair(self):
        if self.pair_index < len(self.pairs) - 1:
            self.pair_index += 1
            reactor.callLater(1.5, self.process_current_pair)
        else:
            logger.info("All pairs processed; stopping")
            reactor.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a single scan of pairs or test a forced signal.")
    parser.add_argument("--pair", type=str, default=None, help="Run only this pair (e.g., EUR/USD)")
    parser.add_argument("--force-signal", type=str, choices=["BUY", "SELL"], default=None, help="Force a trade signal for testing")
    parser.add_argument("--force-entry", type=float, default=None, help="Forced entry price")
    parser.add_argument("--force-sl", type=float, default=None, help="Forced stop loss")
    parser.add_argument("--force-tp", type=float, default=None, help="Forced take profit")
    parser.add_argument("--dry-run-notify", action="store_true", help="Do not send Pushover; log message instead")
    args = parser.parse_args()

    try:
        trader = SimpleTrader()
        # Apply testing args
        trader.force_signal = args.force_signal
        trader.force_entry = args.force_entry
        trader.force_sl = args.force_sl
        trader.force_tp = args.force_tp
        trader.dry_run_notify = bool(args.dry_run_notify)

        if args.pair:
            trader.pairs = [args.pair]

        trader.run()
    except KeyboardInterrupt:
        print("\nStopped.")
    except Exception as e:
        print(f"Error: {e}")


