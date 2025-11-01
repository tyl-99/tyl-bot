# trader.py - Summary of Functionality

## Overview
`trader.py` is an automated forex trading orchestrator that monitors 6 currency pairs, analyzes them using Supply & Demand strategies, and generates AI-powered trading recommendations with notifications.

## Core Purpose
The script automatically:
1. Connects to cTrader API to fetch live market data
2. Analyzes each pair using specialized Supply & Demand strategies
3. Scrapes ForexFactory for economic news events
4. Uses Gemini AI (2.5 Flash) to generate trading recommendations
5. Sends Pushover notifications with complete trade analysis

---

## Architecture

### **Main Class: `SimpleTrader`**

A comprehensive trading orchestrator that manages the entire pipeline from data fetching to notification delivery.

---

## Workflow Overview

### **1. Initialization (`__init__`)**
- Loads environment variables (cTrader credentials, Gemini API key, Pushover tokens)
- Initializes strategy instances for 6 pairs:
  - EUR/USD → `EURUSDSTRATEGY`
  - GBP/USD → `GBPUSDStrategy`
  - EUR/JPY → `EURJPYStrategy`
  - EUR/GBP → `EURGBPStrategy`
  - USD/JPY → `USDJPYStrategy`
  - GBP/JPY → `GBPJPYStrategy`
- Sets up testing flags (force signal, dry run mode)
- Initializes news cache storage

### **2. Connection & Authentication**
- **`connect()`**: Establishes connection to cTrader OpenAPI (demo server)
- **`authenticate_app()`**: Authenticates application with client ID/secret
- **`on_user_auth_success()`**: After authentication:
  - Scrapes ForexFactory news **once at startup** (efficient caching)
  - Begins processing pairs sequentially

### **3. News Scraping (One-Time at Startup)**
- **`scrape_all_news()`**: 
  - Uses Selenium to scrape ForexFactory calendar for today's events
  - Extracts: time, currency, impact (high/medium/low), title, actual/forecast/previous values
  - Stores all events in `self.all_news_events` cache
- **`get_news_for_pair()`**: 
  - Filters cached news for specific pair's currencies
  - Returns only high/medium impact events (max 10 events)
  - Much faster than scraping per pair!

### **4. Data Fetching**
- **`fetch_m30_trendbars()`**: 
  - Requests 6 weeks of M30 (30-minute) candles from cTrader
  - Uses asynchronous callbacks (Twisted reactor)
- **`_on_trendbars_response()`**: 
  - Parses protobuf response into pandas DataFrame
  - Calculates `candle_range` column
  - Triggers analysis pipeline

### **5. Strategy Analysis**
- **`analyze_strategy()`**: 
  - Calls pair-specific Supply & Demand strategy
  - Each strategy identifies supply/demand zones and entry signals
  - Returns signal dict: `{decision, entry_price, stop_loss, take_profit, meta, reason}`

### **6. Technical Indicators Calculation**
- **`compute_indicators_snapshot()`**: Computes comprehensive indicators:
  
  **Price & Trend:**
  - `current_price`: Current market price
  - `ema_20`, `ema_50`, `ema_200`: Exponential Moving Averages
  - `price_vs_ema_*`: Position relative to each EMA (above/below)
  - `trend_alignment`: Overall trend (bullish/bearish/mixed)
  
  **Momentum:**
  - `rsi_14`: Relative Strength Index (14-period)
  - `rsi_status`: overbought/oversold/neutral
  
  **Volatility:**
  - `atr_14`: Average True Range (14-period)
  - `volatility_vs_atr`: Current volatility context (high/normal/low)
  
  **Price Action:**
  - `recent_high_5`, `recent_low_5`: Last 5 candles high/low
  
  **Volume:**
  - `volume`: Current volume
  - `volume_vs_avg`: Compared to 20-period average
  
  **Time Context:**
  - `timestamp`: Current time
  - `session`: Trading session (Asian/European/American)

### **7. AI Recommendation**
- **`llm_recommendation()`**: 
  - Enhances signal with risk metrics (risk_pips, reward_pips, R:R ratio, distance_to_entry)
  - Builds comprehensive prompt for Gemini 2.5 Flash
  - Sends: PAIR, SIGNAL, INDICATORS, NEWS (max 4000 chars)
  - Returns AI-generated recommendation (action, confidence, rationale)

### **8. Notification**
- **`format_notification()`**: Formats complete trade analysis:
  - Trade levels (Entry, SL, TP, R:R)
  - Technical indicators (EMAs, RSI, ATR, trend, session)
  - Top 3 ForexFactory news events
  - AI recommendation
  - Timestamp
- **`send_pushover()`**: Sends formatted notification via Pushover API

### **9. Pair Processing Loop**
- **`process_current_pair()`**: Processes one pair at a time
- **`_after_trendbars_ready()`**: Main pipeline execution:
  1. Analyze strategy → get signal
  2. If NO TRADE → log reason and skip
  3. If BUY/SELL → continue:
     - Compute indicators
     - Get filtered news
     - Get AI recommendation
     - Format and send notification
- **`move_next_pair()`**: Moves to next pair (1.5s delay) or stops when done

---

## Key Features

### **Efficient News Handling**
- **One-time scraping**: Scrapes ForexFactory once at startup
- **Cached filtering**: Filters cached events per pair (much faster)
- **Smart filtering**: Only high/medium impact events for relevant currencies

### **Comprehensive Indicator Suite**
- 15+ indicators computed automatically
- Trend, momentum, volatility, volume, and time context
- Helps AI make informed recommendations

### **Enhanced Signal Data**
- Adds risk metrics (risk_pips, reward_pips, R:R ratio)
- Calculates distance to entry
- Provides complete trade context to AI

### **Testing Capabilities**
- `--force-signal BUY/SELL`: Force a signal for testing
- `--force-entry`, `--force-sl`, `--force-tp`: Override prices
- `--dry-run-notify`: Test without sending notifications
- `--pair EUR/USD`: Test single pair

### **Error Handling**
- Graceful fallbacks for missing strategies
- Error logging for API failures
- Continues processing even if one pair fails

---

## Command Line Usage

```bash
# Run normally (all pairs)
python trader.py

# Test single pair
python trader.py --pair EUR/USD

# Force a BUY signal for testing
python trader.py --pair EUR/USD --force-signal BUY

# Dry run (no notifications)
python trader.py --dry-run-notify

# Force signal with custom prices
python trader.py --pair EUR/USD --force-signal BUY --force-entry 1.08500 --force-sl 1.08200 --force-tp 1.09100
```

---

## Environment Variables Required

```bash
# cTrader API
CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_secret
CTRADER_ACCOUNT_ID=your_account_id
CTRADER_ACCESS_TOKEN=optional_token

# Gemini AI
GEMINI_APIKEY=your_gemini_key

# Pushover (optional, has fallback)
PUSHOVER_APP_TOKEN=your_app_token
PUSHOVER_USER_KEY=your_user_key
```

---

## Data Flow Diagram

```
Start
  ↓
Connect to cTrader
  ↓
Authenticate App → Authenticate User
  ↓
Scrape ForexFactory News (once) → Cache Events
  ↓
For Each Pair:
  ├─ Fetch M30 Candles (6 weeks)
  ├─ Parse to DataFrame
  ├─ Analyze Strategy (Supply/Demand)
  │   └─ Returns: BUY/SELL/NO TRADE + Entry/SL/TP
  ├─ If NO TRADE → Log & Skip
  ├─ If BUY/SELL:
  │   ├─ Compute Indicators (EMA/RSI/ATR/Trend/Volume/Session)
  │   ├─ Filter News for Pair
  │   ├─ Get AI Recommendation (Gemini)
  │   ├─ Format Notification
  │   └─ Send Pushover Notification
  └─ Move to Next Pair
  ↓
Stop
```

---

## Output Example

When a trade signal is generated, you receive a Pushover notification containing:

```
🎯 BUY SIGNAL
💱 Pair: EUR/USD

📊 TRADE LEVELS:
Entry: 1.08500
Stop Loss: 1.08200
Take Profit: 1.09100
R:R: 3.0

📈 INDICATORS:
EMA_20: 1.08350
EMA_50: 1.08100
EMA_200: 1.07800
RSI_14: 58.50
ATR_14: 0.00125
Trend: BULLISH
RSI Status: NEUTRAL
Session: European

📰 FOREX FACTORY NEWS:
- [14:00] EUR high: ECB Interest Rate Decision
- [14:30] USD medium: GDP Growth Rate
Summary: Found 2 relevant events for EUR/USD today.

🤖 LLM RECOMMENDATION:
[AI-generated recommendation with confidence level and rationale]

⏰ 2025-10-30 14:43:12 UTC
```

---

## Strategy Type

All 6 pairs use **Supply & Demand strategies** that:
- Identify supply/demand zones based on price consolidation patterns
- Enter trades when price returns to "fresh" (untested) zones
- Use strict risk management (typically 1:3 Risk:Reward ratio)
- Filter by trading session hours (pair-specific optimized windows)
- Use EMA/RSI filters (varies by pair)

---

## Technical Stack

- **cTrader OpenAPI**: Market data via protobuf/Twisted
- **Selenium**: Web scraping for ForexFactory
- **Pandas/NumPy**: Data processing and technical analysis
- **Gemini 2.5 Flash**: AI-powered trade recommendations
- **Pushover API**: Mobile/desktop notifications
- **Python 3.11+**: Runtime environment

---

## Status

✅ **Fully Functional**: Complete end-to-end pipeline
✅ **Optimized**: News scraped once, cached and filtered per pair
✅ **Comprehensive**: 15+ indicators for AI analysis
✅ **Robust**: Error handling and graceful degradation
✅ **Testable**: Multiple testing modes available

