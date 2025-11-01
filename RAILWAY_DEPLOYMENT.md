# Railway Deployment Guide

## ✅ Deployment Readiness

Your project is **mostly ready** for Railway deployment, but requires a few configuration steps.

## 📋 Pre-Deployment Checklist

### ✅ What's Already Done:
- ✅ `requirements.txt` - All dependencies listed
- ✅ `runtime.txt` - Python version specified (3.10.16)
- ✅ `Procfile` - Worker process defined
- ✅ `railway.json` - Railway configuration
- ✅ `nixpacks.toml` - Build configuration with Chrome/ChromeDriver
- ✅ Selenium configured for headless Chrome in Railway environment
- ✅ Environment variable handling via `python-dotenv`

### ⚠️ What You Need to Do:

#### 1. **Set Environment Variables in Railway Dashboard**

Go to your Railway project → Variables tab and add:

```
# Required
CTRADER_CLIENT_ID=your_client_id
CTRADER_CLIENT_SECRET=your_secret
CTRADER_ACCOUNT_ID=your_account_id
GEMINI_APIKEY=your_gemini_key

# Optional
CTRADER_ACCESS_TOKEN=optional_token

# Pushover (OPTIONAL - already hardcoded as fallback in trader.py)
# Only set these if you want to use different credentials
PUSHOVER_APP_TOKEN=your_app_token
PUSHOVER_USER_KEY=your_user_key
```

#### 2. **Verify Strategy Files**

Ensure all strategy files in `strategy/` directory are present:
- `eurusd_strategy.py`
- `gbpusd_strategy.py`
- `eurjpy_strategy.py`
- `eurgbp_strategy.py`
- `usdjpy_strategy.py`
- `gbpjpy_strategy.py`

#### 3. **Test Locally First**

Before deploying, test that the app runs:
```bash
python trader.py --dry-run-notify
```

## 🚀 Deployment Steps

### Option 1: Deploy via Railway Dashboard

1. **Connect Repository**
   - Go to [Railway.app](https://railway.app)
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose your repository: `tyl-99/tyl-bot`

2. **Configure Service**
   - Railway will auto-detect Python project
   - It will use `Procfile` for the start command
   - Set all environment variables (see above)

3. **Deploy**
   - Railway will automatically build and deploy
   - Check logs for any errors

### Option 2: Deploy via Railway CLI

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Link project
railway link

# Set environment variables
railway variables set CTRADER_CLIENT_ID=your_value
railway variables set CTRADER_CLIENT_SECRET=your_value
# ... (repeat for all variables)

# Deploy
railway up
```

## 🔧 How It Works on Railway

### Build Process:
1. Railway detects Python project
2. Uses Nixpacks builder (configured via `nixpacks.toml`)
3. Installs Python 3.10.16
4. Installs Chromium and ChromeDriver (for Selenium)
5. Installs Python dependencies from `requirements.txt`

### Runtime:
1. Starts worker process (`python trader.py`)
2. App connects to cTrader API
3. Scrapes ForexFactory news (once at startup)
4. Processes pairs sequentially
5. Sends notifications via Pushover

## ⚠️ Important Notes

### **Selenium/Chrome Setup**
- Railway uses Nixpacks which installs Chromium automatically
- ChromeDriver path is set via environment variables
- Falls back gracefully if ChromeDriver not found
- Uses headless mode (no GUI needed)

### **Application Type**
- This is a **worker** process (not a web server)
- Runs continuously (keeps checking pairs)
- Uses Twisted reactor for async operations
- Will restart automatically on failure (configured in `railway.json`)

### **Resource Usage**
- Memory: ~500MB-1GB (due to Selenium/Chrome)
- CPU: Low (mostly waiting for API responses)
- Network: Moderate (API calls + web scraping)

### **Monitoring**
- Check Railway logs for errors
- Monitor Pushover notifications to verify it's working
- App will log all activities

## 🐛 Troubleshooting

### If Selenium Fails:
- Check Railway logs for ChromeDriver errors
- Verify `nixpacks.toml` includes chromium packages
- Environment variables `GOOGLE_CHROME_BIN` and `CHROMEDRIVER_PATH` should be auto-set

### If Build Fails:
- Check `requirements.txt` for version conflicts
- Verify Python version in `runtime.txt` matches Railway's supported versions
- Check Railway build logs for specific errors

### If App Crashes:
- Check cTrader API credentials
- Verify Gemini API key is valid
- Check network connectivity from Railway's IPs
- Review error logs in Railway dashboard

## 📊 Expected Behavior

Once deployed:
1. ✅ App starts and connects to cTrader
2. ✅ Authenticates successfully
3. ✅ Scrapes ForexFactory news (takes ~10-30 seconds)
4. ✅ Processes each pair sequentially
5. ✅ Sends notifications when signals are found
6. ✅ Stops after processing all pairs

**Note**: The app runs once per deployment. If you want it to run continuously on a schedule, consider:
- Railway Cron Jobs (for scheduled runs)
- External scheduler (cron, GitHub Actions, etc.)
- Keep app running with a loop (modify code)

## 🔄 Continuous Operation

If you want the app to run continuously (check pairs every X hours), you can:

1. **Add a loop in trader.py**:
```python
import time
while True:
    trader.run()
    time.sleep(3600)  # Wait 1 hour
```

2. **Use Railway Cron**:
   - Create a cron job that runs `python trader.py` every hour

3. **External Scheduler**:
   - Use GitHub Actions, AWS EventBridge, or similar

## ✅ Deployment Checklist

- [ ] All environment variables set in Railway
- [ ] Strategy files present in `strategy/` directory
- [ ] Tested locally with `--dry-run-notify`
- [ ] Repository pushed to GitHub
- [ ] Railway project created and linked
- [ ] First deployment successful
- [ ] Verified notifications are working

---

**Status**: ✅ Ready for Railway deployment (after setting environment variables)

