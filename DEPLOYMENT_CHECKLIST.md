# Railway Deployment Checklist ✅

## Step-by-Step Deployment Guide

### ✅ Step 1: Code Preparation
- [x] All files committed to Git
- [x] Code pushed to GitHub: `tyl-99/tyl-bot`

### ⏳ Step 2: Railway Setup
- [ ] Go to https://railway.app
- [ ] Sign in (GitHub recommended)
- [ ] Click "New Project"
- [ ] Select "Deploy from GitHub repo"
- [ ] Choose repository: `tyl-99/tyl-bot`
- [ ] Railway will auto-detect Python and start building

### ⏳ Step 3: Environment Variables
Go to your Railway project → **Variables** tab → Add each:

**REQUIRED (must set):**
```
CTRADER_CLIENT_ID=________________
CTRADER_CLIENT_SECRET=________________
CTRADER_ACCOUNT_ID=________________
GEMINI_APIKEY=________________
```

**OPTIONAL (has hardcoded fallback in trader.py):**
```
PUSHOVER_APP_TOKEN=________________ (optional - will use fallback if not set)
PUSHOVER_USER_KEY=________________ (optional - will use fallback if not set)
```

**Note:** Pushover credentials are already hardcoded in `trader.py` as fallback, so you don't need to set them unless you want to use different credentials.

### ⏳ Step 4: Monitor Deployment
- [ ] Check **Deployments** tab for build progress
- [ ] Watch build logs for errors
- [ ] Wait for deployment to complete (2-5 minutes)

### ⏳ Step 5: Verify Running
- [ ] Check **Logs** tab
- [ ] Should see: "Connected to cTrader"
- [ ] Should see: "User authenticated"
- [ ] Should see: "Scraping ForexFactory news..."
- [ ] Should see: "Processing [pair]"
- [ ] Receive Pushover notifications when signals found

## 🐛 Troubleshooting

### Build Fails?
- Check Railway build logs
- Verify `requirements.txt` is correct
- Check Python version in `runtime.txt`

### Selenium/Chrome Errors?
- Check Railway logs for ChromeDriver path
- Verify `nixpacks.toml` includes chromium packages
- Should auto-configure via Nixpacks

### App Crashes?
- Check all environment variables are set
- Verify API credentials are correct
- Check Railway logs for specific error messages

### No Notifications?
- Verify Pushover credentials
- Check if signals are being generated (check logs)
- Verify Pushover app token and user key

## 📊 Expected Log Output

```
INFO: Connected to cTrader
INFO: User authenticated
INFO: Scraping ForexFactory news...
INFO: Loading ForexFactory: https://www.forexfactory.com/calendar?day=...
INFO: ✅ Scraped X total events from ForexFactory
INFO: Processing EUR/USD
INFO: Processing GBP/USD
...
```

## 🔄 After Deployment

The app will:
1. ✅ Run once per deployment
2. ✅ Process all 6 pairs sequentially
3. ✅ Send notifications for any trade signals
4. ✅ Stop after completing all pairs

**To run continuously**, you'll need to:
- Add a loop in the code, OR
- Set up Railway Cron job, OR
- Use external scheduler

---

**Status**: Ready to deploy! Follow steps 2-5 above.

