#!/usr/bin/env python3
"""
ForexFactory Selenium Scraper
Uses Selenium to scrape ForexFactory economic calendar.
"""

import argparse
import json
import sys
from datetime import datetime
from typing import List, Dict, Any, Optional

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service


class ForexFactoryScraper:
    """Scrapes ForexFactory using Selenium"""
    
    def __init__(self, headless: bool = True):
        """Initialize Chrome driver"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        self.driver = webdriver.Chrome(options=chrome_options)
        self.driver.implicitly_wait(10)
    
    def fetch_news(self, date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Fetch forex news from ForexFactory
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
        
        Returns:
            List of news events
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        print(f"🔄 Scraping ForexFactory for {date}...", file=sys.stderr)
        
        try:
            # Format date for ForexFactory (e.g., oct29.2025)
            dt = datetime.strptime(date, "%Y-%m-%d")
            month_abbr = dt.strftime("%b").lower()
            ff_date = f"{month_abbr}{dt.day}.{dt.year}"
            
            url = f"https://www.forexfactory.com/calendar?day={ff_date}"
            
            print(f"🌐 Loading: {url}", file=sys.stderr)
            self.driver.get(url)
            
            # Wait for calendar table to load
            wait = WebDriverWait(self.driver, 20)
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "calendar__table")))
            
            print("📊 Parsing calendar data...", file=sys.stderr)
            
            events = []
            rows = self.driver.find_elements(By.CLASS_NAME, "calendar__row")
            
            current_date = date
            
            for row in rows:
                try:
                    # Check for date row
                    date_cells = row.find_elements(By.CLASS_NAME, "calendar__date")
                    if date_cells and date_cells[0].text.strip():
                        current_date = date_cells[0].text.strip()
                    
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
                    
                    # Get actual value
                    actual_cells = row.find_elements(By.CLASS_NAME, "calendar__actual")
                    actual = actual_cells[0].text.strip() if actual_cells else None
                    
                    # Get forecast
                    forecast_cells = row.find_elements(By.CLASS_NAME, "calendar__forecast")
                    forecast = forecast_cells[0].text.strip() if forecast_cells else None
                    
                    # Get previous
                    previous_cells = row.find_elements(By.CLASS_NAME, "calendar__previous")
                    previous = previous_cells[0].text.strip() if previous_cells else None
                    
                    event = {
                        "date": current_date,
                        "time": time_str,
                        "currency": currency,
                        "impact": impact,
                        "title": title,
                        "actual": actual if actual else None,
                        "forecast": forecast if forecast else None,
                        "previous": previous if previous else None
                    }
                    
                    events.append(event)
                    
                except Exception as e:
                    continue
            
            print(f"✅ Scraped {len(events)} events", file=sys.stderr)
            return events
            
        except Exception as e:
            print(f"❌ Error: {e}", file=sys.stderr)
            return []
    
    def filter_events(
        self,
        events: List[Dict[str, Any]],
        currencies: Optional[List[str]] = None,
        importance_levels: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Filter events"""
        filtered = events
        
        if currencies:
            currencies_upper = [c.upper() for c in currencies]
            filtered = [e for e in filtered if e["currency"] in currencies_upper]
        
        if importance_levels:
            importance_lower = [i.lower() for i in importance_levels]
            filtered = [e for e in filtered if e["impact"] in importance_lower]
        
        return filtered
    
    def close(self):
        """Close browser"""
        if self.driver:
            self.driver.quit()
    
    def get_news(
        self,
        date: Optional[str] = None,
        currencies: Optional[List[str]] = None,
        importance_levels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Get forex news"""
        try:
            events = self.fetch_news(date)
            
            if currencies or importance_levels:
                events = self.filter_events(events, currencies, importance_levels)
            
            return {
                "success": True,
                "source": "ForexFactory",
                "date": date or datetime.now().strftime("%Y-%m-%d"),
                "total_events": len(events),
                "events": events
            }
        finally:
            self.close()


def print_table(events: List[Dict[str, Any]]):
    """Print events in table format"""
    if not events:
        print("\n❌ No events found\n")
        return
    
    # Try to use tabulate if available
    try:
        from tabulate import tabulate
        
        headers = ["Time", "Currency", "Impact", "Event", "Actual", "Forecast", "Previous"]
        rows = []
        
        for event in events:
            # Color code impact
            impact = event['impact'].upper()
            if impact == "HIGH":
                impact = "🔴 HIGH"
            elif impact == "MEDIUM":
                impact = "🟠 MED"
            else:
                impact = "🟡 LOW"
            
            rows.append([
                event['time'],
                event['currency'],
                impact,
                event['title'][:50] + "..." if len(event['title']) > 50 else event['title'],
                event['actual'] or "-",
                event['forecast'] or "-",
                event['previous'] or "-"
            ])
        
        print("\n" + tabulate(rows, headers=headers, tablefmt="grid") + "\n")
        
    except ImportError:
        # Fallback to simple table
        print("\n" + "="*120)
        print(f"{'Time':<10} {'Cur':<5} {'Impact':<8} {'Event':<40} {'Actual':<10} {'Forecast':<10} {'Previous':<10}")
        print("="*120)
        
        for event in events:
            impact = event['impact'].upper()[:3]
            if event['impact'] == "high":
                impact = "🔴" + impact
            elif event['impact'] == "medium":
                impact = "🟠" + impact
            else:
                impact = "🟡" + impact
            
            print(f"{event['time']:<10} {event['currency']:<5} {impact:<8} {event['title'][:40]:<40} "
                  f"{(event['actual'] or '-'):<10} {(event['forecast'] or '-'):<10} {(event['previous'] or '-'):<10}")
        
        print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Scrape ForexFactory with Selenium")
    
    parser.add_argument("--date", type=str, help="Date (YYYY-MM-DD)")
    parser.add_argument("--currency", type=str, help="Currencies (USD,EUR,GBP)")
    parser.add_argument("--importance", type=str, help="Importance (low,medium,high)")
    parser.add_argument("--output", type=str, help="Output JSON file")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    parser.add_argument("--pretty", action="store_true", help="Pretty JSON")
    parser.add_argument("--show-browser", action="store_true", help="Show browser (not headless)")
    
    args = parser.parse_args()
    
    # Parse filters
    currencies = [c.strip() for c in args.currency.split(",")] if args.currency else None
    importance_levels = [i.strip() for i in args.importance.split(",")] if args.importance else None
    
    # Scrape
    scraper = ForexFactoryScraper(headless=not args.show_browser)
    result = scraper.get_news(date=args.date, currencies=currencies, importance_levels=importance_levels)
    
    # Output
    if args.output:
        # Save JSON to file
        indent = 2 if args.pretty else None
        json_output = json.dumps(result, indent=indent, ensure_ascii=False)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(json_output)
        print(f"✅ Saved to {args.output}", file=sys.stderr)
        print(f"📊 Total: {result['total_events']}", file=sys.stderr)
    elif args.json:
        # Print JSON
        indent = 2 if args.pretty else None
        json_output = json.dumps(result, indent=indent, ensure_ascii=False)
        print(json_output)
    else:
        # Print table (default)
        print(f"\n📅 Date: {result['date']}")
        print(f"📊 Total Events: {result['total_events']}")
        print_table(result['events'])


if __name__ == "__main__":
    main()
