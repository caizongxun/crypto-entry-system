#!/usr/bin/env python3
"""
Market Fear & Greed Sentiment Data Collection Script

This script automatically collects daily market sentiment data from Alternative.me
and appends it to the CSV file in data_collection/ folder.

Usage:
    python scripts/update_market_sentiment.py

Scheduling (Cron):
    0 0 * * * cd /path/to/crypto-entry-system && python scripts/update_market_sentiment.py
"""

import requests
import pandas as pd
from datetime import datetime
import os
import sys
from pathlib import Path

# Configuration
ALTERNATIVE_ME_API = "https://api.alternative.me/fng/"
DATA_FILE = "data_collection/market_fear_greed_sentiment.csv"
BACKUP_FILE = "data_collection/market_fear_greed_sentiment_backup.csv"

def fetch_sentiment_data(limit=1):
    """
    Fetch Fear & Greed Index data from Alternative.me API
    
    Args:
        limit: Number of records to fetch (0 = all, 1 = latest)
    
    Returns:
        list: List of sentiment data dictionaries
    """
    try:
        params = {
            'limit': limit,
            'format': 'json',
            'date_format': 'world'
        }
        response = requests.get(ALTERNATIVE_ME_API, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        if 'data' in data:
            return data['data']
        else:
            print(f"Error: Unexpected API response format: {data}")
            return []
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Alternative.me API: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def sentiment_to_classification(value):
    """
    Convert numeric sentiment value to classification
    
    Args:
        value: Integer from 0-100
    
    Returns:
        str: Sentiment classification
    """
    value = int(value)
    if value <= 24:
        return "Extreme Fear"
    elif value <= 46:
        return "Fear"
    elif value <= 54:
        return "Neutral"
    elif value <= 75:
        return "Greed"
    else:
        return "Extreme Greed"

def format_record(record):
    """
    Format API record to CSV format
    
    Args:
        record: Dictionary from API response
    
    Returns:
        dict: Formatted record for CSV
    """
    timestamp = int(record['timestamp'])
    value = record['value']
    
    # Convert timestamp to datetime
    dt = datetime.fromtimestamp(timestamp)
    date_str = dt.strftime('%Y-%m-%d')
    
    return {
        'date': date_str,
        'timestamp': timestamp,
        'fear_greed_value': int(value),
        'sentiment_classification': sentiment_to_classification(value),
        'data_source': 'Alternative.me',
        'notes': ''
    }

def append_to_csv(new_records):
    """
    Append new records to CSV file (avoiding duplicates)
    
    Args:
        new_records: List of formatted records
    
    Returns:
        bool: Success status
    """
    try:
        # Check if file exists
        if os.path.exists(DATA_FILE):
            # Read existing data
            df_existing = pd.read_csv(DATA_FILE, comment='#')
            df_existing = df_existing[df_existing['date'].notna()]  # Remove header comments
        else:
            df_existing = pd.DataFrame()
        
        # Convert new records to DataFrame
        df_new = pd.DataFrame(new_records)
        
        # Combine and remove duplicates (by date)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=['date'], keep='last')
        df_combined = df_combined.sort_values('date')
        
        # Backup existing file
        if os.path.exists(DATA_FILE):
            os.rename(DATA_FILE, BACKUP_FILE)
        
        # Write updated CSV
        df_combined.to_csv(DATA_FILE, index=False)
        
        # Append schema comments
        with open(DATA_FILE, 'a') as f:
            f.write("\n# Schema Description:\n")
            f.write("# date: YYYY-MM-DD format\n")
            f.write("# timestamp: Unix timestamp (seconds)\n")
            f.write("# fear_greed_value: Integer from 0-100\n")
            f.write("#   0-24: Extreme Fear\n")
            f.write("#   25-46: Fear\n")
            f.write("#   47-54: Neutral\n")
            f.write("#   55-75: Greed\n")
            f.write("#   76-100: Extreme Greed\n")
            f.write("# sentiment_classification: Categorical sentiment\n")
            f.write("# data_source: Data provider (Alternative.me)\n")
            f.write("# notes: Additional context or market events\n")
        
        print(f"Successfully updated {DATA_FILE}")
        print(f"Total records: {len(df_combined)}")
        return True
    
    except Exception as e:
        print(f"Error writing to CSV: {e}")
        # Restore backup if failed
        if os.path.exists(BACKUP_FILE):
            os.rename(BACKUP_FILE, DATA_FILE)
        return False

def main():
    """
    Main execution function
    """
    print(f"Starting market sentiment data collection at {datetime.now()}")
    
    # Fetch latest data
    print("Fetching data from Alternative.me API...")
    records = fetch_sentiment_data(limit=1)
    
    if not records:
        print("No data fetched. Exiting.")
        return False
    
    print(f"Fetched {len(records)} record(s)")
    
    # Format records
    formatted_records = [format_record(r) for r in records]
    
    # Append to CSV
    if append_to_csv(formatted_records):
        print("Update completed successfully!")
        print(f"Latest sentiment value: {formatted_records[0]['fear_greed_value']} ({formatted_records[0]['sentiment_classification']})")
        return True
    else:
        print("Update failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
