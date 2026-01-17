#!/usr/bin/env python3
"""
Whale Movements Data Collection Script

This script automatically collects daily whale movement data from multiple sources:
- Whale Alert (large transactions)
- Glassnode (on-chain metrics)
- CryptoQuant (exchange flows)

Usage:
    python scripts/update_whale_data.py --source whale-alert
    python scripts/update_whale_data.py --source glassnode
    python scripts/update_whale_data.py --source cryptoquant
    python scripts/update_whale_data.py --all

Scheduling (Cron):
    0 1 * * * cd /path/to/crypto-entry-system && python scripts/update_whale_data.py --all

Required API Keys (set as environment variables):
    WHALE_ALERT_API_KEY
    GLASSNODE_API_KEY
    CRYPTOQUANT_API_KEY
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import argparse
from pathlib import Path

# Configuration
DATA_FILE = "data_collection/daily_whale_movements.csv"
BACKUP_FILE = "data_collection/daily_whale_movements_backup.csv"

# API Endpoints
WHALE_ALERT_API = "https://api.whale-alert.io/v1/transactions"
GLASSNODE_API = "https://api.glassnode.com/v1/metrics"
CRYPTOQUANT_API = "https://api.cryptoquant.com/v1"

class WhaleDataCollector:
    """
    Collects whale transaction data from multiple sources
    """
    
    def __init__(self):
        self.whale_alert_key = os.getenv('WHALE_ALERT_API_KEY')
        self.glassnode_key = os.getenv('GLASSNODE_API_KEY')
        self.cryptoquant_key = os.getenv('CRYPTOQUANT_API_KEY')
    
    def validate_keys(self):
        """
        Validate that required API keys are set
        """
        missing_keys = []
        if not self.whale_alert_key:
            missing_keys.append('WHALE_ALERT_API_KEY')
        if not self.glassnode_key:
            missing_keys.append('GLASSNODE_API_KEY')
        if not self.cryptoquant_key:
            missing_keys.append('CRYPTOQUANT_API_KEY')
        
        if missing_keys:
            print(f"Warning: Missing API keys: {', '.join(missing_keys)}")
            print("Some data sources will be skipped.")
            return len(missing_keys) < 3
        return True
    
    def fetch_whale_alert_data(self, days=1):
        """
        Fetch whale transactions from Whale Alert API
        
        Args:
            days: Number of days to look back
        
        Returns:
            list: Formatted whale transactions
        """
        if not self.whale_alert_key:
            print("Whale Alert API key not set. Skipping.")
            return []
        
        try:
            records = []
            params = {
                'api_key': self.whale_alert_key,
                'min_value': 500000,  # 500k USD minimum
                'limit': 100
            }
            
            # Fetch transactions for each blockchain
            for blockchain in ['bitcoin', 'ethereum', 'ripple']:
                params['blockchain'] = blockchain
                response = requests.get(WHALE_ALERT_API, params=params, timeout=10)
                
                if response.status_code != 200:
                    print(f"Whale Alert API error for {blockchain}: {response.status_code}")
                    continue
                
                data = response.json()
                if 'result' in data and 'transactions' in data['result']:
                    for tx in data['result']['transactions']:
                        records.append(self.format_whale_alert_record(tx))
            
            print(f"Fetched {len(records)} records from Whale Alert")
            return records
        
        except Exception as e:
            print(f"Error fetching Whale Alert data: {e}")
            return []
    
    def format_whale_alert_record(self, tx):
        """
        Format Whale Alert transaction record
        """
        dt = datetime.fromtimestamp(tx.get('timestamp', int(datetime.now().timestamp())))
        
        return {
            'date': dt.strftime('%Y-%m-%d'),
            'timestamp': int(tx.get('timestamp', int(datetime.now().timestamp()))),
            'blockchain': tx.get('blockchain', 'unknown'),
            'transaction_type': tx.get('transaction_type', 'transfer'),
            'from_address': tx.get('from', {}).get('owner', 'unknown'),
            'to_address': tx.get('to', {}).get('owner', 'unknown'),
            'amount': float(tx.get('amount', 0)),
            'usd_value': float(tx.get('amount_usd', 0)),
            'whale_classification': 'Whale Alert Signal',
            'exchange_flow': self.determine_flow(tx),
            'transaction_hash': tx.get('hash', ''),
            'data_source': 'Whale Alert',
            'notes': tx.get('from', {}).get('tag', '') if tx.get('from') else ''
        }
    
    def determine_flow(self, tx):
        """
        Determine if transaction is inflow or outflow
        """
        to_tag = tx.get('to', {}).get('tag', '').lower() if tx.get('to') else ''
        from_tag = tx.get('from', {}).get('tag', '').lower() if tx.get('from') else ''
        
        if 'exchange' in to_tag:
            return 'outflow'
        elif 'exchange' in from_tag:
            return 'inflow'
        else:
            return 'transfer'
    
    def fetch_glassnode_data(self):
        """
        Fetch whale metrics from Glassnode API
        
        Returns:
            list: Formatted whale metrics
        """
        if not self.glassnode_key:
            print("Glassnode API key not set. Skipping.")
            return []
        
        try:
            records = []
            
            # Key metrics to fetch
            metrics = [
                'distribution_whale_entities',
                'supply_active_addresses',
            ]
            
            for metric in metrics:
                params = {
                    'api_key': self.glassnode_key,
                    'a': 'eth',
                    'timestamp_format': 'iso8601'
                }
                
                url = f"{GLASSNODE_API}/{metric}"
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code != 200:
                    print(f"Glassnode API error for {metric}: {response.status_code}")
                    continue
                
                data = response.json()
                if 'data' in data:
                    for entry in data['data'][-1:]:  # Get latest entry
                        records.append(self.format_glassnode_record(entry, metric))
            
            print(f"Fetched {len(records)} records from Glassnode")
            return records
        
        except Exception as e:
            print(f"Error fetching Glassnode data: {e}")
            return []
    
    def format_glassnode_record(self, entry, metric):
        """
        Format Glassnode record
        """
        ts = entry.get('t', int(datetime.now().timestamp()))
        if isinstance(ts, str):
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            ts = int(dt.timestamp())
        else:
            dt = datetime.fromtimestamp(ts)
        
        return {
            'date': dt.strftime('%Y-%m-%d'),
            'timestamp': ts,
            'blockchain': 'ethereum',
            'transaction_type': 'metric',
            'from_address': 'glassnode_metric',
            'to_address': 'whale_analysis',
            'amount': float(entry.get('v', 0)),
            'usd_value': 0,
            'whale_classification': 'Large ETH Holder',
            'exchange_flow': 'transfer',
            'transaction_hash': '',
            'data_source': 'Glassnode',
            'notes': metric
        }
    
    def fetch_cryptoquant_data(self):
        """
        Fetch exchange flow data from CryptoQuant API
        
        Returns:
            list: Formatted exchange flow records
        """
        if not self.cryptoquant_key:
            print("CryptoQuant API key not set. Skipping.")
            return []
        
        try:
            records = []
            headers = {'Authorization': f'Bearer {self.cryptoquant_key}'}
            
            # Fetch exchange netflow
            url = f"{CRYPTOQUANT_API}/btc/exchange-flows/netflow-total"
            params = {
                'window': 'day',
                'start_time': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'end_time': datetime.now().strftime('%Y-%m-%d')
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                print(f"CryptoQuant API error: {response.status_code}")
                return []
            
            data = response.json()
            if 'data' in data:
                for entry in data['data'][-1:]:  # Get latest entry
                    records.append(self.format_cryptoquant_record(entry))
            
            print(f"Fetched {len(records)} records from CryptoQuant")
            return records
        
        except Exception as e:
            print(f"Error fetching CryptoQuant data: {e}")
            return []
    
    def format_cryptoquant_record(self, entry):
        """
        Format CryptoQuant record
        """
        ts = entry.get('timestamp', int(datetime.now().timestamp()))
        dt = datetime.fromtimestamp(ts)
        
        netflow = float(entry.get('netflow_total', 0))
        flow_type = 'inflow' if netflow > 0 else 'outflow' if netflow < 0 else 'neutral'
        
        return {
            'date': dt.strftime('%Y-%m-%d'),
            'timestamp': ts,
            'blockchain': 'bitcoin',
            'transaction_type': 'exchange_flow',
            'from_address': 'cryptoquant_aggregate',
            'to_address': 'exchange_network',
            'amount': abs(netflow),
            'usd_value': 0,
            'whale_classification': 'Exchange Flow',
            'exchange_flow': flow_type,
            'transaction_hash': '',
            'data_source': 'CryptoQuant',
            'notes': f'Daily netflow: {netflow:.2f} BTC'
        }
    
    def append_to_csv(self, new_records):
        """
        Append new records to CSV file (avoiding duplicates)
        """
        if not new_records:
            print("No new records to append.")
            return False
        
        try:
            # Read existing data
            if os.path.exists(DATA_FILE):
                df_existing = pd.read_csv(DATA_FILE, comment='#')
                df_existing = df_existing[df_existing['date'].notna()]
            else:
                df_existing = pd.DataFrame()
            
            # Convert new records to DataFrame
            df_new = pd.DataFrame(new_records)
            
            # Combine and remove duplicates
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.drop_duplicates(
                subset=['date', 'blockchain', 'transaction_hash'],
                keep='last'
            )
            df_combined = df_combined.sort_values('date')
            
            # Backup existing file
            if os.path.exists(DATA_FILE):
                os.rename(DATA_FILE, BACKUP_FILE)
            
            # Write updated CSV
            df_combined.to_csv(DATA_FILE, index=False)
            
            print(f"Successfully updated {DATA_FILE}")
            print(f"Total records: {len(df_combined)}")
            return True
        
        except Exception as e:
            print(f"Error writing to CSV: {e}")
            if os.path.exists(BACKUP_FILE):
                os.rename(BACKUP_FILE, DATA_FILE)
            return False
    
    def collect_all(self):
        """
        Collect data from all sources
        """
        print(f"Starting whale data collection at {datetime.now()}")
        
        all_records = []
        all_records.extend(self.fetch_whale_alert_data())
        all_records.extend(self.fetch_glassnode_data())
        all_records.extend(self.fetch_cryptoquant_data())
        
        if all_records:
            return self.append_to_csv(all_records)
        else:
            print("No data collected from any source.")
            return False

def main():
    """
    Main execution function
    """
    parser = argparse.ArgumentParser(description='Collect whale movement data')
    parser.add_argument('--source', choices=['whale-alert', 'glassnode', 'cryptoquant'],
                       help='Specific data source to collect from')
    parser.add_argument('--all', action='store_true', help='Collect from all sources')
    args = parser.parse_args()
    
    collector = WhaleDataCollector()
    
    if not collector.validate_keys():
        print("Critical API keys missing. Exiting.")
        return False
    
    if args.all or (not args.source):
        return collector.collect_all()
    elif args.source == 'whale-alert':
        records = collector.fetch_whale_alert_data()
        return collector.append_to_csv(records) if records else False
    elif args.source == 'glassnode':
        records = collector.fetch_glassnode_data()
        return collector.append_to_csv(records) if records else False
    elif args.source == 'cryptoquant':
        records = collector.fetch_cryptoquant_data()
        return collector.append_to_csv(records) if records else False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
