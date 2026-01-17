import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)


class WhaleTracker:
    """
    Tracks whale activity and large transactions using Glassnode API.
    Requires GLASSNODE_API_KEY environment variable.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GLASSNODE_API_KEY')
        self.base_url = 'https://api.glassnode.com/v1'
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({'x-api-key': self.api_key})
        else:
            logger.warning('Glassnode API key not configured. Using mock data.')
        
    def get_large_transactions(
        self,
        contract: str = 'bitcoin',
        limit: int = 20,
        hours: int = 24,
        min_value_usd: float = 1_000_000
    ) -> List[Dict]:
        """
        Get large whale transactions from Glassnode.
        
        Args:
            contract: 'bitcoin' or 'ethereum'
            limit: Number of transactions to return
            hours: Time window in hours
            min_value_usd: Minimum transaction value in USD
            
        Returns:
            List of transaction dictionaries
        """
        
        if not self.api_key:
            return self._get_mock_transactions(contract, limit)
        
        try:
            timestamp_since = int((datetime.utcnow() - timedelta(hours=hours)).timestamp())
            
            endpoint = f'{self.base_url}/transactions/large_transfers'
            params = {
                'a': contract,
                'i': '24h',
                's': timestamp_since,
                'limit': limit
            }
            
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            transactions = []
            
            if isinstance(data, list):
                for tx in data:
                    transactions.append({
                        'txid': tx.get('txid', ''),
                        'timestamp': tx.get('timestamp', ''),
                        'from_address': tx.get('from', '')[:10] + '...',
                        'to_address': tx.get('to', '')[:10] + '...',
                        'amount': float(tx.get('value', 0)) / (10 ** 8 if contract == 'bitcoin' else 10 ** 18),
                        'value_usd': float(tx.get('value_usd', 0)),
                        'type': self._classify_transaction(tx.get('from'), tx.get('to')),
                        'block_height': tx.get('block_height', '')
                    })
            
            return transactions[:limit]
            
        except requests.exceptions.RequestException as e:
            logger.error(f'Glassnode API error: {str(e)}')
            return self._get_mock_transactions(contract, limit)
    
    def get_whale_transactions_by_exchange(
        self,
        contract: str = 'bitcoin',
        limit: int = 50
    ) -> Dict:
        """
        Get whale transactions grouped by exchange.
        """
        
        if not self.api_key:
            return self._get_mock_exchange_transactions(contract)
        
        try:
            transactions = self.get_large_transactions(contract, limit * 2, 24)
            
            grouped = {
                'binance_inflow': [],
                'binance_outflow': [],
                'coinbase_inflow': [],
                'coinbase_outflow': [],
                'kraken_inflow': [],
                'kraken_outflow': [],
                'other': []
            }
            
            exchange_addresses = {
                'binance': ['0x00000000219ab540356cbb839cbe05303d7705fa'],
                'coinbase': ['0x02666cceed5b02a2cb9ecc14cb2ac1d8d0919ba1'],
                'kraken': ['0x2910543af39aba0cd09dbb2d0ff3aeb56e2c1347']
            }
            
            for tx in transactions:
                classified = False
                for exchange, addresses in exchange_addresses.items():
                    if tx['to_address'].lower() in [addr.lower() for addr in addresses]:
                        key = f'{exchange}_inflow'
                        grouped[key].append(tx)
                        classified = True
                        break
                    elif tx['from_address'].lower() in [addr.lower() for addr in addresses]:
                        key = f'{exchange}_outflow'
                        grouped[key].append(tx)
                        classified = True
                        break
                
                if not classified:
                    grouped['other'].append(tx)
            
            return grouped
            
        except Exception as e:
            logger.error(f'Error getting exchange transactions: {str(e)}')
            return self._get_mock_exchange_transactions(contract)
    
    def get_whale_wallets(
        self,
        contract: str = 'bitcoin',
        min_balance_btc: float = 1000
    ) -> List[Dict]:
        """
        Get list of whale wallets by balance.
        """
        
        if not self.api_key:
            return self._get_mock_whale_wallets(contract, min_balance_btc)
        
        try:
            endpoint = f'{self.base_url}/entities/top_holders'
            params = {
                'a': contract,
                'limit': 50
            }
            
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            wallets = []
            
            if isinstance(data, list):
                for holder in data:
                    balance = float(holder.get('balance', 0))
                    if balance >= min_balance_btc:
                        wallets.append({
                            'address': holder.get('address', '')[:16] + '...',
                            'balance': balance,
                            'balance_usd': float(holder.get('balance_usd', 0)),
                            'rank': holder.get('rank', 0),
                            'is_exchange': self._is_exchange_address(holder.get('address', ''))
                        })
            
            return wallets
            
        except requests.exceptions.RequestException as e:
            logger.error(f'Glassnode API error: {str(e)}')
            return self._get_mock_whale_wallets(contract, min_balance_btc)
    
    def get_transaction_volume(
        self,
        contract: str = 'bitcoin',
        hours: int = 24
    ) -> Dict:
        """
        Get total transaction volume metrics.
        """
        
        try:
            if not self.api_key:
                return self._get_mock_volume(contract, hours)
            
            endpoint = f'{self.base_url}/on_chain/transfers_volume'
            params = {
                'a': contract,
                'i': '1h',
                'limit': hours
            }
            
            response = self.session.get(endpoint, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, list) and len(data) > 0:
                total_volume = sum(float(item.get('v', 0)) for item in data)
                avg_volume = total_volume / len(data) if data else 0
                
                return {
                    'total_volume_24h': total_volume,
                    'avg_hourly_volume': avg_volume,
                    'peak_volume': max(float(item.get('v', 0)) for item in data) if data else 0,
                    'datapoints': len(data)
                }
            
            return self._get_mock_volume(contract, hours)
            
        except requests.exceptions.RequestException as e:
            logger.error(f'Error fetching volume: {str(e)}')
            return self._get_mock_volume(contract, hours)
    
    def _classify_transaction(self, from_addr: str, to_addr: str) -> str:
        """
        Classify transaction as inflow or outflow.
        """
        exchange_patterns = [
            '0x', '3j', '3', 'bc1',
            '0x00000000219ab540',
            '0x02666cceed5b02'
        ]
        
        to_is_exchange = any(pattern in str(to_addr).lower() for pattern in exchange_patterns)
        from_is_exchange = any(pattern in str(from_addr).lower() for pattern in exchange_patterns)
        
        if to_is_exchange and not from_is_exchange:
            return 'inflow'
        elif from_is_exchange and not to_is_exchange:
            return 'outflow'
        else:
            return 'transfer'
    
    def _is_exchange_address(self, address: str) -> bool:
        """
        Check if address belongs to known exchange.
        """
        known_exchanges = [
            '0x00000000219ab540356cbb839cbe05303d7705fa',
            '0x02666cceed5b02a2cb9ecc14cb2ac1d8d0919ba1',
            '0x2910543af39aba0cd09dbb2d0ff3aeb56e2c1347'
        ]
        return any(address.lower() in exc.lower() for exc in known_exchanges)
    
    def _get_mock_transactions(self, contract: str, limit: int) -> List[Dict]:
        """
        Return mock transaction data for testing.
        """
        mock_data = [
            {
                'txid': 'mock_tx_001',
                'timestamp': (datetime.utcnow() - timedelta(hours=1)).isoformat(),
                'from_address': '1A1z7agoat...xxx',
                'to_address': '3J98t1wp48...yyy',
                'amount': 150.5,
                'value_usd': 6_321_000,
                'type': 'outflow',
                'block_height': 821234
            },
            {
                'txid': 'mock_tx_002',
                'timestamp': (datetime.utcnow() - timedelta(hours=2)).isoformat(),
                'from_address': '1BoatSLRHtK...aaa',
                'to_address': '3QJmV3qsKtD...bbb',
                'amount': 250.75,
                'value_usd': 10_531_750,
                'type': 'inflow',
                'block_height': 821200
            },
            {
                'txid': 'mock_tx_003',
                'timestamp': (datetime.utcnow() - timedelta(hours=4)).isoformat(),
                'from_address': '1dice8EMCNqASmnGvMyafcFi3vkagxLFZ...ccc',
                'to_address': '1A1z7agoatK...ddd',
                'amount': 75.25,
                'value_usd': 3_157_500,
                'type': 'transfer',
                'block_height': 821100
            }
        ]
        return mock_data[:limit]
    
    def _get_mock_exchange_transactions(self, contract: str) -> Dict:
        """
        Return mock exchange transaction data.
        """
        return {
            'binance_inflow': [
                {
                    'amount': 500.0,
                    'value_usd': 21_000_000,
                    'timestamp': datetime.utcnow().isoformat()
                }
            ],
            'binance_outflow': [
                {
                    'amount': 300.0,
                    'value_usd': 12_600_000,
                    'timestamp': datetime.utcnow().isoformat()
                }
            ],
            'coinbase_inflow': [],
            'coinbase_outflow': [],
            'kraken_inflow': [],
            'kraken_outflow': [],
            'other': []
        }
    
    def _get_mock_whale_wallets(self, contract: str, min_balance: float) -> List[Dict]:
        """
        Return mock whale wallet data.
        """
        return [
            {
                'address': '1A1z7agoatK...',
                'balance': 161500.0,
                'balance_usd': 6_783_000_000,
                'rank': 1,
                'is_exchange': False
            },
            {
                'address': '1BoatSLRHtK...',
                'balance': 130000.0,
                'balance_usd': 5_460_000_000,
                'rank': 2,
                'is_exchange': False
            }
        ]
    
    def _get_mock_volume(self, contract: str, hours: int) -> Dict:
        """
        Return mock volume data.
        """
        return {
            'total_volume_24h': 45_000_000_000,
            'avg_hourly_volume': 1_875_000_000,
            'peak_volume': 3_500_000_000,
            'datapoints': hours
        }
