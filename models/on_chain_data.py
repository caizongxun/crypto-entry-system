import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import requests
from pathlib import Path


class OnChainDataProvider:
    """Fetch and process on-chain data from multiple sources."""

    def __init__(self, symbol: str = 'BTCUSDT'):
        self.symbol = symbol
        self.base_symbol = symbol.replace('USDT', '').upper()
        self.glassnode_api_key = None
        self.alternative_api_key = None

    def set_api_keys(self, glassnode_key: Optional[str] = None, alternative_key: Optional[str] = None):
        """Set API keys for data providers."""
        self.glassnode_api_key = glassnode_key
        self.alternative_api_key = alternative_key

    def get_whale_transactions(self) -> Dict:
        """Get large whale transaction data."""
        try:
            url = "https://api.glassnode.com/v1/metrics/transactions/large_transactions_count"
            params = {
                'a': self.base_symbol.lower(),
                'api_key': self.glassnode_api_key,
                'f': 'json'
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'success',
                    'whale_activity': data.get('data', []),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'status': 'error', 'message': 'Failed to fetch whale transactions'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_miner_revenue(self) -> Dict:
        """Get miner revenue data."""
        try:
            url = "https://api.glassnode.com/v1/metrics/mining/revenue_sum"
            params = {
                'a': self.base_symbol.lower(),
                'api_key': self.glassnode_api_key,
                'f': 'json'
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'success',
                    'miner_revenue': data.get('data', []),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'status': 'error', 'message': 'Failed to fetch miner revenue'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_exchange_flows(self) -> Dict:
        """Get exchange inflow/outflow data."""
        try:
            url = "https://api.glassnode.com/v1/metrics/exchanges/exchange_net_flow"
            params = {
                'a': self.base_symbol.lower(),
                'api_key': self.glassnode_api_key,
                'f': 'json'
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'success',
                    'exchange_flows': data.get('data', []),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'status': 'error', 'message': 'Failed to fetch exchange flows'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_holder_distribution(self) -> Dict:
        """Get holder distribution data."""
        try:
            url = "https://api.glassnode.com/v1/metrics/holders/distribution"
            params = {
                'a': self.base_symbol.lower(),
                'api_key': self.glassnode_api_key,
                'f': 'json'
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'success',
                    'holder_distribution': data.get('data', {}),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {'status': 'error', 'message': 'Failed to fetch holder distribution'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def calculate_on_chain_signal(self, whale_activity: float, exchange_flow: float,
                                   holder_concentration: float) -> Dict:
        """Calculate on-chain trading signal.
        
        Returns:
            Dict with signal strength (-1 to 1) and reasoning
        """
        signal = 0
        reasoning = []
        
        if whale_activity > 0.7:
            signal += 0.2
            reasoning.append('High whale activity detected')
        elif whale_activity < 0.3:
            signal -= 0.2
            reasoning.append('Low whale activity')
            
        if exchange_flow < -0.5:
            signal += 0.3
            reasoning.append('Coins flowing out of exchanges (bullish)')
        elif exchange_flow > 0.5:
            signal -= 0.3
            reasoning.append('Coins flowing into exchanges (bearish)')
            
        if holder_concentration > 0.7:
            signal += 0.1
            reasoning.append('High holder concentration')
        elif holder_concentration < 0.3:
            signal -= 0.1
            reasoning.append('Low holder concentration')
        
        signal = np.clip(signal, -1, 1)
        
        return {
            'signal_strength': signal,
            'signal_direction': 'BULLISH' if signal > 0.2 else 'BEARISH' if signal < -0.2 else 'NEUTRAL',
            'confidence': abs(signal),
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        }

    def get_all_on_chain_data(self) -> Dict:
        """Fetch all on-chain data."""
        result = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'data': {}
        }
        
        if self.glassnode_api_key:
            result['data']['whale_transactions'] = self.get_whale_transactions()
            result['data']['miner_revenue'] = self.get_miner_revenue()
            result['data']['exchange_flows'] = self.get_exchange_flows()
            result['data']['holder_distribution'] = self.get_holder_distribution()
        else:
            result['data'] = {
                'status': 'warning',
                'message': 'Glassnode API key not configured. Please set API keys.'
            }
        
        return result
