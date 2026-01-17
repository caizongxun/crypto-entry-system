import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import os

logger = logging.getLogger(__name__)


class ExchangeFlowAnalyzer:
    """
    Analyzes cryptocurrency flows to and from exchanges using Glassnode.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('GLASSNODE_API_KEY')
        self.base_url = 'https://api.glassnode.com/v1'
        self.session = requests.Session()
        
        if self.api_key:
            self.session.headers.update({'x-api-key': self.api_key})
        else:
            logger.warning('Glassnode API key not configured. Using mock data.')
        
        self.exchanges = {
            'binance': {
                'addresses': ['0x28c6c06298d161e15667f0f124c5f6356763e8a3f07c666c', '0xf977814e90da44bfa03b6295602c0d0e52527ec3d'],
                'name': 'Binance'
            },
            'coinbase': {
                'addresses': ['0xddfabcde', '0x503828976d22510aad0201f7e4b2d759'],
                'name': 'Coinbase'
            },
            'kraken': {
                'addresses': ['0x2910543af39aba0cd09dbb2d0ff3aeb56e2c1347'],
                'name': 'Kraken'
            },
            'huobi': {
                'addresses': ['0xec3281124d4a4311a82628f48f592a8fb4ba3c5d'],
                'name': 'Huobi'
            },
            'ftx': {
                'addresses': ['0x9f7f4e7c'],
                'name': 'FTX'
            }
        }
    
    def get_exchange_flows(
        self,
        contract: str = 'bitcoin',
        hours: int = 24
    ) -> List[Dict]:
        """
        Get exchange inflow/outflow data.
        
        Args:
            contract: 'bitcoin' or 'ethereum'
            hours: Time window in hours
            
        Returns:
            List of exchange flow data
        """
        
        if not self.api_key:
            return self._get_mock_flows(contract)
        
        try:
            flows = []
            
            for exchange_key, exchange_info in self.exchanges.items():
                try:
                    inflow = self._get_exchange_inflow(
                        contract,
                        exchange_info['addresses'],
                        hours
                    )
                    outflow = self._get_exchange_outflow(
                        contract,
                        exchange_info['addresses'],
                        hours
                    )
                    total_reserve = self._get_exchange_reserve(
                        contract,
                        exchange_info['addresses']
                    )
                    
                    flows.append({
                        'exchange_name': exchange_info['name'],
                        'exchange_key': exchange_key,
                        'inflow': inflow,
                        'outflow': outflow,
                        'net_flow': inflow - outflow,
                        'total_reserve': total_reserve,
                        'flow_type': 'inflow' if inflow > outflow else 'outflow',
                        'timestamp': datetime.utcnow().isoformat()
                    })
                except Exception as e:
                    logger.error(f'Error fetching flows for {exchange_key}: {str(e)}')
                    continue
            
            return flows if flows else self._get_mock_flows(contract)
            
        except Exception as e:
            logger.error(f'Error in get_exchange_flows: {str(e)}')
            return self._get_mock_flows(contract)
    
    def _get_exchange_inflow(
        self,
        contract: str,
        addresses: List[str],
        hours: int
    ) -> float:
        """
        Get total inflow to exchange in last N hours.
        """
        try:
            if not self.api_key:
                return 0.0
            
            timestamp_since = int((datetime.utcnow() - timedelta(hours=hours)).timestamp())
            
            total_inflow = 0.0
            for address in addresses:
                endpoint = f'{self.base_url}/on_chain/transfers_to_address'
                params = {
                    'a': contract,
                    'address': address,
                    's': timestamp_since,
                    'i': '24h'
                }
                
                response = self.session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if isinstance(data, list):
                    for item in data:
                        total_inflow += float(item.get('v', 0))
            
            divisor = 10 ** 8 if contract == 'bitcoin' else 10 ** 18
            return total_inflow / divisor
            
        except Exception as e:
            logger.error(f'Error getting inflow: {str(e)}')
            return 0.0
    
    def _get_exchange_outflow(
        self,
        contract: str,
        addresses: List[str],
        hours: int
    ) -> float:
        """
        Get total outflow from exchange in last N hours.
        """
        try:
            if not self.api_key:
                return 0.0
            
            timestamp_since = int((datetime.utcnow() - timedelta(hours=hours)).timestamp())
            
            total_outflow = 0.0
            for address in addresses:
                endpoint = f'{self.base_url}/on_chain/transfers_from_address'
                params = {
                    'a': contract,
                    'address': address,
                    's': timestamp_since,
                    'i': '24h'
                }
                
                response = self.session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if isinstance(data, list):
                    for item in data:
                        total_outflow += float(item.get('v', 0))
            
            divisor = 10 ** 8 if contract == 'bitcoin' else 10 ** 18
            return total_outflow / divisor
            
        except Exception as e:
            logger.error(f'Error getting outflow: {str(e)}')
            return 0.0
    
    def _get_exchange_reserve(
        self,
        contract: str,
        addresses: List[str]
    ) -> float:
        """
        Get total cryptocurrency reserve at exchange addresses.
        """
        try:
            if not self.api_key:
                return 0.0
            
            total_reserve = 0.0
            for address in addresses:
                endpoint = f'{self.base_url}/entities/balance'
                params = {
                    'a': contract,
                    'address': address
                }
                
                response = self.session.get(endpoint, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if isinstance(data, dict):
                    total_reserve += float(data.get('value', 0))
            
            divisor = 10 ** 8 if contract == 'bitcoin' else 10 ** 18
            return total_reserve / divisor
            
        except Exception as e:
            logger.error(f'Error getting reserve: {str(e)}')
            return 0.0
    
    def analyze_exchange_pressure(
        self,
        contract: str = 'bitcoin'
    ) -> Dict:
        """
        Analyze selling/buying pressure based on exchange flows.
        """
        
        flows = self.get_exchange_flows(contract, 24)
        
        if not flows:
            return self._get_mock_pressure_analysis()
        
        total_inflow = sum(f.get('inflow', 0) for f in flows)
        total_outflow = sum(f.get('outflow', 0) for f in flows)
        net_flow = total_inflow - total_outflow
        
        pressure_score = 0
        if total_inflow > total_outflow:
            pressure_score = (total_inflow / (total_inflow + total_outflow)) * 100
        else:
            pressure_score = -(total_outflow / (total_inflow + total_outflow)) * 100
        
        return {
            'total_inflow': total_inflow,
            'total_outflow': total_outflow,
            'net_flow': net_flow,
            'inflow_outflow_ratio': total_inflow / total_outflow if total_outflow > 0 else 0,
            'pressure_score': pressure_score,
            'interpretation': self._interpret_pressure(pressure_score),
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def _interpret_pressure(self, pressure_score: float) -> str:
        """
        Interpret exchange pressure score.
        """
        if pressure_score > 60:
            return 'Strong buying pressure'
        elif pressure_score > 30:
            return 'Mild buying pressure'
        elif pressure_score > -30:
            return 'Balanced'
        elif pressure_score > -60:
            return 'Mild selling pressure'
        else:
            return 'Strong selling pressure'
    
    def _get_mock_flows(self, contract: str) -> List[Dict]:
        """
        Return mock exchange flow data.
        """
        return [
            {
                'exchange_name': 'Binance',
                'exchange_key': 'binance',
                'inflow': 1250.5,
                'outflow': 980.75,
                'net_flow': 269.75,
                'total_reserve': 585420.3,
                'flow_type': 'inflow',
                'timestamp': datetime.utcnow().isoformat()
            },
            {
                'exchange_name': 'Coinbase',
                'exchange_key': 'coinbase',
                'inflow': 450.25,
                'outflow': 320.1,
                'net_flow': 130.15,
                'total_reserve': 285610.2,
                'flow_type': 'inflow',
                'timestamp': datetime.utcnow().isoformat()
            },
            {
                'exchange_name': 'Kraken',
                'exchange_key': 'kraken',
                'inflow': 320.75,
                'outflow': 450.5,
                'net_flow': -129.75,
                'total_reserve': 156230.1,
                'flow_type': 'outflow',
                'timestamp': datetime.utcnow().isoformat()
            },
            {
                'exchange_name': 'Huobi',
                'exchange_key': 'huobi',
                'inflow': 180.3,
                'outflow': 220.2,
                'net_flow': -39.9,
                'total_reserve': 98420.5,
                'flow_type': 'outflow',
                'timestamp': datetime.utcnow().isoformat()
            }
        ]
    
    def _get_mock_pressure_analysis(self) -> Dict:
        """
        Return mock pressure analysis.
        """
        return {
            'total_inflow': 2201.8,
            'total_outflow': 1971.55,
            'net_flow': 230.25,
            'inflow_outflow_ratio': 1.117,
            'pressure_score': 5.5,
            'interpretation': 'Balanced',
            'timestamp': datetime.utcnow().isoformat()
        }
