import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
import requests
from textblob import TextBlob


class MarketSentimentAnalyzer:
    """Analyze market sentiment from multiple sources."""

    def __init__(self, symbol: str = 'BTC'):
        self.symbol = symbol
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.sentiment_cache = {}

    def get_fear_greed_index(self) -> Dict:
        """Fetch Fear and Greed Index from API.
        
        Range: 0 (Extreme Fear) to 100 (Extreme Greed)
        """
        try:
            response = requests.get(self.fear_greed_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['data']:
                    latest = data['data'][0]
                    fgi_value = int(latest['value'])
                    fgi_classification = latest['value_classification']
                    
                    return {
                        'status': 'success',
                        'index_value': fgi_value,
                        'classification': fgi_classification,
                        'timestamp': latest['timestamp'],
                        'interpretation': self._interpret_fgi(fgi_value)
                    }
                else:
                    return {'status': 'error', 'message': 'No data available'}
            else:
                return {'status': 'error', 'message': f'API error: {response.status_code}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _interpret_fgi(self, value: int) -> str:
        """Interpret Fear and Greed Index value."""
        if value < 25:
            return 'Extreme Fear - Strong selling pressure, potential buy opportunity'
        elif value < 45:
            return 'Fear - Bearish sentiment, cautious approach recommended'
        elif value < 55:
            return 'Neutral - Market balanced between buyers and sellers'
        elif value < 75:
            return 'Greed - Bullish sentiment, potential overbought conditions'
        else:
            return 'Extreme Greed - Strong buying pressure, potential sell opportunity'

    def get_social_sentiment(self) -> Dict:
        """Analyze social media sentiment for the symbol."""
        try:
            url = "https://api.lunarcrush.com/v2/coins"
            params = {
                'symbol': self.symbol,
                'key': 'your_lunar_crush_key'  # Need to be set
            }
            
            return {
                'status': 'pending',
                'message': 'Lunar Crush API key required for social sentiment'
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def get_funding_rates(self, exchange: str = 'binance') -> Dict:
        """Get current funding rates for perpetual futures.
        
        High positive rates indicate bullish sentiment (potential reversal signal)
        High negative rates indicate bearish sentiment
        """
        try:
            if exchange.lower() == 'binance':
                url = "https://fapi.binance.com/fapi/v1/fundingRate"
                params = {'symbol': self.symbol + 'USDT', 'limit': 10}
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    rates = [float(item['fundingRate']) for item in data]
                    avg_rate = np.mean(rates)
                    
                    return {
                        'status': 'success',
                        'exchange': exchange,
                        'current_funding_rate': rates[0] if rates else 0,
                        'avg_funding_rate': avg_rate,
                        'rates_history': rates,
                        'sentiment': self._interpret_funding_rate(avg_rate),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {'status': 'error', 'message': f'API error: {response.status_code}'}
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def _interpret_funding_rate(self, rate: float) -> str:
        """Interpret funding rate sentiment."""
        if rate > 0.001:
            return 'Strong Bullish (Longs paying premiums)'
        elif rate > 0.0001:
            return 'Mildly Bullish'
        elif rate < -0.001:
            return 'Strong Bearish (Shorts paying premiums)'
        elif rate < -0.0001:
            return 'Mildly Bearish'
        else:
            return 'Neutral'

    def get_liquidation_data(self, exchange: str = 'binance') -> Dict:
        """Get liquidation data indicating market extremes."""
        try:
            if exchange.lower() == 'binance':
                url = "https://api.coinglass.com/api/v4/liquidation_history"
                params = {'symbol': self.symbol + 'USDT'}
                
                return {
                    'status': 'pending',
                    'message': 'Coinglass API key required for liquidation data'
                }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

    def calculate_composite_sentiment(self, fgi_value: int, funding_rate: float,
                                      on_chain_signal: float) -> Dict:
        """Calculate composite market sentiment from multiple sources.
        
        Combines FGI, funding rates, and on-chain data for holistic view
        """
        sentiment_score = 0
        components = {}
        
        fgi_normalized = (fgi_value - 50) / 50
        sentiment_score += fgi_normalized * 0.4
        components['fgi_contribution'] = fgi_normalized * 0.4
        
        funding_normalized = np.clip(funding_rate * 1000, -1, 1)
        sentiment_score += funding_normalized * 0.35
        components['funding_contribution'] = funding_normalized * 0.35
        
        sentiment_score += on_chain_signal * 0.25
        components['on_chain_contribution'] = on_chain_signal * 0.25
        
        composite_score = np.clip(sentiment_score, -1, 1)
        
        return {
            'composite_sentiment': composite_score,
            'sentiment_direction': 'BULLISH' if composite_score > 0.2 else 'BEARISH' if composite_score < -0.2 else 'NEUTRAL',
            'confidence': abs(composite_score),
            'components': components,
            'interpretation': self._interpret_composite_sentiment(composite_score),
            'timestamp': datetime.now().isoformat()
        }

    def _interpret_composite_sentiment(self, score: float) -> str:
        """Interpret composite sentiment score."""
        if score > 0.6:
            return 'Very Bullish - Strong buy setup conditions'
        elif score > 0.2:
            return 'Bullish - Favorable conditions for long positions'
        elif score > -0.2:
            return 'Neutral - Mixed signals, await confirmation'
        elif score > -0.6:
            return 'Bearish - Favorable conditions for short positions'
        else:
            return 'Very Bearish - Strong sell setup conditions'

    def get_all_sentiment_data(self) -> Dict:
        """Fetch all sentiment data."""
        result = {
            'symbol': self.symbol,
            'timestamp': datetime.now().isoformat(),
            'data': {}
        }
        
        result['data']['fear_greed_index'] = self.get_fear_greed_index()
        result['data']['funding_rates'] = self.get_funding_rates()
        result['data']['social_sentiment'] = self.get_social_sentiment()
        
        return result
