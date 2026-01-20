import numpy as np
import pandas as pd
from loguru import logger


class MultiLayerFeatureEngineer:
    """
    Multi-layer feature engineering for 15-minute timeframe.
    
    Layer 1: Pattern features (from pattern_detector)
    Layer 2: Momentum confirmation
    Layer 3: Volume confirmation
    Layer 4: Extremum signals (RSI, MACD, Bollinger)
    Layer 5: Risk filtering
    Layer 6: Environment context
    """
    
    def __init__(self, config=None):
        self.config = config
        self.logger = logger
    
    def add_momentum_features(self, df):
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        
        self.logger.info('Computing momentum features')
        
        lookback_4 = 4  # 1 hour = 4 x 15m
        lookback_16 = 16  # 4 hours = 16 x 15m
        lookback_10 = 10
        
        features['momentum_1h'] = close - np.concatenate([[close[0]] * lookback_4, close[:-lookback_4]])
        features['momentum_1h_pct'] = (features['momentum_1h'] / np.concatenate([[close[0]] * lookback_4, close[:-lookback_4]])) * 100
        
        features['momentum_4h'] = close - np.concatenate([[close[0]] * lookback_16, close[:-lookback_16]])
        
        features['velocity'] = np.gradient(close, edge_order=2)
        features['acceleration'] = np.gradient(np.gradient(close, edge_order=2), edge_order=2)
        
        features['roc_10'] = ((close - np.concatenate([[close[0]] * lookback_10, close[:-lookback_10]])) / 
                             np.concatenate([[close[0]] * lookback_10, close[:-lookback_10]])) * 100
        
        ema_20 = pd.Series(close).ewm(span=20).mean().values
        features['ema_slope'] = np.gradient(ema_20, edge_order=2)
        
        return features
    
    def add_volume_features(self, df):
        features = pd.DataFrame(index=df.index)
        volume = df['volume'].values
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        self.logger.info('Computing volume features')
        
        avg_volume = pd.Series(volume).rolling(window=20).mean().values
        features['volume_ratio'] = np.where(avg_volume > 0, volume / avg_volume, 1.0)
        
        features['volume_trend'] = volume - np.concatenate([[volume[0]], volume[:-1]])
        
        obv = np.zeros(len(close))
        obv[0] = volume[0]
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        features['obv'] = obv
        features['obv_slope'] = np.gradient(obv, edge_order=2)
        
        clv_values = np.zeros(len(close))
        for i in range(len(close)):
            range_hl = high[i] - low[i]
            if range_hl > 0:
                clv_values[i] = ((close[i] - low[i]) - (high[i] - close[i])) / range_hl
            else:
                clv_values[i] = 0
        
        ad = clv_values * volume
        features['ad_indicator'] = pd.Series(ad).rolling(window=14).mean().values
        
        return features
    
    def add_extremum_features(self, df):
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        
        self.logger.info('Computing extremum features')
        
        delta = np.concatenate([[0], np.diff(close)])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=14).mean().values
        avg_loss = pd.Series(loss).rolling(window=14).mean().values
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        features['rsi'] = 100 - (100 / (1 + rs))
        
        ema_12 = pd.Series(close).ewm(span=12).mean().values
        ema_26 = pd.Series(close).ewm(span=26).mean().values
        macd = ema_12 - ema_26
        signal = pd.Series(macd).ewm(span=9).mean().values
        
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = macd - signal
        
        sma_20 = pd.Series(close).rolling(window=20).mean().values
        std_20 = pd.Series(close).rolling(window=20).std().values
        
        bb_upper = sma_20 + 2 * std_20
        bb_lower = sma_20 - 2 * std_20
        
        features['bb_upper'] = bb_upper
        features['bb_lower'] = bb_lower
        
        bb_range = bb_upper - bb_lower
        features['bb_position'] = np.where(bb_range > 0, (close - bb_lower) / bb_range, 0.5)
        
        stoch_k = np.zeros(len(close))
        for i in range(14, len(close)):
            low_14 = low[i-14:i].min()
            high_14 = high[i-14:i].max()
            if high_14 > low_14:
                stoch_k[i] = 100 * (close[i] - low_14) / (high_14 - low_14)
            else:
                stoch_k[i] = 50
        
        features['stochastic_k'] = stoch_k
        features['stochastic_d'] = pd.Series(stoch_k).rolling(window=3).mean().values
        
        return features
    
    def add_risk_features(self, df):
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        
        self.logger.info('Computing risk features')
        
        returns = np.concatenate([[0], np.diff(close) / close[:-1]])
        volatility_15m = pd.Series(returns).rolling(window=20).std().values
        features['volatility_15m'] = volatility_15m
        
        high_low_range = pd.Series(close).rolling(window=4).std().values / pd.Series(close).rolling(window=4).mean().values
        features['volatility_1h'] = high_low_range
        
        avg_vol_1h = pd.Series(high_low_range).rolling(window=20).mean().values
        features['volatility_ratio'] = np.where(avg_vol_1h > 0, high_low_range / avg_vol_1h, 1.0)
        
        gap = np.abs(open_price - np.concatenate([[close[0]], close[:-1]])) / np.concatenate([[close[0]], close[:-1]]) * 100
        features['gap_from_previous_close'] = gap
        
        range_hl = (high - low) / low * 100
        features['range_hl_pct'] = range_hl
        
        return features
    
    def add_environment_features(self, df):
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        volume = df['volume'].values
        
        self.logger.info('Computing environment features')
        
        ma_4h_fast = pd.Series(close).rolling(window=16).mean().values
        ma_4h_slow = pd.Series(close).rolling(window=32).mean().values
        features['trend_4h'] = np.where(ma_4h_fast > ma_4h_slow, 1.0, 0.0)
        
        ma_1h_fast = pd.Series(close).rolling(window=4).mean().values
        ma_1h_slow = pd.Series(close).rolling(window=8).mean().values
        features['trend_1h'] = np.where(ma_1h_fast > ma_1h_slow, 1.0, 0.0)
        
        avg_vol = pd.Series(volume).rolling(window=20).mean().values
        features['session_strength'] = np.where(avg_vol > 0, volume / avg_vol, 1.0)
        
        volatility = pd.Series(np.concatenate([[0], np.diff(close) / close[:-1]])).rolling(window=20).std().values
        vol_75th = pd.Series(volatility).rolling(window=100).quantile(0.75).values
        features['volatility_regime'] = np.where(volatility > vol_75th, 1.0, 0.0)
        
        concentration = (pd.Series(close).rolling(window=20).std().values / 
                        pd.Series(close).rolling(window=20).mean().values)
        features['concentration'] = concentration
        
        return features
    
    def engineer_multilayer_features(self, df):
        """
        Combine all multi-layer features into a single dataframe.
        
        Returns dataframe with 32 new features:
        - 6 momentum features
        - 5 volume features
        - 5 extremum features
        - 5 risk features
        - 5 environment features
        """
        features = pd.DataFrame(index=df.index)
        
        momentum_feats = self.add_momentum_features(df)
        volume_feats = self.add_volume_features(df)
        extremum_feats = self.add_extremum_features(df)
        risk_feats = self.add_risk_features(df)
        env_feats = self.add_environment_features(df)
        
        for col in momentum_feats.columns:
            features[f'momentum_{col}'] = momentum_feats[col]
        
        for col in volume_feats.columns:
            features[f'volume_{col}'] = volume_feats[col]
        
        for col in extremum_feats.columns:
            features[f'extremum_{col}'] = extremum_feats[col]
        
        for col in risk_feats.columns:
            features[f'risk_{col}'] = risk_feats[col]
        
        for col in env_feats.columns:
            features[f'env_{col}'] = env_feats[col]
        
        self.logger.info(f'Generated {len(features.columns)} multi-layer features')
        return features
