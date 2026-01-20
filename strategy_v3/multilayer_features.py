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
        
        lookback_4 = 4
        lookback_16 = 16
        lookback_10 = 10
        
        close_shift_4 = np.concatenate([[close[0]] * lookback_4, close[:-lookback_4]])
        close_shift_16 = np.concatenate([[close[0]] * lookback_16, close[:-lookback_16]])
        close_shift_10 = np.concatenate([[close[0]] * lookback_10, close[:-lookback_10]])
        
        features['momentum_1h'] = close - close_shift_4
        
        with np.errstate(divide='ignore', invalid='ignore'):
            momentum_1h_pct = (features['momentum_1h'] / close_shift_4) * 100
            momentum_1h_pct = np.where(np.isfinite(momentum_1h_pct), momentum_1h_pct, 0)
        features['momentum_1h_pct'] = momentum_1h_pct
        
        features['momentum_4h'] = close - close_shift_16
        
        features['velocity'] = np.gradient(close, edge_order=2)
        features['acceleration'] = np.gradient(np.gradient(close, edge_order=2), edge_order=2)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            roc_10 = ((close - close_shift_10) / close_shift_10) * 100
            roc_10 = np.where(np.isfinite(roc_10), roc_10, 0)
        features['roc_10'] = roc_10
        
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
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
            vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 1.0)
        features['volume_ratio'] = vol_ratio
        
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
            if range_hl > 1e-10:
                clv_values[i] = ((close[i] - low[i]) - (high[i] - close[i])) / range_hl
            else:
                clv_values[i] = 0
        
        ad = clv_values * volume
        ad_smooth = pd.Series(ad).rolling(window=14).mean().values
        features['ad_indicator'] = np.where(np.isfinite(ad_smooth), ad_smooth, 0)
        
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
        
        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / np.where(avg_loss > 0, avg_loss, 1)
            rs = np.where(np.isfinite(rs), rs, 0)
        rsi = 100 - (100 / (1 + np.maximum(rs, 0)))
        features['rsi'] = np.where(np.isfinite(rsi), rsi, 50)
        
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
        with np.errstate(divide='ignore', invalid='ignore'):
            bb_pos = (close - bb_lower) / np.where(bb_range > 0, bb_range, 1)
            bb_pos = np.where(np.isfinite(bb_pos), bb_pos, 0.5)
        features['bb_position'] = bb_pos
        
        stoch_k = np.zeros(len(close))
        for i in range(14, len(close)):
            low_14 = low[max(0, i-14):i].min()
            high_14 = high[max(0, i-14):i].max()
            if high_14 > low_14:
                stoch_k[i] = 100 * (close[i] - low_14) / (high_14 - low_14)
            else:
                stoch_k[i] = 50
        
        features['stochastic_k'] = stoch_k
        features['stochastic_d'] = pd.Series(stoch_k).rolling(window=3).mean().values
        features['stochastic_d'] = np.where(np.isfinite(features['stochastic_d']), features['stochastic_d'], 50)
        
        return features
    
    def add_risk_features(self, df):
        features = pd.DataFrame(index=df.index)
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        open_price = df['open'].values
        
        self.logger.info('Computing risk features')
        
        returns = np.concatenate([[0], np.diff(close) / np.maximum(close[:-1], 1e-10)])
        returns = np.where(np.isfinite(returns), returns, 0)
        volatility_15m = pd.Series(returns).rolling(window=20).std().values
        features['volatility_15m'] = np.where(np.isfinite(volatility_15m), volatility_15m, 0)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            high_low_range = pd.Series(close).rolling(window=4).std().values / np.maximum(pd.Series(close).rolling(window=4).mean().values, 1e-10)
            high_low_range = np.where(np.isfinite(high_low_range), high_low_range, 0)
        features['volatility_1h'] = high_low_range
        
        avg_vol_1h = pd.Series(high_low_range).rolling(window=20).mean().values
        with np.errstate(divide='ignore', invalid='ignore'):
            vol_ratio = high_low_range / np.where(avg_vol_1h > 0, avg_vol_1h, 1)
            vol_ratio = np.where(np.isfinite(vol_ratio), vol_ratio, 1.0)
        features['volatility_ratio'] = vol_ratio
        
        gap = np.abs(open_price - np.concatenate([[close[0]], close[:-1]])) / np.maximum(np.concatenate([[close[0]], close[:-1]]), 1e-10) * 100
        features['gap_from_previous_close'] = np.where(np.isfinite(gap), gap, 0)
        
        range_hl = (high - low) / np.maximum(low, 1e-10) * 100
        features['range_hl_pct'] = np.where(np.isfinite(range_hl), range_hl, 0)
        
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
        with np.errstate(divide='ignore', invalid='ignore'):
            sess_strength = volume / np.where(avg_vol > 0, avg_vol, 1)
            sess_strength = np.where(np.isfinite(sess_strength), sess_strength, 1.0)
        features['session_strength'] = sess_strength
        
        returns = np.concatenate([[0], np.diff(close) / np.maximum(close[:-1], 1e-10)])
        returns = np.where(np.isfinite(returns), returns, 0)
        volatility = pd.Series(returns).rolling(window=20).std().values
        vol_75th = pd.Series(volatility).rolling(window=100).quantile(0.75).values
        features['volatility_regime'] = np.where(volatility > vol_75th, 1.0, 0.0)
        
        concentration = (pd.Series(close).rolling(window=20).std().values / 
                        np.maximum(pd.Series(close).rolling(window=20).mean().values, 1e-10))
        features['concentration'] = np.where(np.isfinite(concentration), concentration, 0)
        
        return features
    
    def engineer_multilayer_features(self, df):
        """
        Combine all multi-layer features into a single dataframe.
        
        Returns dataframe with 32 new features:
        - 7 momentum features
        - 5 volume features
        - 7 extremum features
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
