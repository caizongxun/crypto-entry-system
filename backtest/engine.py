import pandas as pd
import numpy as np
from typing import Dict, List
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class Trade:
    """Represents a single trade."""
    
    def __init__(self, entry_time, entry_price, direction, size, confidence):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.confidence = confidence
        self.exit_time = None
        self.exit_price = None
        self.pnl = None
        self.pnl_percent = None
    
    def exit(self, exit_time, exit_price):
        self.exit_time = exit_time
        self.exit_price = exit_price
        
        if self.direction == 'UP':
            self.pnl = (exit_price - self.entry_price) * self.size
            self.pnl_percent = (exit_price - self.entry_price) / self.entry_price
        else:
            self.pnl = (self.entry_price - exit_price) * self.size
            self.pnl_percent = (self.entry_price - exit_price) / self.entry_price

class BacktestEngine:
    """Backtesting engine for evaluating trading strategy."""
    
    def __init__(self, initial_capital: float = 10000, position_size_percent: float = 1.0):
        self.initial_capital = initial_capital
        self.position_size_percent = position_size_percent
        self.balance = initial_capital
        self.trades = []
        self.equity_curve = []
        self.results = {}
    
    def execute_signal(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
        min_confidence: float = 0.60,
        take_profit_percent: float = 1.0,
        stop_loss_percent: float = 0.5
    ) -> Dict:
        """Execute backtest based on predictions.
        
        Args:
            df: Raw OHLCV DataFrame
            predictions: DataFrame with predictions
            min_confidence: Minimum confidence threshold
            take_profit_percent: Take profit percentage
            stop_loss_percent: Stop loss percentage
        
        Returns:
            Backtest results dictionary
        """
        logger.info("Starting backtest...")
        
        self.trades = []
        self.balance = self.initial_capital
        self.equity_curve = [self.balance]
        
        active_position = None
        
        # Filter signals by confidence
        signals = predictions[predictions['confidence'] >= min_confidence].copy()
        
        for idx, signal in signals.iterrows():
            if idx >= len(df) - 1:
                break
            
            current_candle = df.iloc[len(df) - len(predictions) + idx]
            next_candle = df.iloc[len(df) - len(predictions) + idx + 1]
            
            entry_price = next_candle['open']
            exit_price = next_candle['close']
            
            # Check if we should exit active position
            if active_position:
                tp_price = active_position.entry_price * (1 + take_profit_percent / 100)
                sl_price = active_position.entry_price * (1 - stop_loss_percent / 100)
                
                if next_candle['high'] >= tp_price or next_candle['low'] <= sl_price:
                    if next_candle['high'] >= tp_price:
                        exit_price = tp_price
                    else:
                        exit_price = sl_price
                    
                    active_position.exit(next_candle['close_time'], exit_price)
                    self.balance += active_position.pnl
                    self.trades.append(active_position)
                    logger.info(f"Position exited: PnL {active_position.pnl_percent:.2%}")
                    active_position = None
            
            # Enter new position if no active position
            if not active_position:
                position_size = (self.balance * self.position_size_percent / 100) / entry_price
                
                trade = Trade(
                    entry_time=next_candle['open_time'],
                    entry_price=entry_price,
                    direction=signal['direction'],
                    size=position_size,
                    confidence=signal['confidence']
                )
                
                active_position = trade
                logger.info(f"Position opened: {signal['direction']} at {entry_price}")
            
            self.equity_curve.append(self.balance)
        
        # Close active position at end
        if active_position:
            last_close = df['close'].iloc[-1]
            active_position.exit(df['close_time'].iloc[-1], last_close)
            self.balance += active_position.pnl
            self.trades.append(active_position)
        
        self.results = self._calculate_metrics()
        return self.results
    
    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics."""
        if len(self.trades) == 0:
            return {'error': 'No trades executed'}
        
        pnls = [t.pnl for t in self.trades if t.pnl is not None]
        pnl_percents = [t.pnl_percent for t in self.trades if t.pnl_percent is not None]
        
        winning_trades = [p for p in pnl_percents if p > 0]
        losing_trades = [p for p in pnl_percents if p <= 0]
        
        total_return = (self.balance - self.initial_capital) / self.initial_capital
        
        equity_array = np.array(self.equity_curve)
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        
        max_equity = np.max(equity_array)
        min_equity = np.min(equity_array)
        max_drawdown = (max_equity - min_equity) / max_equity if max_equity > 0 else 0
        
        metrics = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if len(self.trades) > 0 else 0,
            'total_return': total_return,
            'final_balance': self.balance,
            'total_pnl': self.balance - self.initial_capital,
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'max_win': np.max(pnl_percents) if pnl_percents else 0,
            'max_loss': np.min(pnl_percents) if pnl_percents else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
        }
        
        logger.info(f"Backtest Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                if key.endswith('_rate') or key.endswith('_return') or key.endswith('_drawdown'):
                    logger.info(f"  {key}: {value:.2%}")
                else:
                    logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        return metrics
    
    def get_trades_df(self) -> pd.DataFrame:
        """Return trades as DataFrame."""
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'entry_price': trade.entry_price,
                'exit_time': trade.exit_time,
                'exit_price': trade.exit_price,
                'direction': trade.direction,
                'size': trade.size,
                'confidence': trade.confidence,
                'pnl': trade.pnl,
                'pnl_percent': trade.pnl_percent,
            })
        
        return pd.DataFrame(trades_data)
    
    def export_results(self, path: str):
        """Export backtest results to CSV."""
        trades_df = self.get_trades_df()
        trades_df.to_csv(path, index=False)
        logger.info(f"Backtest results exported to {path}")
