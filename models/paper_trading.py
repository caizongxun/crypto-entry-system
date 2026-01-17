import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum


class OrderType(Enum):
    """Order types."""
    BUY = 'BUY'
    SELL = 'SELL'
    SHORT = 'SHORT'
    CLOSE_SHORT = 'CLOSE_SHORT'


class OrderStatus(Enum):
    """Order statuses."""
    PENDING = 'PENDING'
    FILLED = 'FILLED'
    CANCELLED = 'CANCELLED'
    PARTIALLY_FILLED = 'PARTIALLY_FILLED'


@dataclass
class Order:
    """Order record."""
    order_id: str
    timestamp: str
    order_type: str
    symbol: str
    quantity: float
    price: float
    status: str = OrderStatus.PENDING.value
    filled_quantity: float = 0.0
    fill_price: float = 0.0
    commission: float = 0.0
    pnl: float = 0.0
    notes: str = ''


@dataclass
class Position:
    """Active position record."""
    position_id: str
    symbol: str
    entry_price: float
    entry_time: str
    quantity: float
    position_type: str
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    pnl_percentage: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0


class PaperTradingEngine:
    """Simulate trading without real money."""

    def __init__(self, initial_balance: float = 10000.0, commission_rate: float = 0.001):
        self.initial_balance = initial_balance
        self.available_balance = initial_balance
        self.total_balance = initial_balance
        self.commission_rate = commission_rate
        
        self.orders: List[Order] = []
        self.positions: Dict[str, Position] = {}
        self.trade_history: List[Dict] = []
        self.balance_history: List[Dict] = []
        
        self.order_counter = 0
        self.position_counter = 0

    def open_position(self, symbol: str, order_type: str, quantity: float,
                      entry_price: float, stop_loss: float = 0.0,
                      take_profit: float = 0.0, notes: str = '') -> Dict:
        """Open a new position.
        
        Args:
            symbol: Trading pair (e.g., 'BTCUSDT')
            order_type: 'BUY' or 'SHORT'
            quantity: Position size
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            notes: Additional notes
            
        Returns:
            Result dict with position details
        """
        
        position_cost = quantity * entry_price * (1 + self.commission_rate)
        
        if position_cost > self.available_balance:
            return {
                'status': 'error',
                'message': f'Insufficient balance. Required: {position_cost:.2f}, Available: {self.available_balance:.2f}'
            }
        
        self.order_counter += 1
        self.position_counter += 1
        
        order_id = f'ORD_{self.order_counter:06d}'
        position_id = f'POS_{self.position_counter:06d}'
        timestamp = datetime.now().isoformat()
        
        commission = quantity * entry_price * self.commission_rate
        self.available_balance -= position_cost
        
        order = Order(
            order_id=order_id,
            timestamp=timestamp,
            order_type=order_type,
            symbol=symbol,
            quantity=quantity,
            price=entry_price,
            status=OrderStatus.FILLED.value,
            filled_quantity=quantity,
            fill_price=entry_price,
            commission=commission,
            notes=notes
        )
        
        self.orders.append(order)
        
        position = Position(
            position_id=position_id,
            symbol=symbol,
            entry_price=entry_price,
            entry_time=timestamp,
            quantity=quantity,
            position_type=order_type,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[position_id] = position
        
        return {
            'status': 'success',
            'position_id': position_id,
            'order_id': order_id,
            'message': f'Position opened: {order_type} {quantity} {symbol} at {entry_price:.2f}',
            'commission': commission,
            'remaining_balance': self.available_balance
        }

    def close_position(self, position_id: str, close_price: float, notes: str = '') -> Dict:
        """Close an existing position.
        
        Args:
            position_id: Position ID to close
            close_price: Exit price
            notes: Additional notes
            
        Returns:
            Result dict with P&L details
        """
        
        if position_id not in self.positions:
            return {'status': 'error', 'message': f'Position {position_id} not found'}
        
        position = self.positions[position_id]
        
        self.order_counter += 1
        order_id = f'ORD_{self.order_counter:06d}'
        timestamp = datetime.now().isoformat()
        
        close_value = position.quantity * close_price
        commission = close_value * self.commission_rate
        
        if position.position_type == 'BUY':
            pnl = close_value - (position.quantity * position.entry_price) - commission
        else:
            pnl = (position.quantity * position.entry_price) - close_value - commission
        
        pnl_percentage = (pnl / (position.quantity * position.entry_price)) * 100
        
        self.available_balance += close_value - commission
        self.total_balance = self.available_balance
        
        for pos in self.positions.values():
            if pos.position_type == 'BUY':
                pos.unrealized_pnl = (pos.current_price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - pos.current_price) * pos.quantity
            pos.pnl_percentage = (pos.unrealized_pnl / (pos.entry_price * pos.quantity)) * 100
        
        self.total_balance = self.available_balance + sum(p.unrealized_pnl for p in self.positions.values())
        
        close_order = Order(
            order_id=order_id,
            timestamp=timestamp,
            order_type='SELL' if position.position_type == 'BUY' else 'CLOSE_SHORT',
            symbol=position.symbol,
            quantity=position.quantity,
            price=close_price,
            status=OrderStatus.FILLED.value,
            filled_quantity=position.quantity,
            fill_price=close_price,
            commission=commission,
            pnl=pnl,
            notes=notes
        )
        
        self.orders.append(close_order)
        
        trade_record = {
            'position_id': position_id,
            'open_order_id': next(o.order_id for o in self.orders if o.symbol == position.symbol and o.status == OrderStatus.FILLED.value),
            'close_order_id': order_id,
            'symbol': position.symbol,
            'entry_price': position.entry_price,
            'close_price': close_price,
            'quantity': position.quantity,
            'entry_time': position.entry_time,
            'close_time': timestamp,
            'position_type': position.position_type,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'commission': commission
        }
        
        self.trade_history.append(trade_record)
        
        del self.positions[position_id]
        
        return {
            'status': 'success',
            'position_id': position_id,
            'order_id': order_id,
            'pnl': pnl,
            'pnl_percentage': pnl_percentage,
            'message': f'Position closed with P&L: {pnl:.2f} ({pnl_percentage:.2f}%)',
            'remaining_balance': self.available_balance,
            'total_balance': self.total_balance
        }

    def update_market_price(self, symbol: str, current_price: float) -> None:
        """Update current market prices for all positions."""
        for position in self.positions.values():
            if position.symbol == symbol:
                position.current_price = current_price
                
                if position.position_type == 'BUY':
                    position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                else:
                    position.unrealized_pnl = (position.entry_price - current_price) * position.quantity
                
                position.pnl_percentage = (position.unrealized_pnl / (position.entry_price * position.quantity)) * 100
        
        self.total_balance = self.available_balance + sum(p.unrealized_pnl for p in self.positions.values())

    def get_account_summary(self) -> Dict:
        """Get account summary."""
        total_pnl = sum(t['pnl'] for t in self.trade_history)
        total_commission = sum(o.commission for o in self.orders if o.status == OrderStatus.FILLED.value)
        
        return {
            'initial_balance': self.initial_balance,
            'available_balance': self.available_balance,
            'total_balance': self.total_balance,
            'balance_change': self.total_balance - self.initial_balance,
            'balance_change_pct': ((self.total_balance - self.initial_balance) / self.initial_balance) * 100,
            'total_trades': len(self.trade_history),
            'winning_trades': len([t for t in self.trade_history if t['pnl'] > 0]),
            'losing_trades': len([t for t in self.trade_history if t['pnl'] < 0]),
            'total_pnl': total_pnl,
            'total_commission': total_commission,
            'active_positions': len(self.positions),
            'unrealized_pnl': sum(p.unrealized_pnl for p in self.positions.values()),
            'win_rate': (len([t for t in self.trade_history if t['pnl'] > 0]) / len(self.trade_history) * 100) if self.trade_history else 0
        }

    def get_positions(self) -> List[Dict]:
        """Get all active positions."""
        return [asdict(p) for p in self.positions.values()]

    def get_trade_history(self, limit: int = 100) -> List[Dict]:
        """Get trade history."""
        return self.trade_history[-limit:]

    def get_order_history(self, limit: int = 100) -> List[Dict]:
        """Get order history."""
        orders_dict = [asdict(o) for o in self.orders]
        return orders_dict[-limit:]
