import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    CANCELLED = "CANCELLED"
    CLOSED = "CLOSED"


@dataclass
class Trade:
    trade_id: str
    symbol: str
    order_type: OrderType
    entry_price: float
    entry_quantity: float
    entry_time: str
    exit_price: Optional[float] = None
    exit_quantity: Optional[float] = None
    exit_time: Optional[str] = None
    status: OrderStatus = OrderStatus.PENDING
    confidence: float = 0.0
    profit_loss: float = 0.0
    profit_loss_percent: float = 0.0
    commission: float = 0.0

    def to_dict(self):
        data = asdict(self)
        data['order_type'] = self.order_type.value
        data['status'] = self.status.value
        return data


class BitgetFeeCalculator:
    
    MAKER_FEE = 0.0002
    TAKER_FEE = 0.0005
    
    @classmethod
    def calculate_entry_fee(cls, quantity: float, price: float) -> float:
        notional = quantity * price
        return notional * cls.TAKER_FEE
    
    @classmethod
    def calculate_exit_fee(cls, quantity: float, price: float) -> float:
        notional = quantity * price
        return notional * cls.TAKER_FEE
    
    @classmethod
    def calculate_total_fee(cls, entry_quantity: float, entry_price: float,
                           exit_quantity: float, exit_price: float) -> float:
        entry_fee = cls.calculate_entry_fee(entry_quantity, entry_price)
        exit_fee = cls.calculate_exit_fee(exit_quantity, exit_price)
        return entry_fee + exit_fee


class AutoTrader:
    
    def __init__(self, initial_balance: float = 1000.0, 
                 position_size_percent: float = 0.1,
                 confidence_threshold: float = 0.5):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.position_size_percent = position_size_percent
        self.confidence_threshold = confidence_threshold
        self.trades: List[Trade] = []
        self.open_positions: Dict[str, Trade] = {}
        self.trade_counter = 0
        
    def set_parameters(self, initial_balance: Optional[float] = None,
                      position_size_percent: Optional[float] = None,
                      confidence_threshold: Optional[float] = None):
        if initial_balance is not None:
            self.initial_balance = initial_balance
            self.current_balance = initial_balance
        if position_size_percent is not None:
            self.position_size_percent = position_size_percent
        if confidence_threshold is not None:
            self.confidence_threshold = confidence_threshold
    
    def calculate_position_size(self, price: float) -> float:
        available_balance = self.current_balance
        capital_to_use = available_balance * self.position_size_percent
        quantity = capital_to_use / price
        return quantity
    
    def should_trade(self, confidence: float) -> bool:
        return confidence >= self.confidence_threshold
    
    def execute_buy_signal(self, signal_data: Dict) -> Optional[Trade]:
        confidence = signal_data.get('confidence', 0.0)
        
        if not self.should_trade(confidence):
            logger.info(f"Confidence {confidence:.2%} below threshold {self.confidence_threshold:.2%}, skipping")
            return None
        
        symbol = signal_data.get('symbol', 'UNKNOWN')
        price = signal_data.get('price', 0.0)
        timestamp = signal_data.get('timestamp', datetime.now().isoformat())
        
        if price <= 0:
            logger.error("Invalid price")
            return None
        
        quantity = self.calculate_position_size(price)
        if quantity <= 0:
            logger.error("Invalid position size")
            return None
        
        entry_fee = BitgetFeeCalculator.calculate_entry_fee(quantity, price)
        
        self.trade_counter += 1
        trade = Trade(
            trade_id=f"TRADE_{self.trade_counter}",
            symbol=symbol,
            order_type=OrderType.BUY,
            entry_price=price,
            entry_quantity=quantity,
            entry_time=timestamp,
            status=OrderStatus.EXECUTED,
            confidence=confidence,
            commission=entry_fee
        )
        
        self.current_balance -= (quantity * price + entry_fee)
        self.open_positions[symbol] = trade
        self.trades.append(trade)
        
        logger.info(f"BUY signal executed: {symbol} qty={quantity:.4f} price=${price:.2f} fee=${entry_fee:.4f}")
        return trade
    
    def execute_sell_signal(self, signal_data: Dict) -> Optional[Trade]:
        confidence = signal_data.get('confidence', 0.0)
        
        if not self.should_trade(confidence):
            logger.info(f"Confidence {confidence:.2%} below threshold {self.confidence_threshold:.2%}, skipping")
            return None
        
        symbol = signal_data.get('symbol', 'UNKNOWN')
        price = signal_data.get('price', 0.0)
        timestamp = signal_data.get('timestamp', datetime.now().isoformat())
        
        if symbol not in self.open_positions:
            logger.warning(f"No open position for {symbol}")
            return None
        
        open_trade = self.open_positions[symbol]
        exit_fee = BitgetFeeCalculator.calculate_exit_fee(
            open_trade.entry_quantity, price
        )
        
        gross_proceeds = open_trade.entry_quantity * price
        total_fee = open_trade.commission + exit_fee
        net_proceeds = gross_proceeds - exit_fee
        
        profit_loss = net_proceeds - (open_trade.entry_quantity * open_trade.entry_price + open_trade.commission)
        profit_loss_percent = profit_loss / (open_trade.entry_quantity * open_trade.entry_price + open_trade.commission)
        
        open_trade.exit_price = price
        open_trade.exit_quantity = open_trade.entry_quantity
        open_trade.exit_time = timestamp
        open_trade.status = OrderStatus.CLOSED
        open_trade.profit_loss = profit_loss
        open_trade.profit_loss_percent = profit_loss_percent
        open_trade.commission += exit_fee
        
        self.current_balance += net_proceeds
        del self.open_positions[symbol]
        
        logger.info(f"SELL signal executed: {symbol} price=${price:.2f} pnl=${profit_loss:.4f} ({profit_loss_percent:.2%}) fee=${total_fee:.4f}")
        return open_trade
    
    def process_signal(self, signal_data: Dict) -> Optional[Trade]:
        signal_type = signal_data.get('signal_type', '').upper()
        
        if signal_type == 'BUY':
            return self.execute_buy_signal(signal_data)
        elif signal_type == 'SELL':
            return self.execute_sell_signal(signal_data)
        else:
            logger.warning(f"Unknown signal type: {signal_type}")
            return None
    
    def get_account_summary(self) -> Dict:
        closed_trades = [t for t in self.trades if t.status == OrderStatus.CLOSED]
        open_trades = list(self.open_positions.values())
        
        total_profit_loss = sum(t.profit_loss for t in closed_trades)
        winning_trades = len([t for t in closed_trades if t.profit_loss > 0])
        losing_trades = len([t for t in closed_trades if t.profit_loss < 0])
        win_rate = winning_trades / len(closed_trades) if closed_trades else 0
        
        return {
            'total_balance': self.current_balance,
            'initial_balance': self.initial_balance,
            'available_balance': self.current_balance,
            'total_trades': len(self.trades),
            'closed_trades': len(closed_trades),
            'open_trades': len(open_trades),
            'total_profit_loss': total_profit_loss,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'open_positions': [t.to_dict() for t in open_trades]
        }
    
    def export_trades(self) -> List[Dict]:
        return [t.to_dict() for t in self.trades]
    
    def reset(self, initial_balance: Optional[float] = None):
        if initial_balance is not None:
            self.initial_balance = initial_balance
        self.current_balance = self.initial_balance
        self.trades.clear()
        self.open_positions.clear()
        self.trade_counter = 0
        logger.info("Auto trader reset")
