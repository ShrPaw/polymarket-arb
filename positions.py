#!/usr/bin/env python3
"""
Position Tracker & Risk Manager
Tracks open positions, P&L, and enforces risk limits.
"""

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class Position:
    token_id: str
    market_question: str
    side: str        # "BUY" (long) or "SELL" (short)
    avg_price: float
    size: float
    filled_size: float = 0.0
    cost_basis: float = 0.0
    opened_at: str = ""
    status: str = "open"  # open, filled, partial, closed, resolved
    arb_group: str = ""   # groups legs of same arb trade


@dataclass
class RiskLimits:
    max_position_usd: float = 50.0        # max $ per single position
    max_total_exposure_usd: float = 200.0  # max total $ across all positions
    max_positions_open: int = 20           # max simultaneous positions
    min_edge_pct: float = 1.0              # minimum edge % to enter trade
    min_liquidity_usd: float = 10.0        # min book depth at fillable price
    min_volume_usd: float = 5_000.0        # min 24h volume
    max_daily_loss_usd: float = 50.0       # stop trading after this daily loss
    cooldown_seconds: float = 60.0         # min time between re-entering same market


class PositionTracker:
    """Persistent position tracking with risk checks."""

    def __init__(self, state_file: str = "positions.json", limits: Optional[RiskLimits] = None):
        self.state_file = Path(state_file)
        self.limits = limits or RiskLimits()
        self.positions: list[Position] = []
        self.trade_log: list[dict] = []
        self.daily_pnl: float = 0.0
        self._last_trade_time: dict[str, float] = {}  # market_key → timestamp
        self._load()

    def _load(self):
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text())
                self.positions = [Position(**p) for p in data.get("positions", [])]
                self.trade_log = data.get("trade_log", [])
                self.daily_pnl = data.get("daily_pnl", 0.0)
            except Exception:
                pass

    def save(self):
        data = {
            "positions": [vars(p) for p in self.positions],
            "trade_log": self.trade_log[-500:],  # keep last 500
            "daily_pnl": self.daily_pnl,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        self.state_file.write_text(json.dumps(data, indent=2))

    @property
    def open_positions(self) -> list[Position]:
        return [p for p in self.positions if p.status in ("open", "partial")]

    @property
    def total_exposure(self) -> float:
        return sum(p.cost_basis for p in self.open_positions)

    @property
    def position_count(self) -> int:
        return len(self.open_positions)

    def can_open(self, market_key: str, cost: float) -> tuple[bool, str]:
        """Check if we can open a new position."""
        if self.daily_pnl <= -self.limits.max_daily_loss_usd:
            return False, f"Daily loss limit hit: ${self.daily_pnl:.2f}"

        if self.position_count >= self.limits.max_positions_open:
            return False, f"Max positions reached: {self.position_count}"

        if cost > self.limits.max_position_usd:
            return False, f"Position too large: ${cost:.2f} > ${self.limits.max_position_usd}"

        if self.total_exposure + cost > self.limits.max_total_exposure_usd:
            return False, f"Total exposure limit: ${self.total_exposure + cost:.2f} > ${self.limits.max_total_exposure_usd}"

        last = self._last_trade_time.get(market_key, 0)
        if time.time() - last < self.limits.cooldown_seconds:
            return False, f"Cooldown active: {self.limits.cooldown_seconds - (time.time() - last):.0f}s remaining"

        return True, "OK"

    def record_fill(self, pos: Position):
        """Record a filled or partially filled position."""
        self.positions.append(pos)
        self._last_trade_time[pos.market_question] = time.time()

        self.trade_log.append({
            "time": datetime.now(timezone.utc).isoformat(),
            "token_id": pos.token_id,
            "question": pos.market_question[:80],
            "side": pos.side,
            "price": pos.avg_price,
            "size": pos.filled_size,
            "cost": pos.cost_basis,
            "arb_group": pos.arb_group,
        })
        self.save()

    def record_resolution(self, token_id: str, payout: float):
        """Record a market resolution payout."""
        for p in self.positions:
            if p.token_id == token_id and p.status in ("open", "filled"):
                pnl = payout - p.cost_basis
                self.daily_pnl += pnl
                p.status = "resolved"
                self.trade_log.append({
                    "time": datetime.now(timezone.utc).isoformat(),
                    "type": "resolution",
                    "token_id": token_id,
                    "payout": payout,
                    "pnl": pnl,
                })
        self.save()

    def reset_daily(self):
        """Call at start of each day."""
        self.daily_pnl = 0.0
        self.save()

    def summary(self) -> dict:
        return {
            "open_positions": self.position_count,
            "total_exposure": f"${self.total_exposure:.2f}",
            "daily_pnl": f"${self.daily_pnl:.2f}",
            "total_trades": len(self.trade_log),
            "limits": {
                "max_position": f"${self.limits.max_position_usd}",
                "max_exposure": f"${self.limits.max_total_exposure_usd}",
                "max_positions": self.limits.max_positions_open,
                "min_edge": f"{self.limits.min_edge_pct}%",
            },
        }
