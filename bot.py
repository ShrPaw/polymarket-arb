#!/usr/bin/env python3
"""
Polymarket Autonomous Arbitrage Bot v2

Combines the v2 engine with autonomous trading loop,
position management, and live execution.

Usage:
    python3 bot.py --mode scan --once
    python3 bot.py --mode paper --interval 60
    POLYMARKET_KEY=0x... python3 bot.py --mode live --interval 30 --min-edge 1.5

Ref: arxiv.org/abs/2508.03474
"""

import os
import sys
import json
import time
import signal
import argparse
import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from collections import defaultdict

from engine import ArbEngine, ArbOpportunity, ArbLeg
from positions import PositionTracker, Position, RiskLimits

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bot")


@dataclass
class BotConfig:
    mode: str = "scan"
    poll_interval: int = 60
    max_events: int = 200
    min_edge_pct: float = 0.5
    min_book_depth: float = 5.0
    output_dir: str = "runs"
    max_trades_per_cycle: int = 5


class LiveExecutor:
    """Places orders via Polymarket CLOB API."""

    def __init__(self, private_key: str):
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import OrderArgs, OrderType
        from py_clob_client.order_builder.constants import BUY, SELL

        self.client = ClobClient(
            "https://clob.polymarket.com",
            key=private_key, chain_id=137, signature_type=0,
        )
        try:
            self.client.set_api_creds(self.client.create_or_derive_api_creds())
        except Exception as e:
            log.warning(f"API cred setup: {e}")

        self.OrderArgs = OrderArgs
        self.OrderType = OrderType
        self.BUY = BUY
        self.SELL = SELL

    def place(self, token_id: str, price: float, size: float, side: str) -> dict:
        order_args = self.OrderArgs(
            token_id=token_id,
            price=round(price, 4),
            size=round(size, 2),
            side=self.BUY if side == "BUY" else self.SELL,
        )
        signed = self.client.create_order(order_args)
        resp = self.client.post_order(signed, self.OrderType.GTC)
        log.info(f"  ORDER: {side} {size:.2f} @ ${price:.4f} → {resp}")
        return resp


class ArbBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.engine = ArbEngine()
        self.tracker = PositionTracker(
            "positions.json",
            limits=RiskLimits(
                min_edge_pct=config.min_edge_pct,
                max_position_usd=100.0,
                max_total_exposure_usd=500.0,
            ),
        )
        self.executor = None
        self.running = False
        self.stats = {
            "scans": 0, "opportunities": 0,
            "trades_attempted": 0, "trades_filled": 0,
            "total_profit": 0.0,
        }

    def init_executor(self, key: str):
        self.executor = LiveExecutor(key)

    def execute(self, opp: ArbOpportunity) -> bool:
        arb_group = f"arb_{int(time.time())}"
        executed = 0

        for leg in opp.legs:
            if not leg.token_id or leg.fillable_size < 1.0:
                continue

            size = min(
                leg.fillable_size,
                self.tracker.limits.max_position_usd / leg.price if leg.price > 0.01 else 10,
            )
            if size < 1.0:
                continue

            cost = size * leg.price
            can, reason = self.tracker.can_open(leg.token_id[:20], cost)
            if not can:
                log.info(f"  SKIP: {reason}")
                continue

            if self.config.mode == "live" and self.executor:
                try:
                    self.executor.place(leg.token_id, leg.price, size, leg.side)
                except Exception as e:
                    log.error(f"  ORDER FAILED: {e}")
                    continue
            else:
                log.info(f"  PAPER: {leg.side} {size:.2f} @ ${leg.price:.4f} — {leg.market_question[:50]}")

            pos = Position(
                token_id=leg.token_id,
                market_question=leg.market_question[:200],
                side=leg.side,
                avg_price=leg.price,
                size=size,
                filled_size=size,
                cost_basis=cost,
                opened_at=datetime.now(timezone.utc).isoformat(),
                status="filled",
                arb_group=arb_group,
            )
            self.tracker.record_fill(pos)
            self.stats["trades_filled"] += 1
            executed += 1

        self.stats["trades_attempted"] += 1
        return executed > 0

    def run_loop(self):
        self.running = True
        log.info(f"Bot v2 starting | mode={self.config.mode} | interval={self.config.poll_interval}s")
        log.info(f"  Min edge: {self.config.min_edge_pct}% | Min depth: ${self.config.min_book_depth}")

        while self.running:
            try:
                result = self.engine.scan(
                    self.config.max_events,
                    self.config.min_edge_pct,
                    self.config.min_book_depth,
                )

                self.stats["scans"] += 1
                self.stats["opportunities"] += result["verified_opportunities"]

                log.info(f"Scan #{self.stats['scans']}: "
                         f"{result['markets_scanned']} markets | "
                         f"{result['verified_opportunities']} verified | "
                         f"{result['elapsed_seconds']}s")

                if result["verified_opportunities"] > 0 and self.config.mode in ("paper", "live"):
                    for opp_data in result["top_opportunities"][:self.config.max_trades_per_cycle]:
                        legs = [
                            ArbLeg(
                                market_question=l["market"],
                                token_id="",  # stored in full data
                                side=l["side"],
                                outcome=l["outcome"],
                                price=l["price"],
                                fillable_size=l["depth"],
                                slippage=l["slippage"],
                            )
                            for l in opp_data["legs"]
                        ]
                        opp = ArbOpportunity(
                            kind=opp_data["kind"],
                            legs=legs,
                            cost=opp_data["cost"],
                            payout=1.0,
                            profit=opp_data["profit"],
                            roi_pct=opp_data["roi_pct"],
                        )
                        self.execute(opp)

                # Save
                os.makedirs(self.config.output_dir, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                with open(f"{self.config.output_dir}/bot_{ts}.json", "w") as f:
                    json.dump(result, f, indent=2)

                self.tracker.save()
                time.sleep(self.config.poll_interval)

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                log.error(f"Error: {e}", exc_info=True)
                time.sleep(30)

        self.tracker.save()
        log.info(f"Bot stopped. Stats: {json.dumps(self.stats)}")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Arb Bot v2")
    parser.add_argument("--mode", choices=["scan", "paper", "live"], default="scan")
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--min-edge", type=float, default=0.5)
    parser.add_argument("--min-depth", type=float, default=5.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    config = BotConfig(
        mode=args.mode, poll_interval=args.interval,
        max_events=args.events, min_edge_pct=args.min_edge,
        min_book_depth=args.min_depth,
    )

    bot = ArbBot(config)

    if args.mode == "live":
        key = os.environ.get("POLYMARKET_KEY", "")
        if not key:
            log.error("POLYMARKET_KEY env var required for live mode")
            sys.exit(1)
        bot.init_executor(key)

    signal.signal(signal.SIGINT, lambda s, f: setattr(bot, 'running', False))
    signal.signal(signal.SIGTERM, lambda s, f: setattr(bot, 'running', False))

    if args.once:
        result = bot.engine.scan(config.max_events, config.min_edge_pct, config.min_book_depth)
        print(json.dumps(result, indent=2))
    else:
        bot.run_loop()


if __name__ == "__main__":
    main()
