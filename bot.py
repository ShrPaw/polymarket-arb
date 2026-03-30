#!/usr/bin/env python3
"""
Polymarket Autonomous Arbitrage Bot

Continuously monitors markets, detects arbitrage edges,
and executes trades with position management.

Usage:
    # Discovery mode (no trading, just scan and report)
    python3 bot.py --mode scan

    # Paper trading (simulated fills)
    python3 bot.py --mode paper

    # Live trading (requires private key)
    POLYMARKET_KEY=0x... python3 bot.py --mode live

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
from dataclasses import dataclass, field
from collections import defaultdict

from market_relationships import (
    Market, fetch_events,
    detect_within_event, detect_negrisk_groups,
    detect_temporal_subsets, detect_entity_similarity,
    scan_single_condition, scan_combinatorial,
)
from orderbook import BookFetcher, OrderBook
from positions import PositionTracker, Position, RiskLimits

# ── Logging ─────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("arb-bot")

# ── Config ──────────────────────────────────────────────────────────────────

@dataclass
class BotConfig:
    mode: str = "scan"           # scan | paper | live
    poll_interval: int = 60      # seconds between scans
    max_events: int = 200
    min_volume: int = 5_000
    min_edge_pct: float = 1.0    # minimum % edge to act
    min_book_depth: float = 10.0 # min $ depth at fillable price
    output_dir: str = "runs"
    log_all_opps: bool = True    # log all opportunities, not just tradeable


# ── Live Executor ───────────────────────────────────────────────────────────

class LiveExecutor:
    """Places orders via Polymarket CLOB API."""

    def __init__(self, private_key: str):
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL
        except ImportError:
            log.error("py-clob-client not installed. Run: pip install py-clob-client")
            sys.exit(1)

        self.host = "https://clob.polymarket.com"
        self.chain_id = 137
        self.client = ClobClient(
            self.host, key=private_key, chain_id=self.chain_id, signature_type=0
        )
        try:
            self.client.set_api_creds(self.client.create_or_derive_api_creds())
        except Exception as e:
            log.warning(f"API cred setup: {e}")

        self.OrderArgs = OrderArgs
        self.OrderType = OrderType
        self.BUY = BUY
        self.SELL = SELL

    def place_limit(self, token_id: str, price: float, size: float, side: str) -> dict:
        """Place a limit order."""
        order_args = self.OrderArgs(
            token_id=token_id,
            price=round(price, 2),
            size=round(size, 2),
            side=self.BUY if side == "BUY" else self.SELL,
        )
        signed = self.client.create_order(order_args)
        resp = self.client.post_order(signed, self.OrderType.GTC)
        log.info(f"  ORDER PLACED: {side} {size} @ ${price:.2f} → {resp}")
        return resp


# ── Opportunity Model ───────────────────────────────────────────────────────

@dataclass
class EvaluatedOpportunity:
    """An opportunity with orderbook verification."""
    kind: str           # single_condition | combinatorial
    markets: list[str]
    legs: list[dict]    # {token_id, side, price, book_depth, ...}
    cost: float
    payout: float
    profit: float
    roi_pct: float
    fillable: bool = True
    skip_reason: str = ""


# ── Bot Core ────────────────────────────────────────────────────────────────

class ArbBot:
    def __init__(self, config: BotConfig):
        self.config = config
        self.book_fetcher = BookFetcher()
        self.tracker = PositionTracker(
            "positions.json",
            limits=RiskLimits(min_edge_pct=config.min_edge_pct)
        )
        self.executor = None
        self.running = False
        self.stats = {
            "scans": 0,
            "opportunities_found": 0,
            "trades_attempted": 0,
            "trades_filled": 0,
            "total_profit": 0.0,
        }

    def init_executor(self, key: str):
        if self.config.mode == "live":
            self.executor = LiveExecutor(key)
            log.info("Live executor initialized")

    def evaluate_with_books(
        self,
        raw_opps: list,
        markets: list[Market],
    ) -> list[EvaluatedOpportunity]:
        """Enrich raw opportunities with orderbook depth."""
        qmap = {m.question: m for m in markets}
        evaluated = []

        for opp in raw_opps:
            ev = EvaluatedOpportunity(
                kind=opp.kind,
                markets=opp.markets,
                legs=[],
                cost=opp.cost,
                payout=opp.payout,
                profit=opp.profit,
                roi_pct=opp.roi_pct,
            )

            # Skip tiny-edge opportunities
            if ev.roi_pct < self.config.min_edge_pct:
                ev.fillable = False
                ev.skip_reason = f"edge {ev.roi_pct:.1f}% < min {self.config.min_edge_pct}%"
                evaluated.append(ev)
                continue

            # Check orderbook depth for each leg
            all_fillable = True
            for leg in opp.legs:
                token_id = leg.get("token_id", "")
                if not token_id:
                    all_fillable = False
                    leg["book_depth"] = 0
                    leg["skip_reason"] = "no token_id"
                    continue

                book = self.book_fetcher.get_book(token_id)
                if not book:
                    all_fillable = False
                    leg["book_depth"] = 0
                    leg["skip_reason"] = "book fetch failed"
                    continue

                side = leg.get("side", "BUY")
                price = leg.get("price", 0)

                if side == "BUY":
                    depth = book.available_size_at_price("BUY", price)
                else:
                    depth = book.available_size_at_price("SELL", price)

                leg["book_depth"] = round(depth, 2)
                leg["best_bid"] = book.best_bid
                leg["best_ask"] = book.best_ask
                leg["spread"] = book.spread

                if depth < self.config.min_book_depth:
                    all_fillable = False
                    leg["skip_reason"] = f"depth ${depth:.2f} < min ${self.config.min_book_depth}"

            ev.fillable = all_fillable
            if not all_fillable:
                ev.skip_reason = "insufficient book depth"
            evaluated.append(ev)

        return evaluated

    def execute_trade(self, opp: EvaluatedOpportunity) -> bool:
        """Execute a verified opportunity."""
        arb_group = f"arb_{int(time.time())}"

        for leg in opp.legs:
            token_id = leg.get("token_id", "")
            side = leg.get("side", "BUY")
            price = leg.get("price", 0)
            size = min(
                leg.get("book_depth", 0),
                self.tracker.limits.max_position_usd / price if price > 0 else 0,
            )

            if size < 1.0:
                log.info(f"  SKIP leg: size ${size:.2f} too small")
                continue

            market_key = token_id[:20]
            can, reason = self.tracker.can_open(market_key, size * price)
            if not can:
                log.info(f"  SKIP leg: {reason}")
                continue

            if self.config.mode == "live" and self.executor:
                try:
                    resp = self.executor.place_limit(token_id, price, size, side)
                    filled = True  # optimistic; parse resp for actual fill
                except Exception as e:
                    log.error(f"  ORDER FAILED: {e}")
                    filled = False
            else:
                # Paper mode: assume immediate fill
                filled = True
                log.info(f"  PAPER FILL: {side} {size:.2f} @ ${price:.4f} ({token_id[:16]}...)")

            if filled:
                pos = Position(
                    token_id=token_id,
                    market_question=leg.get("market", "unknown")[:200],
                    side=side,
                    avg_price=price,
                    size=size,
                    filled_size=size,
                    cost_basis=price * size,
                    opened_at=datetime.now(timezone.utc).isoformat(),
                    status="filled",
                    arb_group=arb_group,
                )
                self.tracker.record_fill(pos)
                self.stats["trades_filled"] += 1

        self.stats["trades_attempted"] += 1
        return True

    def scan_once(self) -> dict:
        """Run a single scan cycle."""
        log.info(f"{'='*60}")
        log.info(f"Scan #{self.stats['scans'] + 1} starting...")

        # Fetch markets
        markets = fetch_events(self.config.max_events)
        markets = [m for m in markets if m.price_sum > 0.01]
        log.info(f"  Markets: {len(markets)}")

        # Detect relationships
        within = detect_within_event(markets)
        negrisk = detect_negrisk_groups(markets)
        subset = detect_temporal_subsets(markets)
        entity = detect_entity_similarity(markets)
        all_rels = within + negrisk + subset + entity
        log.info(f"  Relationships: {len(all_rels)}")

        # Find raw opportunities
        single_opps = []
        for m in markets:
            opp = scan_single_condition(m)
            if opp:
                single_opps.append(opp)
        combo_opps = scan_combinatorial(markets, all_rels)
        raw = single_opps + combo_opps
        log.info(f"  Raw opportunities: {len(raw)}")

        # Evaluate with orderbooks
        evaluated = self.evaluate_with_books(raw, markets)
        fillable = [e for e in evaluated if e.fillable]
        skipped = [e for e in evaluated if not e.fillable]

        self.stats["opportunities_found"] += len(raw)
        log.info(f"  Fillable: {len(fillable)} | Skipped: {len(skipped)}")

        # Execute fillable opportunities
        if fillable and self.config.mode in ("paper", "live"):
            fillable.sort(key=lambda o: o.roi_pct, reverse=True)
            for opp in fillable[:5]:  # max 5 trades per cycle
                log.info(f"  EXECUTING: {opp.roi_pct:.1f}% ROI | {opp.markets[0][:50]}")
                self.execute_trade(opp)

        # Log
        if self.config.log_all_opps:
            for opp in fillable[:10]:
                log.info(f"  ✓ {opp.roi_pct:.1f}% | cost=${opp.cost:.4f} | {opp.markets[0][:50]}")

        self.stats["scans"] += 1

        # Save scan results
        os.makedirs(self.config.output_dir, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        scan_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "markets": len(markets),
            "relationships": len(all_rels),
            "raw_opps": len(raw),
            "fillable": len(fillable),
            "fillable_opps": [
                {"kind": o.kind, "markets": o.markets, "roi_pct": o.roi_pct,
                 "cost": o.cost, "legs": o.legs}
                for o in fillable
            ],
            "skipped": [
                {"kind": o.kind, "markets": o.markets, "roi_pct": o.roi_pct,
                 "reason": o.skip_reason}
                for o in skipped[:20]
            ],
        }
        with open(f"{self.config.output_dir}/scan_{ts}.json", "w") as f:
            json.dump(scan_data, f, indent=2, default=str)

        return scan_data

    def run_loop(self):
        """Main autonomous loop."""
        self.running = True
        log.info(f"Bot starting in {self.config.mode} mode")
        log.info(f"  Poll interval: {self.config.poll_interval}s")
        log.info(f"  Min edge: {self.config.min_edge_pct}%")
        log.info(f"  Max exposure: ${self.tracker.limits.max_total_exposure_usd}")
        log.info("")

        while self.running:
            try:
                self.scan_once()
                log.info(f"  Sleeping {self.config.poll_interval}s...")
                log.info(f"  Stats: {json.dumps(self.stats)}")
                log.info(f"  Positions: {json.dumps(self.tracker.summary())}")
                log.info("")
                time.sleep(self.config.poll_interval)
            except KeyboardInterrupt:
                log.info("Interrupted — shutting down")
                self.running = False
            except Exception as e:
                log.error(f"Scan error: {e}", exc_info=True)
                time.sleep(30)  # back off on errors

        # Final save
        self.tracker.save()
        log.info("Bot stopped")
        log.info(f"Final stats: {json.dumps(self.stats)}")


# ── Entry Point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Polymarket Arbitrage Bot")
    parser.add_argument("--mode", choices=["scan", "paper", "live"], default="scan")
    parser.add_argument("--interval", type=int, default=60, help="Poll interval (seconds)")
    parser.add_argument("--max-events", type=int, default=200)
    parser.add_argument("--min-edge", type=float, default=1.0, help="Min edge %% to trade")
    parser.add_argument("--min-depth", type=float, default=10.0, help="Min $ book depth")
    parser.add_argument("--once", action="store_true", help="Run single scan and exit")
    args = parser.parse_args()

    config = BotConfig(
        mode=args.mode,
        poll_interval=args.interval,
        max_events=args.max_events,
        min_edge_pct=args.min_edge,
        min_book_depth=args.min_depth,
    )

    bot = ArbBot(config)

    # Load private key for live mode
    key = os.environ.get("POLYMARKET_KEY", "")
    if args.mode == "live":
        if not key:
            log.error("POLYMARKET_KEY env var required for live mode")
            sys.exit(1)
        bot.init_executor(key)

    # Graceful shutdown
    def handle_signal(sig, frame):
        log.info("Shutdown signal received")
        bot.running = False
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    if args.once:
        bot.scan_once()
    else:
        bot.run_loop()


if __name__ == "__main__":
    main()
