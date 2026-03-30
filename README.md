# Polymarket Arbitrage Bot

Autonomous detection and execution of arbitrage opportunities on Polymarket.

Based on the methodology from [Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets](https://arxiv.org/abs/2508.03474) — which found $40M in arbitrage profit extracted from Polymarket over 12 months.

## Architecture

```
market_relationships.py   — Fetches markets, maps logical relationships
orderbook.py              — Orderbook fetching + depth analysis
positions.py              — Position tracking, P&L, risk management
bot.py                    — Autonomous loop: scan → evaluate → execute
```

## How It Works

### 1. Market Relationship Discovery
Maps which markets are logically related across 5 relationship types:

| Type | Description | Example |
|------|-------------|---------|
| **Within-event** | Markets grouped by Polymarket | FIFA World Cup winner: 60 markets |
| **NegRisk** | Explicitly mutually exclusive outcomes | Democratic nominee candidates |
| **Temporal subset** | Same entity, different deadlines | "X by June 2026" ⊂ "X by Dec 2026" |
| **Entity similarity** | Same topic across different events | "Bitcoin $80k" + "Bitcoin $80k this week" |
| **Implication** | Logical A→B constraints | "Republicans win PA by 5+" → "Trump wins PA" |

### 2. Arbitrage Detection

**Single-condition:** YES + NO ≠ $1.00 → buy both below $1, guaranteed payout.

**Combinatorial:** Across related markets where joint pricing violates logical constraints.
- If P(by June) > P(by December): mispriced (June ⊂ December)
- Buy the cheaper YES + the cheaper NO → one must resolve true

### 3. Orderbook Verification
Every opportunity is checked against live orderbooks to verify:
- Sufficient depth at fillable prices
- Spread acceptable for the edge
- Liquidity meets minimum thresholds

### 4. Execution
- **Scan mode**: Report only
- **Paper mode**: Simulated fills for backtesting
- **Live mode**: Places actual orders via CLOB API

## Usage

```bash
# Install dependencies
pip install requests py-clob-client

# Single scan (no trading)
python3 bot.py --mode scan --once

# Continuous scan mode (60s intervals)
python3 bot.py --mode scan --interval 60

# Paper trading (simulated)
python3 bot.py --mode paper --interval 30

# Live trading (requires private key)
export POLYMARKET_KEY=0x...
python3 bot.py --mode live --interval 30 --min-edge 1.5
```

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--mode` | `scan` | scan / paper / live |
| `--interval` | `60` | Seconds between scans |
| `--max-events` | `200` | Events to fetch per scan |
| `--min-edge` | `1.0` | Minimum % edge to trade |
| `--min-depth` | `10.0` | Minimum $ depth at price |
| `--once` | false | Single scan and exit |

## Risk Controls (in `positions.py`)

- Max single position: $50
- Max total exposure: $200
- Max open positions: 20
- Min edge: 1%
- Min liquidity: $10 depth
- Daily loss limit: $50 (auto-stop)
- Cooldown: 60s between re-entering same market

## Output

Each scan produces:
- Console log of opportunities
- `runs/scan_YYYYMMDD_HHMMSS.json` with full results
- `positions.json` with trade history and P&L

## First Run Results

```
2,026 markets scanned
44,396 relationships mapped
145 raw opportunities found
106 fillable after orderbook verification

Top opportunities (live snapshot):
  +261% ROI  Trump visit China May vs April temporal violation
  +420% ROI  Trump visit China June vs April
  +723% ROI  SpaceX IPO Dec vs April
  +19.8% ROI  Iran conflict end Dec vs June
  +152% ROI  SpaceX IPO June vs April
```

## Disclaimer

This is a research/educational tool. Prediction market trading involves risk.
Start with paper mode. If going live, use small amounts and understand the risks.
