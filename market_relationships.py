#!/usr/bin/env python3
"""
Polymarket Arbitrage Detector — Market Relationship Mapper

Discovers logical relationships between prediction markets:
  1. Within-event (given by Polymarket grouping)
  2. Temporal/subset (deadline markets: "by June 2026" ⊂ "by Dec 2026")
  3. Entity similarity (same subject across different events)
  4. Logical implication chains (A implies B)
  5. NegRisk groups (mutually exclusive outcomes within events)

Reference: arxiv.org/abs/2508.03474
"""

import re
import json
import time
import requests
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────────

GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
MAX_EVENTS = 200
MIN_VOLUME = 5_000


# ── Data Models ─────────────────────────────────────────────────────────────

@dataclass
class Market:
    id: str
    question: str
    condition_id: str
    token_ids: list[str]
    outcome_prices: list[float]
    volume: float
    liquidity: float
    event_id: str
    event_title: str
    neg_risk: bool
    end_date: Optional[datetime] = None
    group_item_title: str = ""

    @property
    def yes_price(self) -> float:
        return self.outcome_prices[0] if self.outcome_prices else 0.0

    @property
    def no_price(self) -> float:
        return self.outcome_prices[1] if self.outcome_prices else 0.0

    @property
    def price_sum(self) -> float:
        return self.yes_price + self.no_price


@dataclass
class Relationship:
    kind: str
    market_a: str
    market_b: str
    confidence: float
    detail: str = ""


@dataclass
class ArbOpportunity:
    kind: str
    markets: list[str]
    legs: list[dict]
    cost: float
    payout: float
    profit: float
    roi_pct: float


# ── API Layer ───────────────────────────────────────────────────────────────

def fetch_events(limit: int = MAX_EVENTS) -> list[Market]:
    """Fetch active events and flatten their markets."""
    all_markets = []
    offset = 0
    page_size = 100

    while len(all_markets) < limit * 10:  # rough upper bound on market count
        r = requests.get(f"{GAMMA_API}/events", params={
            "limit": page_size,
            "offset": offset,
            "active": True,
            "closed": False,
            "order": "volume24hr",
            "ascending": False,
        }, timeout=15)
        r.raise_for_status()
        batch = r.json()

        if not batch:
            break

        for ev in batch:
            eid = str(ev.get("id", ""))
            etitle = ev.get("title", "")
            neg_risk = bool(ev.get("negRisk", False))

            for m in ev.get("markets", []):
                if not m.get("active") or m.get("closed"):
                    continue
                prices = json.loads(m.get("outcomePrices", "[]") or "[]")
                tokens = json.loads(m.get("clobTokenIds", "[]") or "[]")
                vol = float(m.get("volume", 0) or 0)
                liq = float(m.get("liquidity", 0) or 0)

                if vol < MIN_VOLUME and liq < MIN_VOLUME:
                    continue

                end_str = m.get("endDate")
                end_dt = None
                if end_str:
                    try:
                        end_dt = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
                    except Exception:
                        pass

                all_markets.append(Market(
                    id=str(m["id"]),
                    question=m.get("question", ""),
                    condition_id=m.get("conditionId", ""),
                    token_ids=tokens,
                    outcome_prices=[float(p) for p in prices],
                    volume=vol,
                    liquidity=liq,
                    event_id=eid,
                    event_title=etitle,
                    neg_risk=neg_risk,
                    end_date=end_dt,
                    group_item_title=m.get("groupItemTitle", ""),
                ))

        offset += page_size

    print(f"Fetched {len(all_markets)} active markets from events API")
    return all_markets


# ── Text Utilities ──────────────────────────────────────────────────────────

def extract_entity(q: str) -> str:
    """Strip dates/deadlines, get core topic."""
    q = q.lower().strip().rstrip('?')
    for pat in [
        r'\bby\s+\w+\s+\d{1,2},?\s+\d{4}', r'\bby\s+\w+\s+\d{4}',
        r'\bby\s+\d{4}', r'\bby\s+end of\s+\w+', r'\bin\s+\d{4}\b',
        r'\bbefore\s+\w+\s+\d{4}', r'\bby\s+\w+\s+\d{1,2}\b',
    ]:
        q = re.sub(pat, '', q)
    return re.sub(r'\s+', ' ', q).strip().rstrip('?,. ')


def normalize_entity(e: str) -> set[str]:
    """Get normalized word set for similarity."""
    e = e.lower()
    e = re.sub(r'[^a-z0-9\s]', '', e)
    stopwords = {'will', 'does', 'is', 'are', 'the', 'a', 'an', 'be', 'to', 'of',
                 'in', 'any', 'by', 'end', 'be', 'new', 'before', 'after', 'or',
                 'and', 'have', 'has', 'for', 'on', 'at', 'from', 'with'}
    words = set(e.split()) - stopwords
    return words


def jaccard(a: str, b: str) -> float:
    wa, wb = normalize_entity(a), normalize_entity(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


# ── Relationship Detectors ─────────────────────────────────────────────────

def detect_within_event(markets: list[Market]) -> list[Relationship]:
    by_event: dict[str, list[Market]] = defaultdict(list)
    for m in markets:
        by_event[m.event_id].append(m)

    rels = []
    for eid, group in by_event.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                rels.append(Relationship(
                    kind="within_event",
                    market_a=group[i].question,
                    market_b=group[j].question,
                    confidence=0.95,
                    detail=f"Event: {group[i].event_title}"
                ))
    return rels


def detect_negrisk_groups(markets: list[Market]) -> list[Relationship]:
    rels = []
    by_event: dict[str, list[Market]] = defaultdict(list)
    for m in markets:
        if m.neg_risk:
            by_event[m.event_id].append(m)

    for eid, group in by_event.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                rels.append(Relationship(
                    kind="negrisk",
                    market_a=group[i].question,
                    market_b=group[j].question,
                    confidence=1.0,
                    detail=f"NegRisk event: {group[i].event_title}"
                ))
    return rels


def detect_temporal_subsets(markets: list[Market]) -> list[Relationship]:
    """
    Find markets with same entity but different deadlines.
    "X by June 2026" ⊂ "X by Dec 2026"
    If P(June) > P(Dec), that's a mispricing.
    """
    by_entity: dict[str, list[Market]] = defaultdict(list)
    for m in markets:
        ent = extract_entity(m.question)
        if len(ent) > 5:  # skip trivial entities
            by_entity[ent].append(m)

    rels = []
    for ent, group in by_entity.items():
        if len(group) < 2:
            continue
        dated = [(m, m.end_date) for m in group if m.end_date]
        dated.sort(key=lambda x: x[1])

        for i in range(len(dated)):
            for j in range(i + 1, len(dated)):
                m_early, m_late = dated[i][0], dated[j][0]
                flag = ""
                if m_early.yes_price > m_late.yes_price + 0.01:
                    flag = " ⚠️ MISPRICED (earlier has higher prob)"

                rels.append(Relationship(
                    kind="subset",
                    market_a=m_early.question,
                    market_b=m_late.question,
                    confidence=0.85 if not flag else 0.95,
                    detail=f"Temporal: {m_early.end_date:%Y-%m} → {m_late.end_date:%Y-%m}{flag}"
                ))
    return rels


def detect_entity_similarity(markets: list[Market], threshold: float = 0.45) -> list[Relationship]:
    """Find cross-event markets about the same entity."""
    rels = []
    for i in range(len(markets)):
        for j in range(i + 1, len(markets)):
            if markets[i].event_id == markets[j].event_id:
                continue
            sim = jaccard(markets[i].question, markets[j].question)
            if sim >= threshold:
                rels.append(Relationship(
                    kind="entity",
                    market_a=markets[i].question,
                    market_b=markets[j].question,
                    confidence=sim,
                    detail=f"Jaccard similarity: {sim:.2f}"
                ))
    return rels


# ── Arbitrage Detection ────────────────────────────────────────────────────

def scan_single_condition(m: Market) -> Optional[ArbOpportunity]:
    """YES + NO ≠ $1.00 → guaranteed profit."""
    if len(m.token_ids) < 2 or m.price_sum == 0:
        return None
    total = m.price_sum
    if 0.001 < total < 0.995:
        profit = 1.0 - total
        return ArbOpportunity(
            kind="single_condition",
            markets=[m.question],
            legs=[
                {"side": "BUY", "price": m.yes_price, "token_id": m.token_ids[0]},
                {"side": "BUY", "price": m.no_price, "token_id": m.token_ids[1]},
            ],
            cost=total, payout=1.0, profit=profit,
            roi_pct=profit / total * 100,
        )
    elif total > 1.005:
        profit = total - 1.0
        return ArbOpportunity(
            kind="single_condition",
            markets=[m.question],
            legs=[
                {"side": "SELL", "price": m.yes_price, "token_id": m.token_ids[0]},
                {"side": "SELL", "price": m.no_price, "token_id": m.token_ids[1]},
            ],
            cost=1.0, payout=total, profit=profit,
            roi_pct=profit * 100,
        )
    return None


def scan_combinatorial(markets: list[Market], rels: list[Relationship]) -> list[ArbOpportunity]:
    """
    Combinatorial arbitrage across related markets.

    For temporal subsets A ⊂ B:
      P(A by t1) ≤ P(B by t2) where t1 < t2
      If violated: buy YES on B, buy NO on A → one resolves true → profit = spread

    For entity implication:
      If A logically implies B, then P(A∩B) + P(A∩¬B) + P(¬A∩B) ≤ 1
    """
    opps = []
    qmap = {m.question: m for m in markets}

    for rel in rels:
        ma = qmap.get(rel.market_a)
        mb = qmap.get(rel.market_b)
        if not ma or not mb or not ma.token_ids or not mb.token_ids:
            continue

        if rel.kind == "subset":
            # Temporal subset: P(early) should be ≤ P(late)
            # If P(early_yes) > P(late_yes): mispriced
            if ma.yes_price > mb.yes_price + 0.02:
                spread = ma.yes_price - mb.yes_price
                # Buy YES(late) + NO(early)
                cost = mb.yes_price + (1.0 - ma.yes_price)
                opps.append(ArbOpportunity(
                    kind="combinatorial",
                    markets=[ma.question, mb.question],
                    legs=[
                        {"side": "BUY", "outcome": "YES", "price": mb.yes_price,
                         "token_id": mb.token_ids[0]},
                        {"side": "BUY", "outcome": "NO", "price": 1.0 - ma.yes_price,
                         "token_id": ma.token_ids[1]},
                    ],
                    cost=cost, payout=1.0, profit=1.0 - cost,
                    roi_pct=(1.0 - cost) / cost * 100 if cost > 0 else 0,
                ))

        elif rel.kind == "entity" and rel.confidence > 0.6:
            a_yes, b_yes = ma.yes_price, mb.yes_price
            if a_yes + (1.0 - b_yes) < 0.99:
                cost = a_yes + (1.0 - b_yes)
                opps.append(ArbOpportunity(
                    kind="combinatorial",
                    markets=[ma.question, mb.question],
                    legs=[
                        {"side": "BUY", "outcome": "YES", "price": a_yes,
                         "token_id": ma.token_ids[0]},
                        {"side": "BUY", "outcome": "NO", "price": 1.0 - b_yes,
                         "token_id": mb.token_ids[1]},
                    ],
                    cost=cost, payout=1.0, profit=1.0 - cost,
                    roi_pct=(1.0 - cost) / cost * 100 if cost > 0 else 0,
                ))

    return opps


# ── Main ────────────────────────────────────────────────────────────────────

def scan():
    print("=" * 70)
    print("  POLYMARKET ARBITRAGE DETECTOR — Relationship Mapper")
    print("  Ref: arxiv.org/abs/2508.03474")
    print("=" * 70)

    # 1. Fetch
    print("\n[1/5] Fetching markets via events API...")
    markets = fetch_events(MAX_EVENTS)
    markets = [m for m in markets if m.price_sum > 0.01]
    print(f"  → {len(markets)} tradeable markets\n")

    # 2. Relationships
    print("[2/5] Mapping market relationships...")
    within = detect_within_event(markets)
    print(f"  Within-event:     {len(within)} pairs")

    negrisk = detect_negrisk_groups(markets)
    print(f"  NegRisk groups:   {len(negrisk)} pairs")

    subset = detect_temporal_subsets(markets)
    print(f"  Temporal subset:  {len(subset)} pairs")

    entity = detect_entity_similarity(markets)
    print(f"  Cross-event entity: {len(entity)} pairs")

    all_rels = within + negrisk + subset + entity
    print(f"  TOTAL:            {len(all_rels)} relationships\n")

    # 3. Single-condition
    print("[3/5] Scanning single-condition arbitrage...")
    single_opps = []
    for m in markets:
        opp = scan_single_condition(m)
        if opp:
            single_opps.append(opp)
    single_opps.sort(key=lambda o: o.roi_pct, reverse=True)

    if single_opps:
        print(f"  ✅ Found {len(single_opps)} opportunities!")
        for opp in single_opps[:10]:
            print(f"    {opp.roi_pct:+.1f}% ROI | cost=${opp.cost:.4f} | ${opp.markets[0][:55]}")
    else:
        print("  No single-condition arb (prices sum correctly)")
    print()

    # 4. Combinatorial
    print("[4/5] Scanning combinatorial arbitrage...")
    combo_opps = scan_combinatorial(markets, all_rels)
    combo_opps.sort(key=lambda o: o.roi_pct, reverse=True)

    if combo_opps:
        print(f"  ✅ Found {len(combo_opps)} opportunities!")
        for opp in combo_opps[:10]:
            print(f"    {opp.roi_pct:+.1f}% ROI | cost=${opp.cost:.4f}")
            for leg in opp.legs:
                print(f"      {leg['side']} {leg.get('outcome', '')} @ ${leg['price']:.4f} — {leg.get('token_id', '')[:20]}...")
    else:
        print("  No combinatorial arb in current snapshot")
    print()

    # 5. Report
    print("[5/5] Relationship highlights:")
    print("-" * 70)

    # Temporal violations
    temporal_violations = [r for r in subset if "MISPRICED" in r.detail]
    if temporal_violations:
        print(f"\n  ⚠️  TEMPORAL SUBSET VIOLATIONS ({len(temporal_violations)}):")
        for r in temporal_violations[:10]:
            print(f"    A: {r.market_a[:55]}")
            print(f"    B: {r.market_b[:55]}")
            print(f"    {r.detail}\n")

    # Multi-market events
    ev_groups = defaultdict(list)
    for r in within:
        ev_groups[r.detail].append(r)
    big = {k: v for k, v in ev_groups.items() if len(v) >= 3}
    if big:
        print(f"\n  📦 LARGE EVENTS (≥3 market pairs):")
        for detail, rels in list(big.items())[:5]:
            # Extract event title
            title = detail.replace("Event: ", "")
            print(f"    {title}: {len(rels)} pairs")

    # Cross-event entity
    top_entity = sorted(entity, key=lambda r: r.confidence, reverse=True)[:10]
    if top_entity:
        print(f"\n  🔗 TOP CROSS-EVENT RELATIONSHIPS:")
        for r in top_entity:
            print(f"    [{r.confidence:.0%}] {r.market_a[:45]}")
            print(f"         ↔ {r.market_b[:45]}")

    print()
    print("=" * 70)
    total_opp = len(single_opps) + len(combo_opps)
    print(f"  {total_opp} arb opportunities | {len(all_rels)} relationships | {len(markets)} markets")
    print("=" * 70)

    # Export
    export = {
        "scan_time": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "markets": len(markets),
            "relationships": len(all_rels),
            "within_event": len(within),
            "negrisk": len(negrisk),
            "subset": len(subset),
            "entity": len(entity),
            "single_condition_opps": len(single_opps),
            "combinatorial_opps": len(combo_opps),
        },
        "top_relationships": [
            {"kind": r.kind, "a": r.market_a, "b": r.market_b,
             "confidence": r.confidence, "detail": r.detail}
            for r in sorted(all_rels, key=lambda x: x.confidence, reverse=True)[:100]
        ],
        "arbitrage": [
            {"kind": o.kind, "markets": o.markets, "legs": o.legs,
             "cost": o.cost, "profit": o.profit, "roi_pct": o.roi_pct}
            for o in (single_opps + combo_opps)
        ],
    }
    with open("scan_results.json", "w") as f:
        json.dump(export, f, indent=2, default=str)
    print(f"\nExported to scan_results.json")


if __name__ == "__main__":
    scan()
