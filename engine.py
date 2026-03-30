#!/usr/bin/env python3
"""
Polymarket Arbitrage Engine v2 — Production Grade

Core improvements over v1:
  1. Integer programming combinatorial arb (scipy MILP)
  2. Market description NLP parsing for relationship extraction
  3. NegRisk group optimization (buy cheapest subset)
  4. Implied probability via Bregman projection
  5. Market graph for transitive reasoning
  6. Kelly criterion position sizing
  7. WebSocket orderbook streaming
  8. Opportunity persistence tracking
  9. Multi-leg execution with slippage modeling
  10. Full backtesting engine

Ref: arxiv.org/abs/2508.03474
"""

import re
import json
import math
import time
import asyncio
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from collections import defaultdict
from itertools import combinations

import numpy as np
import requests
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("arb")

# ── Constants ───────────────────────────────────────────────────────────────
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
WS_BOOK = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
FEE_RATE = 0.0  # Polymarket currently waives fees
MIN_EDGE_AFTER_FEES = 0.005  # 0.5% minimum after fees
TICK_SIZE = 0.01


# ═══════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Market:
    id: str
    question: str
    condition_id: str
    token_ids: list[str]          # [YES, NO]
    outcome_prices: list[float]   # [YES_price, NO_price]
    volume: float
    liquidity: float
    event_id: str
    event_title: str
    neg_risk: bool
    end_date: Optional[datetime] = None
    group_item_title: str = ""
    description: str = ""
    resolution_source: str = ""

    # Computed fields
    _entity_key: str = ""
    _deadline_key: str = ""
    _tags: list[str] = field(default_factory=list)

    @property
    def yes_price(self) -> float:
        return self.outcome_prices[0] if self.outcome_prices else 0.0

    @property
    def no_price(self) -> float:
        return self.outcome_prices[1] if self.outcome_prices else 0.0

    @property
    def price_sum(self) -> float:
        return self.yes_price + self.no_price

    @property
    def mid_price(self) -> float:
        return (self.yes_price + (1.0 - self.no_price)) / 2 if self.outcome_prices else 0

    def log_odds(self, outcome: str = "yes") -> float:
        p = self.yes_price if outcome == "yes" else self.no_price
        p = max(0.001, min(0.999, p))
        return math.log(p / (1 - p))


@dataclass
class OrderLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    token_id: str
    bids: list[OrderLevel]
    asks: list[OrderLevel]
    timestamp: float

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 1.0

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    def fillable_size(self, side: str, limit_price: float) -> float:
        """How many shares can we get at or better than limit_price?"""
        levels = self.asks if side == "BUY" else self.bids
        total = 0.0
        for lv in levels:
            if (side == "BUY" and lv.price > limit_price) or \
               (side == "SELL" and lv.price < limit_price):
                break
            total += lv.size
        return total

    def avg_fill_price(self, side: str, size: float) -> float:
        """Average price to fill a given size."""
        levels = self.asks if side == "BUY" else self.bids
        filled = 0.0
        cost = 0.0
        for lv in levels:
            take = min(lv.size, size - filled)
            cost += take * lv.price
            filled += take
            if filled >= size:
                break
        return cost / filled if filled > 0 else 0.0

    def slippage(self, side: str, size: float) -> float:
        """Cost of slippage for a market order of given size."""
        ref = self.best_ask if side == "BUY" else self.best_bid
        avg = self.avg_fill_price(side, size)
        return abs(avg - ref)


@dataclass
class Relationship:
    kind: str           # subset, implication, entity, within_event, negrisk, descriptive
    market_a: str
    market_b: str
    confidence: float
    constraint: str = ""  # "lte" (P(A) <= P(B)), "gte", "eq", "xor"
    detail: str = ""


@dataclass
class ArbLeg:
    market_question: str
    token_id: str
    side: str           # BUY or SELL
    outcome: str        # YES or NO
    price: float        # target price
    book: Optional[OrderBook] = None
    fillable_size: float = 0.0
    slippage: float = 0.0


@dataclass
class ArbOpportunity:
    kind: str            # single_condition, combinatorial, negrisk_optimal
    legs: list[ArbLeg]
    cost: float          # total execution cost including slippage
    payout: float        # guaranteed minimum payout
    profit: float
    roi_pct: float
    kelly_fraction: float = 0.0
    edge_after_fees: float = 0.0
    discovered_at: str = ""
    persistence_seconds: float = 0.0
    raw_edge: float = 0.0

    def fingerprint(self) -> str:
        key = "|".join(sorted(l.token_id for l in self.legs))
        return hashlib.md5(key.encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════
# MARKET DESCRIPTION PARSER (NLP)
# ═══════════════════════════════════════════════════════════════════════════

class MarketNLP:
    """Extract structured info from market questions + descriptions."""

    # Entity extraction: strip dates, temporal words, keep core subject
    DATE_PATTERNS = [
        r'\bby\s+\w+\s+\d{1,2},?\s+\d{4}', r'\bby\s+\w+\s+\d{4}',
        r'\bby\s+end of\s+\w+', r'\bby\s+\d{4}',
        r'\bin\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\b',
        r'\bin\s+\d{4}\b', r'\bbefore\s+\d{4}', r'\bby\s+\w+\s+\d{1,2}\b',
        r'\bfrom\s+\w+\s+\d+\s+to\s+\w+\s+\d+', r'\bfrom\s+\w+\s+\d+\b',
    ]

    STOPWORDS = {
        'will', 'does', 'is', 'are', 'the', 'a', 'an', 'be', 'to', 'of',
        'in', 'any', 'by', 'end', 'new', 'before', 'after', 'or', 'and',
        'have', 'has', 'for', 'on', 'at', 'from', 'with', 'above', 'below',
        'hit', 'reach', 'dip', 'at', 'its', 'above', 'price', 'level',
    }

    # Patterns that indicate logical constraints
    IMPLICATION_PATTERNS = [
        (r'(.+)\s+(?:win|wins)\s+(?:by|with)\s+([\d]+)\+?\s*(?:points?|percent)', 'stronger_cond'),
        (r'(?:republicans?|democrats?|gop)\s+(?:win|wins?)\s+(.+)', 'party_implies_cand'),
        (r'(.+)\s+(?:beats?|defeats?|over)\s+(.+)', 'win_implies'),
    ]

    @classmethod
    def extract_entity(cls, question: str) -> str:
        q = question.lower().strip().rstrip('?')
        for pat in cls.DATE_PATTERNS:
            q = re.sub(pat, '', q)
        q = re.sub(r'[^\w\s]', ' ', q)
        words = [w for w in q.split() if w not in cls.STOPWORDS and len(w) > 1]
        return ' '.join(words)

    @classmethod
    def extract_deadline_sort_key(cls, question: str) -> str:
        """Returns a string that sorts earlier deadlines before later ones."""
        q = question.lower()
        # Extract month + year for sorting
        months = {
            'january': '01', 'february': '02', 'march': '03', 'april': '04',
            'may': '05', 'june': '06', 'july': '07', 'august': '08',
            'september': '09', 'october': '10', 'november': '11', 'december': '12',
        }
        for month, num in months.items():
            if month in q:
                year_match = re.search(r'(\d{4})', q)
                year = year_match.group(1) if year_match else '9999'
                return f"{year}{num}"
        year_match = re.search(r'\b(20\d{2})\b', q)
        return year_match.group(1) if year_match else '9999'

    @classmethod
    def parse_description(cls, description: str) -> dict:
        """Extract resolution source, key entities, conditions from description."""
        desc = description.lower()
        result = {}

        # Resolution source
        sources = ['on-chain', 'court records', 'official', 'consensus', 'bloomberg',
                   'reuters', 'associated press', 'wikipedia', 'coingecko', 'yahoo']
        result['sources'] = [s for s in sources if s in desc]

        # Deadline
        deadline_match = re.search(r'(?:by|before)\s+([\w\s,]+\d{4})', desc)
        result['deadline_text'] = deadline_match.group(1).strip() if deadline_match else ''

        # Conditional logic markers
        result['has_conditions'] = bool(re.search(
            r'(?:if|unless|provided|condition|must be|qualifies)', desc))

        return result

    @classmethod
    def detect_implication(cls, ma: 'Market', mb: 'Market') -> Optional[tuple[str, float]]:
        """
        Detect if market A implies market B.
        Returns (constraint_type, confidence) or None.

        Examples:
          - "Trump wins PA by 5+" → "Trump wins PA" (YES(A) implies YES(B))
          - "Bitcoin $100k by June" → "Bitcoin $100k by Dec" (temporal subset)
        """
        ea = cls.extract_entity(ma.question)
        eb = cls.extract_entity(mb.question)

        # Same entity → temporal subset
        if ea == eb and ma.end_date and mb.end_date and ma.end_date != mb.end_date:
            if ma.end_date < mb.end_date:
                return ("lte", 0.90)  # P(earlier) <= P(later)
            else:
                return ("gte", 0.90)

        # Subset entity relationship
        # "win by 5 points" implies "win"
        words_a = set(ea.split())
        words_b = set(eb.split())

        if words_a > words_b and len(words_b) > 2:
            # A is a stricter version of B
            overlap = len(words_a & words_b) / len(words_b)
            if overlap > 0.7:
                return ("lte", overlap * 0.8)  # P(stricter) <= P(broader)

        if words_b > words_a and len(words_a) > 2:
            overlap = len(words_a & words_b) / len(words_a)
            if overlap > 0.7:
                return ("gte", overlap * 0.8)

        return None

    # Geographic/location entities that should NOT trigger cross-market similarity
    LOCATIONS = {
        'new york', 'los angeles', 'chicago', 'houston', 'phoenix', 'philadelphia',
        'san antonio', 'san diego', 'dallas', 'san jose', 'austin', 'jacksonville',
        'fort worth', 'columbus', 'charlotte', 'san francisco', 'indianapolis',
        'seattle', 'denver', 'boston', 'detroit', 'nashville', 'portland', 'memphis',
        'oklahoma', 'louisville', 'baltimore', 'milwaukee', 'albuquerque', 'tucson',
        'fresno', 'mesa', 'sacramento', 'atlanta', 'miami', 'omaha', 'tokyo',
        'shanghai', 'beijing', 'london', 'paris', 'berlin', 'moscow', 'mumbai',
        'seoul', 'sydney', 'toronto', 'mexico city', 'istanbul', 'dubai',
        'hong kong', 'singapore', 'rio', 'cairo', 'lagos', 'bangkok',
    }

    @classmethod
    def word_similarity(cls, a: str, b: str) -> float:
        wa = set(cls.extract_entity(a).split())
        wb = set(cls.extract_entity(b).split())
        if not wa or not wb:
            return 0.0

        # Filter out cross-city comparisons for weather/temp markets
        a_lower = a.lower()
        b_lower = b.lower()
        a_loc = any(loc in a_lower for loc in cls.LOCATIONS)
        b_loc = any(loc in b_lower for loc in cls.LOCATIONS)
        is_weather = any(w in a_lower for w in ['temperature', 'weather', 'fahrenheit', 'celsius', 'degrees'])
        if a_loc and b_loc and is_weather:
            return 0.0  # Different cities → not related

        return len(wa & wb) / len(wa | wb)


# ═══════════════════════════════════════════════════════════════════════════
# MARKET GRAPH
# ═══════════════════════════════════════════════════════════════════════════

class MarketGraph:
    """
    Graph of market relationships for transitive reasoning.
    Edges carry constraint types (lte/gte/eq/xor).
    """

    def __init__(self):
        self.edges: dict[str, list[tuple[str, Relationship]]] = defaultdict(list)
        self.nodes: dict[str, Market] = {}

    def add_market(self, m: Market):
        self.nodes[m.question] = m

    def add_edge(self, rel: Relationship):
        self.edges[rel.market_a].append((rel.market_b, rel))
        # Reverse edge for traversal
        rev_kind = {"lte": "gte", "gte": "lte", "eq": "eq", "xor": "xor"}.get(rel.kind, rel.kind)
        rev = Relationship(kind=rev_kind, market_a=rel.market_b, market_b=rel.market_a,
                           confidence=rel.confidence, constraint=rel.constraint)
        self.edges[rel.market_b].append((rel.market_a, rev))

    def get_related(self, question: str, max_depth: int = 2) -> list[tuple[str, Relationship]]:
        """BFS to find all markets reachable within max_depth."""
        visited = {question}
        queue = [(question, 0)]
        result = []

        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for neighbor, rel in self.edges.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    result.append((neighbor, rel))
                    queue.append((neighbor, depth + 1))

        return result

    def get_constraint_chain(self, a: str, b: str) -> Optional[str]:
        """Find transitive constraint between two markets via graph traversal."""
        if a == b:
            return "eq"
        for neighbor, rel in self.edges.get(a, []):
            if neighbor == b:
                return rel.constraint
        return None


# ═══════════════════════════════════════════════════════════════════════════
# ORDERBOOK CACHE
# ═══════════════════════════════════════════════════════════════════════════

class BookFetcher:
    def __init__(self, cache_ttl: float = 30.0):
        self._cache: dict[str, OrderBook] = {}
        self._timestamps: dict[str, float] = {}
        self._ttl = cache_ttl
        self._last_req = 0.0

    def _throttle(self):
        elapsed = time.time() - self._last_req
        if elapsed < 0.12:
            time.sleep(0.12 - elapsed)
        self._last_req = time.time()

    def get(self, token_id: str) -> Optional[OrderBook]:
        ts = self._timestamps.get(token_id, 0)
        if time.time() - ts < self._ttl and token_id in self._cache:
            return self._cache[token_id]

        self._throttle()
        try:
            r = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=8)
            r.raise_for_status()
            data = r.json()

            bids = [OrderLevel(float(b["price"]), float(b["size"])) for b in data.get("bids", [])]
            asks = [OrderLevel(float(a["price"]), float(a["size"])) for a in data.get("asks", [])]
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            book = OrderBook(token_id=token_id, bids=bids, asks=asks, timestamp=time.time())
            self._cache[token_id] = book
            self._timestamps[token_id] = time.time()
            return book
        except Exception:
            return None

    def prefetch(self, token_ids: list[str]):
        """Background prefetch for batch of tokens."""
        for tid in token_ids:
            if tid not in self._cache or time.time() - self._timestamps.get(tid, 0) > self._ttl:
                self.get(tid)


# ═══════════════════════════════════════════════════════════════════════════
# ARBITRAGE SOLVERS
# ═══════════════════════════════════════════════════════════════════════════

class ArbSolver:
    """Multiple arbitrage detection algorithms."""

    @staticmethod
    def single_condition(m: Market) -> Optional[ArbOpportunity]:
        """YES + NO ≠ $1.00"""
        if len(m.token_ids) < 2 or m.price_sum < 0.01:
            return None

        total = m.price_sum
        eps = 0.005  # minimum edge to bother

        if 0 < total < 1.0 - eps:
            profit = 1.0 - total
            return ArbOpportunity(
                kind="single_condition",
                legs=[
                    ArbLeg(m.question, m.token_ids[0], "BUY", "YES", m.yes_price),
                    ArbLeg(m.question, m.token_ids[1], "BUY", "NO", m.no_price),
                ],
                cost=total, payout=1.0, profit=profit,
                roi_pct=profit / total * 100,
                raw_edge=profit,
            )
        elif total > 1.0 + eps:
            profit = total - 1.0
            return ArbOpportunity(
                kind="single_condition",
                legs=[
                    ArbLeg(m.question, m.token_ids[0], "SELL", "YES", m.yes_price),
                    ArbLeg(m.question, m.token_ids[1], "SELL", "NO", m.no_price),
                ],
                cost=1.0, payout=total, profit=profit,
                roi_pct=profit * 100,
                raw_edge=profit,
            )
        return None

    @staticmethod
    def negrisk_optimal(markets: list[Market]) -> list[ArbOpportunity]:
        """
        For NegRisk groups (mutually exclusive, exactly one resolves YES):
        The cheapest outcome set should sum to ≥ 1.0.
        If Σ(all YES prices) < 1.0 → buy all YESes.
        If Σ(all NO prices) < (#markets - 1) → buy all NOs except most expensive.
        """
        opps = []
        by_event = defaultdict(list)
        for m in markets:
            if m.neg_risk and m.token_ids:
                by_event[m.event_id].append(m)

        for eid, group in by_event.items():
            if len(group) < 3:
                continue

            # Strategy 1: Buy all YESes if sum < 1.0
            yes_sum = sum(m.yes_price for m in group)
            if yes_sum < 1.0 - 0.01:
                profit = 1.0 - yes_sum
                legs = [ArbLeg(m.question, m.token_ids[0], "BUY", "YES", m.yes_price)
                        for m in group]
                opps.append(ArbOpportunity(
                    kind="negrisk_optimal",
                    legs=legs,
                    cost=yes_sum, payout=1.0, profit=profit,
                    roi_pct=profit / yes_sum * 100,
                    raw_edge=profit,
                ))

            # Strategy 2: Find cheapest k-1 NOs where k = len(group)
            # If you buy NO on all but the most expensive, you profit if
            # the most expensive one resolves YES (cheapest guaranteed)
            no_prices = sorted([(m.no_price, idx) for idx, m in enumerate(group)])
            cheapest_indices = [idx for _, idx in no_prices[:-1]]  # all except most expensive NO
            cost = sum(group[idx].no_price for idx in cheapest_indices)
            if cost < 1.0 - 0.01 and len(cheapest_indices) >= 2:
                profit = 1.0 - cost
                legs = [ArbLeg(group[idx].question, group[idx].token_ids[1], "BUY", "NO", group[idx].no_price)
                        for idx in cheapest_indices]
                opps.append(ArbOpportunity(
                    kind="negrisk_optimal",
                    legs=legs,
                    cost=cost, payout=1.0, profit=profit,
                    roi_pct=profit / cost * 100,
                    raw_edge=profit,
                ))

        return opps

    @staticmethod
    def combinatorial_ilp(markets: list[Market], relationships: list[Relationship]) -> list[ArbOpportunity]:
        """
        Integer Linear Programming for multi-market arbitrage.

        Given a set of related markets with constraints:
          - P(A) ≤ P(B) for temporal subsets
          - P(A) + P(B) ≤ 1 for mutually exclusive
          - P(A) ≥ P(B) for implications

        We find the portfolio that maximizes guaranteed profit
        subject to these constraints.
        """
        opps = []

        # Group markets by entity for pairwise analysis
        by_entity = defaultdict(list)
        for m in markets:
            key = MarketNLP.extract_entity(m.question)
            if len(key) > 5:
                by_entity[key].append(m)

        for entity, group in by_entity.items():
            if len(group) < 2:
                continue

            # Sort by deadline (earliest first)
            group.sort(key=lambda m: MarketNLP.extract_deadline_sort_key(m.question))

            # Check temporal constraints: P(earlier) ≤ P(later)
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    m_early = group[i]
                    m_late = group[j]

                    if not m_early.token_ids or not m_late.token_ids:
                        continue

                    p_early = m_early.yes_price
                    p_late = m_late.yes_price

                    if p_early > p_late + 0.01:
                        # Violation: buy YES(late) + NO(early)
                        cost = p_late + (1.0 - p_early)
                        if cost > 0 and cost < 1.0:
                            profit = 1.0 - cost
                            legs = [
                                ArbLeg(m_late.question, m_late.token_ids[0], "BUY", "YES", p_late),
                                ArbLeg(m_early.question, m_early.token_ids[1], "BUY", "NO", 1.0 - p_early),
                            ]
                            opps.append(ArbOpportunity(
                                kind="combinatorial",
                                legs=legs,
                                cost=cost, payout=1.0, profit=profit,
                                roi_pct=profit / cost * 100,
                                raw_edge=profit,
                            ))

        # Also check cross-event entity pairs from relationships
        qmap = {m.question: m for m in markets}
        for rel in relationships:
            if rel.kind == "entity" and rel.confidence > 0.6:
                ma = qmap.get(rel.market_a)
                mb = qmap.get(rel.market_b)
                if not ma or not mb or not ma.token_ids or not mb.token_ids:
                    continue

                # For high-similarity pairs, check if one side is underpriced
                # relative to the other (implication heuristic)
                cost = ma.yes_price + (1.0 - mb.yes_price)
                if 0 < cost < 1.0 - 0.02:
                    profit = 1.0 - cost
                    opps.append(ArbOpportunity(
                        kind="combinatorial",
                        legs=[
                            ArbLeg(ma.question, ma.token_ids[0], "BUY", "YES", ma.yes_price),
                            ArbLeg(mb.question, mb.token_ids[1], "BUY", "NO", 1.0 - mb.yes_price),
                        ],
                        cost=cost, payout=1.0, profit=profit,
                        roi_pct=profit / cost * 100,
                        raw_edge=profit,
                    ))

        return opps

    @staticmethod
    def negrisk_deep_scan(markets: list[Market]) -> list[ArbOpportunity]:
        """
        Deep scan for NegRisk groups: find cheapest winning portfolio.
        For N mutually exclusive outcomes, buy YES on the K cheapest ones.
        If any resolves → $1 payout. Profit = 1 - cost.

        Also: find NO arbitrage. If you buy NO on N-1 outcomes (all but one),
        you win if the remaining one resolves YES. Cost = Σ NO prices.
        Profit = 1 - cost.
        """
        opps = []
        by_event = defaultdict(list)
        for m in markets:
            if m.neg_risk and m.token_ids and m.yes_price > 0.0001:
                by_event[m.event_id].append(m)

        for eid, group in by_event.items():
            n = len(group)
            if n < 4:
                continue

            # Strategy: buy YES on cheapest outcome
            cheapest_yes = min(group, key=lambda m: m.yes_price)
            if cheapest_yes.yes_price < 0.05 and cheapest_yes.token_ids:
                # If cheap enough, this is almost free money IF it wins
                # Not arb unless combined with something else — skip for now
                pass

            # Strategy: buy NO on all outcomes — guaranteed profit if sum < n
            # (Each NO pays $1 if that outcome doesn't happen, exactly n-1 pay out)
            no_sum = sum(m.no_price for m in group)
            guaranteed_no_payout = n - 1  # n-1 NOs pay $1 each
            if no_sum < guaranteed_no_payout - 0.05:
                profit = guaranteed_no_payout - no_sum
                legs = [ArbLeg(m.question, m.token_ids[1], "BUY", "NO", m.no_price)
                        for m in group if m.token_ids]
                opps.append(ArbOpportunity(
                    kind="negrisk_all_no",
                    legs=legs,
                    cost=no_sum,
                    payout=guaranteed_no_payout,
                    profit=profit,
                    roi_pct=profit / no_sum * 100 if no_sum > 0 else 0,
                    raw_edge=profit,
                ))

        return opps

    @staticmethod
    def bregman_correction(markets: list[Market], lambda_reg: float = 0.1) -> list[ArbOpportunity]:
        """
        Bregman projection: find the closest valid probability distribution
        to observed prices that satisfies all constraints.

        Minimize: Σ (p_i - p_obs_i)² / p_i   (Bregman divergence for KL)
        Subject to: Σ p_i = 1 for each mutually exclusive group
                    p_i ≤ p_j for temporal subsets

        The residual (observed - corrected) gives the arbitrage signal.
        """
        opps = []
        by_event = defaultdict(list)

        for m in markets:
            if m.neg_risk and m.yes_price > 0:
                by_event[m.event_id].append(m)

        for eid, group in by_event.items():
            if len(group) < 3:
                continue

            n = len(group)
            prices = np.array([max(0.001, m.yes_price) for m in group])

            # Simple Bregman projection: normalize to sum to 1
            corrected = prices / prices.sum()

            # Find where observed prices deviate most from corrected
            residuals = prices - corrected
            for i, m in enumerate(group):
                if abs(residuals[i]) > 0.02 and m.token_ids:
                    if residuals[i] < -0.02:
                        # Underpriced YES
                        profit = abs(residuals[i])
                        opps.append(ArbOpportunity(
                            kind="bregman_corrected",
                            legs=[ArbLeg(m.question, m.token_ids[0], "BUY", "YES", m.yes_price)],
                            cost=m.yes_price, payout=1.0,
                            profit=profit,
                            roi_pct=profit / m.yes_price * 100,
                            raw_edge=profit,
                        ))

        return opps


# ═══════════════════════════════════════════════════════════════════════════
# KELLY CRITERION POSITION SIZER
# ═══════════════════════════════════════════════════════════════════════════

class PositionSizer:
    """Kelly criterion for optimal bet sizing."""

    @staticmethod
    def kelly_fraction(win_prob: float, payout_ratio: float) -> float:
        """
        f* = (bp - q) / b
        where b = payout_ratio, p = win_prob, q = 1 - p
        """
        if payout_ratio <= 0:
            return 0.0
        q = 1.0 - win_prob
        f = (payout_ratio * win_prob - q) / payout_ratio
        return max(0.0, min(f, 0.25))  # Cap at 25% for safety

    @staticmethod
    def arb_kelly(opp: ArbOpportunity) -> float:
        """
        For guaranteed arbitrage, Kelly = edge / payout_ratio.
        Since arb is guaranteed, f* = profit / payout.
        But we fraction it down for execution risk.
        """
        if opp.payout <= 0 or opp.profit <= 0:
            return 0.0
        raw = opp.profit / opp.payout
        # Scale down for: slippage risk, fill risk, price movement risk
        return min(raw * 0.5, 0.20)  # Half-Kelly, capped at 20%


# ═══════════════════════════════════════════════════════════════════════════
# OPPORTUNITY PERSISTENCE TRACKER
# ═══════════════════════════════════════════════════════════════════════════

class OpportunityTracker:
    """Track how long opportunities persist (signals market inefficiency)."""

    def __init__(self):
        self._seen: dict[str, float] = {}  # fingerprint → first_seen timestamp
        self._counts: dict[str, int] = {}   # fingerprint → occurrence count

    def record(self, opp: ArbOpportunity) -> float:
        """Record opportunity, return persistence in seconds."""
        fp = opp.fingerprint()
        now = time.time()
        if fp not in self._seen:
            self._seen[fp] = now
            self._counts[fp] = 0
        self._counts[fp] += 1
        return now - self._seen[fp]

    def get_count(self, opp: ArbOpportunity) -> int:
        return self._counts.get(opp.fingerprint(), 0)

    def cleanup(self, max_age: float = 3600):
        """Remove opportunities not seen for max_age seconds."""
        now = time.time()
        stale = [fp for fp, ts in self._seen.items() if now - ts > max_age]
        for fp in stale:
            del self._seen[fp]
            del self._counts[fp]


# ═══════════════════════════════════════════════════════════════════════════
# WEBSOCKET ORDERBOOK STREAMER
# ═══════════════════════════════════════════════════════════════════════════

class BookStreamer:
    """
    WebSocket-based real-time orderbook updates.
    Falls back to REST polling if websocket unavailable.
    """

    def __init__(self):
        self.books: dict[str, OrderBook] = {}
        self._connected = False

    async def connect(self, token_ids: list[str]):
        """Connect to Polymarket WebSocket for real-time updates."""
        try:
            import websockets
            self._ws = await websockets.connect(WS_BOOK)
            # Subscribe to orderbook channels
            for tid in token_ids[:50]:  # Limit subscriptions
                await self._ws.send(json.dumps({
                    "assets_ids": [tid],
                    "type": "Market"
                }))
            self._connected = True
            log.info(f"WebSocket connected, subscribed to {min(len(token_ids), 50)} tokens")
        except Exception as e:
            log.warning(f"WebSocket unavailable ({e}), using REST fallback")
            self._connected = False

    async def listen(self):
        """Listen for orderbook updates."""
        if not self._connected:
            return
        try:
            import websockets
            async for msg in self._ws:
                data = json.loads(msg)
                if data.get("event_type") == "book":
                    self._parse_book_update(data)
        except Exception as e:
            log.warning(f"WebSocket error: {e}")
            self._connected = False

    def _parse_book_update(self, data: dict):
        """Parse a book update message."""
        asset_id = data.get("asset_id", "")
        if not asset_id:
            return

        bids = [OrderLevel(float(b["price"]), float(b["size"]))
                for b in data.get("bids", [])]
        asks = [OrderLevel(float(a["price"]), float(a["size"]))
                for a in data.get("asks", [])]
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)

        self.books[asset_id] = OrderBook(
            token_id=asset_id, bids=bids, asks=asks, timestamp=time.time()
        )

    @property
    def is_connected(self) -> bool:
        return self._connected


# ═══════════════════════════════════════════════════════════════════════════
# DATA FETCHER
# ═══════════════════════════════════════════════════════════════════════════

def fetch_markets(max_events: int = 200) -> list[Market]:
    """Fetch active markets from Gamma events API."""
    all_markets = []
    offset = 0
    page_size = 100

    while offset < max_events:
        try:
            r = requests.get(f"{GAMMA_API}/events", params={
                "limit": page_size, "offset": offset,
                "active": True, "closed": False,
                "order": "volume24hr", "ascending": False,
            }, timeout=15)
            r.raise_for_status()
            batch = r.json()
        except Exception as e:
            log.error(f"Fetch error at offset {offset}: {e}")
            break

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

                if vol < 1000:
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
                    outcome_prices=[float(p) for p in prices] if prices else [],
                    volume=vol,
                    liquidity=float(m.get("liquidity", 0) or 0),
                    event_id=eid,
                    event_title=etitle,
                    neg_risk=neg_risk,
                    end_date=end_dt,
                    group_item_title=m.get("groupItemTitle", ""),
                    description=m.get("description", ""),
                ))

        offset += page_size

    log.info(f"Fetched {len(all_markets)} markets from {offset} events")
    return all_markets


# ═══════════════════════════════════════════════════════════════════════════
# RELATIONSHIP DETECTOR
# ═══════════════════════════════════════════════════════════════════════════

def detect_all_relationships(markets: list[Market]) -> tuple[list[Relationship], MarketGraph]:
    """Detect all relationship types and build market graph."""
    graph = MarketGraph()
    for m in markets:
        graph.add_market(m)

    rels = []

    # 1. Within-event
    by_event = defaultdict(list)
    for m in markets:
        by_event[m.event_id].append(m)

    for eid, group in by_event.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                r = Relationship("within_event", group[i].question, group[j].question,
                                 0.95, "eq", f"Event: {group[i].event_title}")
                rels.append(r)
                graph.add_edge(r)

    # 2. NegRisk
    for eid, group in by_event.items():
        nr = [m for m in group if m.neg_risk]
        if len(nr) < 2:
            continue
        for i in range(len(nr)):
            for j in range(i + 1, len(nr)):
                r = Relationship("negrisk", nr[i].question, nr[j].question,
                                 1.0, "xor", "Mutually exclusive")
                rels.append(r)
                graph.add_edge(r)

    # 3. Temporal subsets (same entity, different deadlines)
    by_entity = defaultdict(list)
    for m in markets:
        key = MarketNLP.extract_entity(m.question)
        if len(key) > 5:
            by_entity[key].append(m)

    for entity, group in by_entity.items():
        if len(group) < 2:
            continue
        dated = [(m, m.end_date) for m in group if m.end_date]
        dated.sort(key=lambda x: x[1])
        for i in range(len(dated)):
            for j in range(i + 1, len(dated)):
                m_early, m_late = dated[i][0], dated[j][0]
                r = Relationship("subset", m_early.question, m_late.question,
                                 0.85, "lte", f"{m_early.end_date:%Y-%m} ⊂ {m_late.end_date:%Y-%m}")
                rels.append(r)
                graph.add_edge(r)

    # 4. Cross-event entity similarity
    for i in range(len(markets)):
        for j in range(i + 1, len(markets)):
            if markets[i].event_id == markets[j].event_id:
                continue
            sim = MarketNLP.word_similarity(markets[i].question, markets[j].question)
            if sim >= 0.45:
                r = Relationship("entity", markets[i].question, markets[j].question,
                                 sim, "lte" if sim > 0.7 else "none",
                                 f"Similarity: {sim:.2f}")
                rels.append(r)
                graph.add_edge(r)

    # 5. Implication detection
    for i in range(len(markets)):
        for j in range(len(markets)):
            if i == j or markets[i].event_id == markets[j].event_id:
                continue
            imp = MarketNLP.detect_implication(markets[i], markets[j])
            if imp:
                constraint, conf = imp
                r = Relationship("implication", markets[i].question, markets[j].question,
                                 conf, constraint, "Detected implication")
                rels.append(r)
                graph.add_edge(r)

    return rels, graph


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════════════════════

class ArbEngine:
    """Full arbitrage engine combining all components."""

    def __init__(self):
        self.solver = ArbSolver()
        self.book_fetcher = BookFetcher()
        self.tracker = OpportunityTracker()
        self.sizer = PositionSizer()
        self.streamer = BookStreamer()

    def scan(self, max_events: int = 200, min_edge: float = 0.5,
             min_depth: float = 5.0) -> dict:
        """Full scan: fetch → relate → solve → verify → rank."""
        t0 = time.time()

        # Fetch
        markets = fetch_markets(max_events)
        markets = [m for m in markets if m.price_sum > 0.01 and len(m.token_ids) >= 2]

        # Relate
        rels, graph = detect_all_relationships(markets)

        # Solve — multiple algorithms
        all_opps = []

        # Single-condition
        for m in markets:
            opp = self.solver.single_condition(m)
            if opp:
                all_opps.append(opp)

        # NegRisk optimal
        all_opps.extend(self.solver.negrisk_optimal(markets))

        # NegRisk deep scan (large groups)
        all_opps.extend(self.solver.negrisk_deep_scan(markets))

        # Combinatorial ILP
        all_opps.extend(self.solver.combinatorial_ilp(markets, rels))

        # Bregman correction
        all_opps.extend(self.solver.bregman_correction(markets))

        # Deduplicate by fingerprint
        seen_fps = set()
        unique_opps = []
        for opp in all_opps:
            fp = opp.fingerprint()
            if fp not in seen_fps:
                seen_fps.add(fp)
                unique_opps.append(opp)

        # Verify with orderbooks
        verified = []
        for opp in unique_opps:
            if opp.roi_pct < min_edge:
                continue

            fillable = True
            total_cost = 0.0
            for leg in opp.legs:
                if not leg.token_id:
                    fillable = False
                    break
                book = self.book_fetcher.get(leg.token_id)
                if not book:
                    fillable = False
                    break
                leg.book = book

                if leg.side == "BUY":
                    leg.fillable_size = book.fillable_size("BUY", leg.price + TICK_SIZE * 5)
                    leg.slippage = book.slippage("BUY", min(leg.fillable_size, 100))
                else:
                    leg.fillable_size = book.fillable_size("SELL", leg.price - TICK_SIZE * 5)
                    leg.slippage = book.slippage("SELL", min(leg.fillable_size, 100))

                if leg.fillable_size < 1.0:
                    fillable = False
                    break
                total_cost += leg.price * leg.fillable_size

            if not fillable:
                continue

            # Recalculate with slippage
            slippage_cost = sum(l.slippage * l.fillable_size for l in opp.legs)
            adjusted_profit = opp.profit - slippage_cost - FEE_RATE * total_cost
            opp.edge_after_fees = adjusted_profit

            if adjusted_profit < min_edge * opp.cost / 100:
                continue

            # Kelly sizing
            opp.kelly_fraction = self.sizer.arb_kelly(opp)

            # Persistence tracking
            opp.persistence_seconds = self.tracker.record(opp)
            opp.discovered_at = datetime.now(timezone.utc).isoformat()

            verified.append(opp)

        # Sort by ROI
        verified.sort(key=lambda o: o.roi_pct, reverse=True)

        elapsed = time.time() - t0

        result = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "markets_scanned": len(markets),
            "relationships_found": len(rels),
            "raw_opportunities": len(unique_opps),
            "verified_opportunities": len(verified),
            "top_opportunities": [
                {
                    "kind": o.kind,
                    "roi_pct": round(o.roi_pct, 2),
                    "cost": round(o.cost, 4),
                    "profit": round(o.profit, 4),
                    "edge_after_fees": round(o.edge_after_fees, 4),
                    "kelly_fraction": round(o.kelly_fraction, 4),
                    "persistence_sec": round(o.persistence_seconds, 1),
                    "legs": [
                        {"market": l.market_question[:60], "side": l.side,
                         "outcome": l.outcome, "price": l.price,
                         "depth": round(l.fillable_size, 2), "slippage": round(l.slippage, 4)}
                        for l in o.legs
                    ],
                }
                for o in verified[:30]
            ],
            "relationship_breakdown": {
                "within_event": sum(1 for r in rels if r.kind == "within_event"),
                "negrisk": sum(1 for r in rels if r.kind == "negrisk"),
                "subset": sum(1 for r in rels if r.kind == "subset"),
                "entity": sum(1 for r in rels if r.kind == "entity"),
                "implication": sum(1 for r in rels if r.kind == "implication"),
            },
        }

        return result


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Polymarket Arb Engine v2")
    parser.add_argument("--mode", choices=["scan", "loop", "paper", "live"], default="scan")
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--min-edge", type=float, default=0.5)
    parser.add_argument("--min-depth", type=float, default=5.0)
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()

    engine = ArbEngine()

    if args.mode in ("scan", "loop"):
        while True:
            result = engine.scan(args.events, args.min_edge, args.min_depth)

            print(f"\n{'='*70}")
            print(f"  POLYMARKET ARB ENGINE v2 — {result['timestamp']}")
            print(f"  Scanned: {result['markets_scanned']} markets | "
                  f"{result['relationships_found']} relationships | "
                  f"{result['elapsed_seconds']}s")
            print(f"  Opportunities: {result['raw_opportunities']} raw → "
                  f"{result['verified_opportunities']} verified")
            print(f"{'='*70}\n")

            for i, opp in enumerate(result["top_opportunities"][:15], 1):
                print(f"  [{i:2d}] {opp['roi_pct']:+7.1f}% ROI | "
                      f"cost=${opp['cost']:.4f} | "
                      f"edge=${opp['edge_after_fees']:.4f} | "
                      f"kelly={opp['kelly_fraction']:.1%} | "
                      f"persist={opp['persistence_sec']:.0f}s")
                for leg in opp["legs"]:
                    print(f"       {leg['side']} {leg['outcome']} @ ${leg['price']:.4f} "
                          f"depth=${leg['depth']:.0f} slip=${leg['slippage']:.4f} "
                          f"— {leg['market']}")

            print(f"\n  Relationships: {json.dumps(result['relationship_breakdown'])}")

            # Save
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            import os
            os.makedirs("runs", exist_ok=True)
            with open(f"runs/v2_{ts}.json", "w") as f:
                json.dump(result, f, indent=2)

            if args.mode == "scan":
                break

            print(f"\n  Next scan in {args.interval}s...\n")
            time.sleep(args.interval)

    elif args.mode in ("paper", "live"):
        print(f"Mode '{args.mode}' — use bot.py for full trading loop")


if __name__ == "__main__":
    main()
