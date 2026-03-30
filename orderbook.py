#!/usr/bin/env python3
"""
Polymarket Orderbook Fetcher
Fetches and caches orderbooks for markets, calculates actual fillable depth.
"""

import time
import requests
from dataclasses import dataclass
from typing import Optional
from collections import OrderedDict

CLOB_API = "https://clob.polymarket.com"
RATE_LIMIT_DELAY = 0.15  # ~7 req/sec safe limit


@dataclass
class OrderLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    token_id: str
    bids: list[OrderLevel]  # descending by price
    asks: list[OrderLevel]  # ascending by price
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

    @property
    def mid(self) -> float:
        return (self.best_bid + self.best_ask) / 2

    def depth_for_buy(self, max_price: float) -> tuple[float, float]:
        """Returns (total_size, avg_price) available for buying up to max_price."""
        total_cost = 0.0
        total_size = 0.0
        for ask in self.asks:
            if ask.price > max_price:
                break
            fill = min(ask.size, (max_price * total_size + ask.price * ask.size) / (total_size + ask.size)
                       if total_size > 0 else ask.size)
            total_cost += ask.price * ask.size
            total_size += ask.size
        avg = total_cost / total_size if total_size > 0 else 0
        return total_size, avg

    def depth_for_sell(self, min_price: float) -> tuple[float, float]:
        """Returns (total_size, avg_price) available for selling down to min_price."""
        total_proceeds = 0.0
        total_size = 0.0
        for bid in self.bids:
            if bid.price < min_price:
                break
            total_proceeds += bid.price * bid.size
            total_size += bid.size
        avg = total_proceeds / total_size if total_size > 0 else 0
        return total_size, avg

    def available_size_at_price(self, side: str, price: float, tick: float = 0.01) -> float:
        """How much can we fill at a given price?"""
        if side == "BUY":
            total = sum(o.size for o in self.asks if o.price <= price + tick)
            return total
        else:
            total = sum(o.size for o in self.bids if o.price >= price - tick)
            return total


class OrderBookCache:
    """LRU cache for orderbooks with TTL."""

    def __init__(self, ttl: float = 30.0, max_size: int = 1000):
        self.ttl = ttl
        self.max_size = max_size
        self._cache: OrderedDict[str, OrderBook] = {}
        self._last_fetch: dict[str, float] = {}

    def get(self, token_id: str) -> Optional[OrderBook]:
        if token_id in self._cache:
            age = time.time() - self._cache[token_id].timestamp
            if age < self.ttl:
                return self._cache[token_id]
        return None

    def put(self, book: OrderBook):
        if len(self._cache) >= self.max_size:
            self._cache.popitem(last=False)
        self._cache[book.token_id] = book


class BookFetcher:
    """Fetches orderbooks from Polymarket CLOB with rate limiting."""

    def __init__(self):
        self.cache = OrderBookCache(ttl=30)
        self._last_request = 0.0

    def _rate_limit(self):
        elapsed = time.time() - self._last_request
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self._last_request = time.time()

    def get_book(self, token_id: str) -> Optional[OrderBook]:
        """Fetch single orderbook, using cache when fresh."""
        cached = self.cache.get(token_id)
        if cached:
            return cached

        self._rate_limit()
        try:
            r = requests.get(f"{CLOB_API}/book", params={"token_id": token_id}, timeout=10)
            r.raise_for_status()
            data = r.json()

            bids = [OrderLevel(price=float(b["price"]), size=float(b["size"]))
                    for b in data.get("bids", [])]
            asks = [OrderLevel(price=float(a["price"]), size=float(a["size"]))
                    for a in data.get("asks", [])]

            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)

            book = OrderBook(
                token_id=token_id,
                bids=bids,
                asks=asks,
                timestamp=time.time(),
            )
            self.cache.put(book)
            return book
        except Exception as e:
            return None

    def get_books_batch(self, token_ids: list[str]) -> dict[str, OrderBook]:
        """Fetch multiple books. Uses single endpoint per token (no batch API available)."""
        results = {}
        for tid in token_ids:
            book = self.get_book(tid)
            if book:
                results[tid] = book
        return results

    def get_market_books(self, token_ids: list[str]) -> dict[str, OrderBook]:
        """Convenience: fetch YES and NO books for a market."""
        return self.get_books_batch(token_ids)
