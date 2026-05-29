"""
CFR Blueprint Trainer for Koda.

Uses External Sampling CFR (Lanctot et al. 2009):
  - For the traversing player:  recurse all actions, accumulate regrets.
  - For the opponent:           sample ONE action from current strategy.

This keeps the recursion linear in depth rather than exponential in |A|^depth,
making each iteration O(|A| * depth) instead of O(|A|^depth).

Usage:
    python3 scripts/train_cfr.py              # full 500k iterations
    python3 scripts/train_cfr.py --iters 1000 # smoke test
"""

import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Abstractions
# ---------------------------------------------------------------------------

FOLD        = 0
CALL        = 1
RAISE_SMALL = 2   # 33% pot
RAISE_LARGE = 3   # 75% pot
ALL_IN      = 4
NUM_ACTIONS = 5

PREFLOP = 0
FLOP    = 1
TURN    = 2
RIVER   = 3

NUM_HAND_TIERS       = 8
NUM_BOARD_TEXTURES   = 6
NUM_STREETS          = 4
NUM_POT_ODDS_BUCKETS = 5
NUM_POSITION_BUCKETS = 3

STARTING_STACK = 10_000
BB = 100
SB = 50

# ---------------------------------------------------------------------------
# Hand tier classification (1–8)
# ---------------------------------------------------------------------------

RANK_ORDER = "23456789TJQKA"

_TIER8 = frozenset(["AA", "KK", "QQ", "AKs"])
_TIER7 = frozenset(["JJ", "TT", "AQs", "AKo", "AJs"])
_TIER6 = frozenset(["99", "88", "ATs", "AQo", "KQs"])
_TIER5 = frozenset(["77", "66", "A9s", "A8s", "A7s", "KJs", "QJs", "AJo"])
_TIER4 = frozenset([
    "55", "44", "33", "22",
    "54s", "65s", "76s", "87s", "98s", "T9s", "JTs",
])
_TIER3 = frozenset([
    "A6s", "A5s", "A4s", "A3s", "A2s",
    "K9s", "K8s", "K7s",
    "KJo", "KQo", "QJo", "JTo",
])
_TIER2 = frozenset([
    "K6s", "K5s", "K4s", "K3s", "K2s",
    "Q8s", "Q7s", "Q6s",
    "J8s", "J7s", "J6s",
    "T8s",
])

_TIER_LOOKUP: dict[str, int] = {}
for _h in _TIER8: _TIER_LOOKUP[_h] = 8
for _h in _TIER7: _TIER_LOOKUP[_h] = 7
for _h in _TIER6: _TIER_LOOKUP[_h] = 6
for _h in _TIER5: _TIER_LOOKUP[_h] = 5
for _h in _TIER4: _TIER_LOOKUP[_h] = 4
for _h in _TIER3: _TIER_LOOKUP[_h] = 3
for _h in _TIER2: _TIER_LOOKUP[_h] = 2


def hand_tier(rank1: str, rank2: str, suited: bool) -> int:
    r1, r2 = rank1, rank2
    if RANK_ORDER.index(r1) < RANK_ORDER.index(r2):
        r1, r2 = r2, r1
    if r1 == r2:
        key = r1 + r2
    elif suited:
        key = r1 + r2 + "s"
    else:
        key = r1 + r2 + "o"
    return _TIER_LOOKUP.get(key, 1)


def sample_hand_tier() -> int:
    ranks = list(RANK_ORDER)
    r1, r2 = random.choices(ranks, k=2)
    suited = (r1 != r2) and random.random() < 0.235
    return hand_tier(r1, r2, suited)


# ---------------------------------------------------------------------------
# Board texture classification (0–5)
# ---------------------------------------------------------------------------

_SUITS = ("h", "d", "c", "s")


def board_texture_from_cards(cards: list) -> int:
    """cards: list of (rank_str, suit_str). Returns texture 1–5."""
    if not cards:
        return 0
    ranks = [c[0] for c in cards]
    suits = [c[1] for c in cards]

    # Paired board
    if len(ranks) != len(set(ranks)):
        return 5

    rank_idxs = sorted(RANK_ORDER.index(r) for r in ranks)
    high = rank_idxs[-1]

    suit_counts: dict[str, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    flush_draw = max(suit_counts.values()) >= 3
    straight_draw = (rank_idxs[-1] - rank_idxs[0]) <= 4

    wet  = flush_draw or straight_draw
    high_board = high >= RANK_ORDER.index("T")

    if wet and high_board:  return 4
    if wet:                 return 3
    if high_board:          return 2
    return 1


def sample_board_texture(street: int) -> int:
    if street == PREFLOP:
        return 0
    n = {FLOP: 3, TURN: 4, RIVER: 5}[street]
    cards = []
    used: set = set()
    while len(cards) < n:
        r = random.choice(RANK_ORDER)
        s = random.choice(_SUITS)
        if (r, s) not in used:
            used.add((r, s))
            cards.append((r, s))
    return board_texture_from_cards(cards)


# ---------------------------------------------------------------------------
# Pot-odds bucket
# ---------------------------------------------------------------------------

def pot_odds_bucket(to_call: float, pot: float) -> int:
    if to_call <= 0:
        return 0
    ratio = to_call / (pot + to_call)
    if ratio < 0.15: return 1
    if ratio < 0.33: return 2
    if ratio < 0.66: return 3
    return 4


# ---------------------------------------------------------------------------
# Game state for CFR traversal
# ---------------------------------------------------------------------------

# Maximum raises per street to bound tree depth
MAX_RAISES_PER_STREET = 2

def _legal_actions(to_call: float, stack: float, raise_count: int) -> list:
    actions = []
    if to_call > 0:
        actions.append(FOLD)
    actions.append(CALL)
    if raise_count < MAX_RAISES_PER_STREET and stack > to_call:
        raise_base = to_call + 1  # ensure we can raise
        if stack >= raise_base:
            actions.append(RAISE_SMALL)
            actions.append(RAISE_LARGE)
            actions.append(ALL_IN)
    return actions


def _raise_amount(action: int, pot: float, to_call: float, stack: float) -> float:
    """Return TOTAL chips the player puts in for a raise action."""
    if action == RAISE_SMALL:
        total = to_call + 0.33 * pot
    elif action == RAISE_LARGE:
        total = to_call + 0.75 * pot
    elif action == ALL_IN:
        total = stack
    else:
        return 0.0
    return min(total, stack)


def _call_amount(to_call: float, stack: float) -> float:
    return min(to_call, stack)


# ---------------------------------------------------------------------------
# CFR Trainer
# ---------------------------------------------------------------------------

class CFRTrainer:
    def __init__(self):
        self.regret_sum:   dict[tuple, np.ndarray] = {}
        self.strategy_sum: dict[tuple, np.ndarray] = {}
        self.iterations_done = 0

    # ---- Info-set helpers --------------------------------------------------

    def _ensure(self, key: tuple):
        if key not in self.regret_sum:
            self.regret_sum[key]   = np.zeros(NUM_ACTIONS, dtype=np.float64)
            self.strategy_sum[key] = np.zeros(NUM_ACTIONS, dtype=np.float64)

    def get_strategy(self, key: tuple, legal: list) -> np.ndarray:
        self._ensure(key)
        pos = np.maximum(self.regret_sum[key], 0.0)
        total = pos.sum()
        if total > 0:
            return pos / total
        strat = np.zeros(NUM_ACTIONS, dtype=np.float64)
        for a in legal:
            strat[a] = 1.0 / len(legal)
        return strat

    def get_average_strategy(self, key: tuple) -> np.ndarray:
        self._ensure(key)
        s = self.strategy_sum[key]
        total = s.sum()
        if total > 0:
            return s / total
        return np.full(NUM_ACTIONS, 1.0 / NUM_ACTIONS, dtype=np.float64)

    # ---- External Sampling CFR --------------------------------------------

    def _cfr(
        self,
        traverser:   int,     # 0 or 1 — which player we're computing for
        tiers:       tuple,   # (tier_p0, tier_p1)
        board_texs:  list,    # board texture per street (sampled once per hand)
        positions:   tuple,   # (pos_p0, pos_p1)
        street:      int,
        pot:         float,
        stacks:      list,    # [stack_p0, stack_p1]
        to_call:     float,   # chips the acting player must call
        actor:       int,     # 0 or 1 — whose turn
        raise_count: int,     # raises so far this street
        depth:       int,     # safety depth limit
    ) -> float:
        """Returns EV for the traverser."""

        MAX_DEPTH = 24
        if depth >= MAX_DEPTH:
            # Give up: treat as check-down / showdown
            return self._showdown_ev(traverser, tiers, pot)

        stack_actor = stacks[actor]

        # ---- Terminal: one player is all-in or out of chips ----
        if stack_actor <= 0:
            return self._showdown_ev(traverser, tiers, pot)

        legal = _legal_actions(to_call, stack_actor, raise_count)

        # Info-set key for the acting player
        tier_actor = tiers[actor]
        pos_actor  = positions[actor]
        po_bkt     = pot_odds_bucket(to_call, pot)
        board_tex  = board_texs[street]
        key = (tier_actor, board_tex, street, po_bkt, pos_actor)

        strategy = self.get_strategy(key, legal)

        # ---- External sampling branch ----
        if actor == traverser:
            # Traverse ALL actions; accumulate regrets
            action_evs = {}
            for action in legal:
                ev = self._apply_action(
                    action, traverser, tiers, board_texs, positions,
                    street, pot, stacks, to_call, actor, raise_count, depth, strategy
                )
                action_evs[action] = ev

            node_ev = sum(strategy[a] * action_evs[a] for a in legal)

            # Update regrets (no reach weight needed for traverser in external sampling)
            self._ensure(key)
            for action in legal:
                self.regret_sum[key][action] += action_evs[action] - node_ev
                self.strategy_sum[key][action] += strategy[action]

            return node_ev

        else:
            # Opponent: sample ONE action
            probs = np.array([strategy[a] for a in legal], dtype=np.float64)
            total = probs.sum()
            if total > 0:
                probs /= total
            else:
                probs = np.ones(len(legal), dtype=np.float64) / len(legal)
            sampled = legal[min(np.searchsorted(np.cumsum(probs), random.random()), len(legal) - 1)]

            # Still update opponent strategy sum
            self._ensure(key)
            for action in legal:
                self.strategy_sum[key][action] += strategy[action]

            return self._apply_action(
                sampled, traverser, tiers, board_texs, positions,
                street, pot, stacks, to_call, actor, raise_count, depth, strategy
            )

    def _apply_action(
        self, action, traverser, tiers, board_texs, positions,
        street, pot, stacks, to_call, actor, raise_count, depth, strategy
    ) -> float:
        """Apply one action and recurse. Returns EV for traverser."""
        opp = 1 - actor
        new_stacks = list(stacks)

        if action == FOLD:
            # Actor folds: pot goes to opponent
            # EV for traverser = +pot/2 if traverser is opp, else -pot/2
            # (pot/2 because each player started with roughly pot/2 in)
            # More precisely: traverser wins the pot if they are the opponent
            if traverser == opp:
                return pot / 2      # we win what's in the middle
            else:
                return -pot / 2     # we lose what we put in

        if action == CALL:
            amount = _call_amount(to_call, stacks[actor])
            new_stacks[actor] -= amount
            new_pot = pot + amount

            # After a call: if street < RIVER, advance; if RIVER, showdown
            if street == RIVER:
                return self._showdown_ev(traverser, tiers, new_pot)
            else:
                next_street = street + 1
                return self._cfr(
                    traverser, tiers, board_texs, positions,
                    next_street, new_pot, new_stacks,
                    to_call=0.0, actor=0,  # p0 acts first on new street
                    raise_count=0, depth=depth + 1
                )

        # Raise actions
        amount = _raise_amount(action, pot, to_call, stacks[actor])
        new_stacks[actor] -= amount
        new_pot = pot + amount
        new_to_call = amount - to_call  # extra chips opponent must call
        new_raise_count = raise_count + 1

        return self._cfr(
            traverser, tiers, board_texs, positions,
            street, new_pot, new_stacks,
            to_call=new_to_call, actor=opp,
            raise_count=new_raise_count, depth=depth + 1
        )

    def _showdown_ev(self, traverser: int, tiers: tuple, pot: float) -> float:
        """Higher tier wins pot; tie splits. Returns EV for traverser."""
        t0, t1 = tiers
        if t0 == t1:
            return 0.0
        traverser_tier = tiers[traverser]
        opp_tier       = tiers[1 - traverser]
        if traverser_tier > opp_tier:
            return pot / 2
        return -pot / 2

    # ---- Training loop ----------------------------------------------------

    def train(self, iterations: int = 500_000):
        start = time.time()
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
        os.makedirs(data_dir, exist_ok=True)

        for i in range(1, iterations + 1):
            tier_p0 = sample_hand_tier()
            tier_p1 = sample_hand_tier()
            tiers   = (tier_p0, tier_p1)

            # Pre-sample board textures for all streets
            board_texs = [0]  # preflop
            for st in (FLOP, TURN, RIVER):
                board_texs.append(sample_board_texture(st))

            pos_p0  = random.randint(0, 2)
            pos_p1  = random.choice([p for p in range(3) if p != pos_p0])
            positions = (pos_p0, pos_p1)

            stacks = [STARTING_STACK - BB, STARTING_STACK - SB]
            pot    = BB + SB

            # Alternate traverser each iteration for balanced training
            traverser = i % 2

            self._cfr(
                traverser, tiers, board_texs, positions,
                street=PREFLOP, pot=pot, stacks=stacks,
                to_call=BB - SB,  # SB faces BB to call
                actor=0,          # p0 = SB acts first preflop
                raise_count=0, depth=0
            )

            if i % 50_000 == 0:
                elapsed = time.time() - start
                rate = i / elapsed
                print(f"  iter {i:>7,} | info sets: {len(self.regret_sum):>5,} | "
                      f"{rate:,.0f} iter/s | elapsed: {elapsed:.1f}s",
                      flush=True)

            if i % 100_000 == 0:
                ckpt = os.path.join(data_dir, f"blueprint_ckpt_{i}.npz")
                self._save(ckpt, i)
                print(f"  [checkpoint: {ckpt}]", flush=True)

        self.iterations_done = iterations
        elapsed = time.time() - start
        print(f"\nDone. {iterations:,} iters in {elapsed:.1f}s "
              f"({iterations/elapsed:,.0f} iter/s)")

    # ---- Persistence -------------------------------------------------------

    def _save(self, path: str, iters: int):
        arrays: dict[str, np.ndarray] = {}
        for key in self.strategy_sum:
            arrays[str(key)] = self.get_average_strategy(key)
            # Visit count = sum of strategy_sum (adds ~1.0 per traversal visit)
            arrays["_visits_" + str(key)] = np.array([self.strategy_sum[key].sum()])
        arrays["_metadata_iterations"] = np.array([iters])
        arrays["_metadata_timestamp"]  = np.array([time.time()])
        np.savez_compressed(path, **arrays)

    def save_blueprint(self, path: str = None):
        if path is None:
            path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "..", "data", "blueprint.npz"
            )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._save(path, self.iterations_done)
        size_kb = os.path.getsize(path) / 1024
        print(f"\nBlueprint saved → {path}")
        print(f"  Size:       {size_kb:.1f} KB")
        print(f"  Info sets:  {len(self.regret_sum)}")
        print(f"  Iterations: {self.iterations_done:,}")
        print(f"  Timestamp:  {datetime.fromtimestamp(time.time())}")


# ---------------------------------------------------------------------------
# Blueprint loader (used by bot at runtime)
# ---------------------------------------------------------------------------

def load_blueprint(path: str = None) -> dict:
    """
    Returns dict: (hand_tier, board_texture, street, pot_odds_bucket, position_bucket)
                  → np.ndarray shape (5,) — average strategy probabilities.
    """
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "data", "blueprint.npz"
        )
    if not os.path.exists(path):
        return {}
    data = np.load(path, allow_pickle=False)
    blueprint = {}
    for k, v in data.items():
        if k.startswith("_metadata"):
            continue
        try:
            key = tuple(int(x) for x in k.strip("()").split(","))
            blueprint[key] = v
        except Exception:
            pass
    return blueprint


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters",  type=int, default=500_000)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    max_sets = (NUM_HAND_TIERS * NUM_BOARD_TEXTURES *
                NUM_STREETS * NUM_POT_ODDS_BUCKETS * NUM_POSITION_BUCKETS)
    print(f"CFR Blueprint Trainer — Koda")
    print(f"Algorithm:   External Sampling CFR")
    print(f"Iterations:  {args.iters:,}")
    print(f"Max info sets: {max_sets:,}  (8 tiers × 6 textures × 4 streets × "
          f"5 pot-odds × 3 positions)")
    print()

    trainer = CFRTrainer()
    trainer.train(args.iters)
    trainer.save_blueprint(args.output)

    # Sample strategies
    print("\nSample average strategies  [fold | call | raise_sm | raise_lg | all_in]:")
    samples = [
        (8, 0, PREFLOP, 1, 2, "premium preflop late, small odds"),
        (1, 0, PREFLOP, 3, 0, "trash preflop early, big odds"),
        (6, 2, FLOP,   2, 2, "good hand dry-high flop, med odds, late"),
        (3, 4, RIVER,  4, 0, "marginal wet-high river, huge odds, early"),
    ]
    for *key_parts, label in samples:
        key = tuple(key_parts)
        strat = trainer.get_average_strategy(key)
        print(f"  {label}")
        print(f"    key={key}  → {np.round(strat, 3)}")


if __name__ == "__main__":
    sys.setrecursionlimit(5000)
    main()
