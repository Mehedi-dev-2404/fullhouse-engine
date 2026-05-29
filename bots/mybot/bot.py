"""
Koda — Phase 4 bot for the Fullhouse Hackathon.
Position-aware preflop ranges + Monte Carlo postflop equity engine.
Phase 4 additions: 3bet logic, SPR awareness, steal logic, river bluffing.
"""

import os
import time
import random

import numpy as np
import eval7

BOT_NAME = "Koda"
BOT_AVATAR = "robot_1"

# Monte Carlo time budget (seconds). Reduced to 0.3 automatically if any
# action is observed to take > 1.5 s total.
_MC_BUDGET = 0.4

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RANK_ORDER = "23456789TJQKA"

# ---------------------------------------------------------------------------
# CFR Blueprint — loaded once at import time
# ---------------------------------------------------------------------------

_BLUEPRINT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "blueprint.npz"
)

# BLUEPRINT: (hand_tier, board_texture, street, pot_odds_bucket, position_bucket)
#            → np.ndarray shape (5,) of action probabilities
# BLUEPRINT_VISITS: same key → int visit count (used for confidence threshold)
BLUEPRINT: dict | None = None
BLUEPRINT_VISITS: dict = {}

# Only trust an info set that was visited this many times during training
_CFR_MIN_VISITS = 50

def _load_blueprint():
    global BLUEPRINT, BLUEPRINT_VISITS
    if not os.path.exists(_BLUEPRINT_PATH):
        BLUEPRINT = None
        return
    try:
        data = np.load(_BLUEPRINT_PATH, allow_pickle=False)
        bp = {}
        visits = {}
        for k, v in data.items():
            if k.startswith("_metadata"):
                continue
            if k.startswith("_visits_"):
                raw = k[len("_visits_"):]
                key = tuple(int(x) for x in raw.strip("()").split(","))
                visits[key] = int(v[0])
            else:
                key = tuple(int(x) for x in k.strip("()").split(","))
                bp[key] = v
        BLUEPRINT = bp
        BLUEPRINT_VISITS = visits
    except Exception:
        BLUEPRINT = None
        BLUEPRINT_VISITS = {}

_load_blueprint()

# ---------------------------------------------------------------------------
# CFR abstraction helpers (must match scripts/train_cfr.py exactly)
# ---------------------------------------------------------------------------

_CFR_TIER_LOOKUP: dict[str, int] = {}
for _h in ("AA", "KK", "QQ", "AKs"):                                    _CFR_TIER_LOOKUP[_h] = 8
for _h in ("JJ", "TT", "AQs", "AKo", "AJs"):                           _CFR_TIER_LOOKUP[_h] = 7
for _h in ("99", "88", "ATs", "AQo", "KQs"):                           _CFR_TIER_LOOKUP[_h] = 6
for _h in ("77", "66", "A9s", "A8s", "A7s", "KJs", "QJs", "AJo"):     _CFR_TIER_LOOKUP[_h] = 5
for _h in ("55", "44", "33", "22",
           "54s", "65s", "76s", "87s", "98s", "T9s", "JTs"):           _CFR_TIER_LOOKUP[_h] = 4
for _h in ("A6s", "A5s", "A4s", "A3s", "A2s",
           "K9s", "K8s", "K7s",
           "KJo", "KQo", "QJo", "JTo"):                                 _CFR_TIER_LOOKUP[_h] = 3
for _h in ("K6s", "K5s", "K4s", "K3s", "K2s",
           "Q8s", "Q7s", "Q6s",
           "J8s", "J7s", "J6s", "T8s"):                                 _CFR_TIER_LOOKUP[_h] = 2


def _cfr_hand_tier(hand_str: str) -> int:
    """Map canonical hand string (e.g. 'AKs') to tier 1-8."""
    return _CFR_TIER_LOOKUP.get(hand_str, 1)


def _cfr_board_texture(community_cards: list, street_int: int) -> int:
    """
    Map community card strings (e.g. ['Ah','Kd','2c']) to texture bucket 0-5.
    Matches board_texture_from_cards() in train_cfr.py.
    """
    if street_int == 0 or not community_cards:
        return 0
    ranks = [c[0] for c in community_cards]
    suits = [c[1] for c in community_cards]

    if len(ranks) != len(set(ranks)):
        return 5  # paired

    rank_idxs = sorted(RANK_ORDER.index(r) for r in ranks)
    suit_counts: dict[str, int] = {}
    for s in suits:
        suit_counts[s] = suit_counts.get(s, 0) + 1
    flush_draw    = max(suit_counts.values()) >= 3
    straight_draw = (rank_idxs[-1] - rank_idxs[0]) <= 4
    wet           = flush_draw or straight_draw
    high_board    = rank_idxs[-1] >= RANK_ORDER.index("T")

    if wet and high_board: return 4
    if wet:                return 3
    if high_board:         return 2
    return 1


def _cfr_position_bucket(position_label: str) -> int:
    """Map position label to bucket: 0=early, 1=middle, 2=late."""
    if position_label in ("UTG",):            return 0
    if position_label in ("MP", "HJ"):        return 1
    return 2  # CO, BTN, SB, BB → late


_STREET_INT = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def _cfr_pot_odds_bucket(amount_owed: float, pot: float) -> int:
    if amount_owed <= 0:         return 0
    ratio = amount_owed / (pot + amount_owed)
    if ratio < 0.15:             return 1
    if ratio < 0.33:             return 2
    if ratio < 0.66:             return 3
    return 4


def cfr_lookup(game_state) -> dict | None:
    """
    Look up current situation in the CFR blueprint and return a real action dict.
    Returns None if blueprint is unavailable or lookup fails.
    """
    if BLUEPRINT is None:
        return None
    try:
        street_str  = game_state["street"]
        street_int  = _STREET_INT.get(street_str, 0)
        amount_owed = game_state["amount_owed"]
        pot         = game_state["pot"]
        can_check   = game_state["can_check"]
        my_cards    = game_state["your_cards"]
        my_stack    = game_state["your_stack"]
        my_bet      = game_state["your_bet_this_street"]
        min_raise   = game_state["min_raise_to"]
        community   = game_state.get("community_cards", [])

        hand_str    = cards_to_hand_str(my_cards[0], my_cards[1])
        tier        = _cfr_hand_tier(hand_str)
        board_tex   = _cfr_board_texture(community, street_int)
        position    = get_position(game_state)
        pos_bkt     = _cfr_position_bucket(position)
        po_bkt      = _cfr_pot_odds_bucket(amount_owed, pot)

        key = (tier, board_tex, street_int, po_bkt, pos_bkt)
        if key not in BLUEPRINT:
            return None
        if BLUEPRINT_VISITS.get(key, 0) < _CFR_MIN_VISITS:
            return None  # not enough training data — fall back to heuristics

        strat = BLUEPRINT[key]
        # Determine legal abstract actions
        max_chips = my_stack + my_bet
        legal = [1]  # call/check always legal
        if amount_owed > 0:
            legal.append(0)  # fold
        if my_stack > amount_owed:
            legal += [2, 3, 4]  # raises

        # Sample from strategy restricted to legal actions
        weights = np.array([strat[a] for a in legal], dtype=np.float64)
        total   = weights.sum()
        if total <= 0:
            return None
        weights /= total
        chosen = legal[min(int(np.searchsorted(np.cumsum(weights), random.random())),
                          len(legal) - 1)]

        # Convert abstract action → real action dict
        if chosen == 0:
            return {"action": "fold"}
        if chosen == 1:
            return {"action": "check"} if can_check else {"action": "call"}
        if chosen == 4:
            return {"action": "all_in"}

        # Raise actions (2 = 33% pot, 3 = 75% pot)
        frac   = 0.33 if chosen == 2 else 0.75
        amount = int(amount_owed + frac * pot)
        amount = max(amount, min_raise)
        amount = min(amount, max_chips)
        if amount >= max_chips:
            return {"action": "all_in"}
        return {"action": "raise", "amount": amount}

    except Exception:
        return None

# ---------------------------------------------------------------------------
# Preflop open-raise ranges (hardcoded lookup tables, zero computation)
# Canonical hand form: "AA", "AKs", "AKo"
# Percentages are approximate: UTG 14%, MP 18-20%, HJ 24-25%, CO 30-34%, BTN 45-47%
# ---------------------------------------------------------------------------

_UTG = frozenset([
    # Pairs
    "AA", "KK", "QQ", "JJ", "TT", "99", "88",
    # Suited
    "AKs", "AQs", "AJs", "ATs", "A9s",
    "KQs", "KJs", "KTs",
    "QJs",
    "JTs",
    "T9s",
    # Offsuit
    "AKo", "AQo", "AJo",
    "KQo", "KJo",
])

_MP = frozenset(_UTG | {
    # Pairs
    "77",
    # Suited
    "A8s", "A7s",
    "K9s",
    "QTs", "Q9s",
    "J9s",
    "98s",
    # Offsuit
    "ATo",
    "QJo",
})

_HJ = frozenset(_MP | {
    # Pairs
    "66", "55",
    # Suited
    "A6s", "A5s",
    "K8s",
    "T8s",
    "87s", "76s",
    # Offsuit
    "KTo",
    "JTo",
})

_CO = frozenset(_HJ | {
    # Pairs
    "44", "33",
    # Suited
    "A4s", "A3s", "A2s",
    "K7s", "K6s",
    "J8s",
    "T7s",
    "65s", "54s",
    # Offsuit
    "K9o",
    "QTo",
    "J9o",
})

_BTN = frozenset(_CO | {
    # Pairs
    "22",
    # Suited
    "K5s", "K4s", "K3s", "K2s",
    "Q8s", "Q7s",
    "J7s",
    "97s", "86s", "75s", "64s", "53s", "43s",
    # Offsuit
    "K8o", "K7o",
    "Q9o",
    "J8o",
    "T9o", "T8o",
    "98o", "97o",
    "87o",
})

# Position -> open-raise range.
# SB and BB get CO/BTN-width when the action folds to them.
OPEN_RANGES = {
    "UTG": _UTG,
    "MP":  _MP,
    "HJ":  _HJ,
    "CO":  _CO,
    "BTN": _BTN,
    "SB":  _CO,   # folded-to SB plays like CO
    "BB":  _BTN,  # BB 3-bet or squeeze range (rarely open, but widest)
}

# ---------------------------------------------------------------------------
# Hand strength tiers for 3bet / call / steal decisions
# ---------------------------------------------------------------------------

# Premium hands: always 3bet when facing a raise
PREMIUM_HANDS = frozenset(["AA", "KK", "QQ", "AKs", "AKo"])

# Top ~15% — call a raise if not premium (UTG range)
_TOP_15 = _UTG

# Top ~40% — call vs maniac (CO + some BTN hands)
_TOP_40 = frozenset(_CO | {
    "22",
    "K5s", "K4s",
    "Q8s", "Q7s",
    "J7s",
    "86s", "75s",
    "K8o", "K7o",
    "Q9o",
    "J8o",
    "T9o",
})

# Top ~55% — steal range from BTN/CO/SB (BTN + extra hands)
_STEAL_55 = frozenset(_BTN | {
    "Q6s", "Q5s",
    "J6s", "J5s",
    "96s",
    "85s",
    "74s",
    "K6o", "K5o",
    "Q8o",
    "J7o",
    "T7o",
    "97o",
})

# ---------------------------------------------------------------------------
# Position label maps: offset from dealer seat -> label
# ---------------------------------------------------------------------------

_POS_LABELS = {
    2: {0: "BTN", 1: "BB"},
    3: {0: "BTN", 1: "SB", 2: "BB"},
    4: {0: "BTN", 1: "SB", 2: "BB", 3: "UTG"},
    5: {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "CO"},
    6: {0: "BTN", 1: "SB", 2: "BB", 3: "UTG", 4: "MP", 5: "CO"},
}

# Cache: hand_id -> position label
_pos_cache = {}


# ---------------------------------------------------------------------------
# Helper: canonical hand string
# ---------------------------------------------------------------------------

def cards_to_hand_str(card1, card2):
    """Return canonical hand string e.g. 'AKs', 'AKo', 'AA'."""
    r1, s1 = card1[0], card1[1]
    r2, s2 = card2[0], card2[1]
    # Put higher rank first
    if RANK_ORDER.index(r1) < RANK_ORDER.index(r2):
        r1, s1, r2, s2 = r2, s2, r1, s1
    if r1 == r2:
        return r1 + r2
    return r1 + r2 + ("s" if s1 == s2 else "o")


def hand_in_range(hand_str, position):
    """True if hand_str is in the open-raise range for position."""
    return hand_str in OPEN_RANGES.get(position, frozenset())


# ---------------------------------------------------------------------------
# Position engine
# ---------------------------------------------------------------------------

def get_position(game_state):
    """
    Infer our position label for this hand. Result is cached by hand_id.

    Algorithm:
      1. Find SB seat from action_log.
      2. Heads-up (n=2): dealer == SB seat.
         Otherwise: dealer is the seat immediately before SB in seat order.
      3. Compute our offset from dealer and look up the label.
    """
    hand_id = game_state["hand_id"]
    if hand_id in _pos_cache:
        return _pos_cache[hand_id]

    players = game_state["players"]
    active = [p for p in players if p["state"] != "busted"]
    n = len(active)
    active_seats = sorted(p["seat"] for p in active)

    my_seat = game_state["seat_to_act"]

    # Find SB from action_log
    sb_seat = None
    for entry in game_state.get("action_log", []):
        if entry.get("action") == "small_blind":
            sb_seat = entry["seat"]
            break

    if sb_seat is None or n < 2:
        label = "BTN"
        _pos_cache[hand_id] = label
        return label

    # Determine dealer seat
    if n == 2:
        # Heads-up: dealer IS the small blind
        dealer_seat = sb_seat
    else:
        sb_idx = active_seats.index(sb_seat) if sb_seat in active_seats else 0
        dealer_seat = active_seats[(sb_idx - 1) % n]

    # Offset of our seat from dealer in active-seat order
    dealer_idx = active_seats.index(dealer_seat) if dealer_seat in active_seats else 0
    my_idx = active_seats.index(my_seat) if my_seat in active_seats else 0
    offset = (my_idx - dealer_idx) % n

    label_map = _POS_LABELS.get(n, _POS_LABELS[6])
    label = label_map.get(offset, "UTG")

    _pos_cache[hand_id] = label
    return label


# ---------------------------------------------------------------------------
# BB finder
# ---------------------------------------------------------------------------

def _find_bb(game_state):
    """Extract big blind size from action_log; fallback to min_raise_to / 2."""
    for entry in game_state.get("action_log", []):
        if entry.get("action") == "big_blind":
            amt = entry.get("amount")
            if amt and amt > 0:
                return amt
    return max(game_state.get("min_raise_to", 200) // 2, 1)


# ---------------------------------------------------------------------------
# Postflop equity engine
# ---------------------------------------------------------------------------

_HAND_TYPE_MAP = {
    "Straight Flush": "straight_flush",
    "Quads":          "quads",
    "Full House":     "full_house",
    "Flush":          "flush",
    "Straight":       "straight",
    "Trips":          "trips",
    "Two Pair":       "two_pair",
    "Pair":           "pair",
    "High Card":      "high_card",
}

_ALL_CARDS = [r + s for r in "23456789TJQKA" for s in "cdhs"]


def estimate_equity(hole_cards, community_cards, num_opponents, time_budget=None):
    """
    Monte Carlo equity estimate vs. num_opponents random hands.
    Returns {"equity": float, "hand_category": str}.
    Hard-capped at time_budget seconds (defaults to _MC_BUDGET, well within 2s limit).
    """
    if time_budget is None:
        time_budget = _MC_BUDGET
    start = time.time()

    my_e7    = [eval7.Card(c) for c in hole_cards]
    board_e7 = [eval7.Card(c) for c in community_cards]
    known    = set(hole_cards) | set(community_cards)

    deck = [eval7.Card(c) for c in _ALL_CARDS if c not in known]

    cards_need_board = 5 - len(community_cards)
    cards_need_opps  = 2 * max(num_opponents, 1)
    total_need       = cards_need_board + cards_need_opps

    wins = ties = iterations = 0

    while time.time() - start < time_budget:
        if total_need > len(deck):
            break
        sample       = random.sample(deck, total_need)
        full_board   = board_e7 + sample[:cards_need_board]
        opp_start    = cards_need_board

        my_score  = eval7.evaluate(my_e7 + full_board)
        best_opp  = max(
            eval7.evaluate(sample[opp_start + i*2 : opp_start + i*2 + 2] + full_board)
            for i in range(max(num_opponents, 1))
        )

        if my_score > best_opp:
            wins += 1
        elif my_score == best_opp:
            ties += 1
        iterations += 1

    equity = (wins + 0.5 * ties) / iterations if iterations > 0 else 0.5

    # Hand category from current hole + board (eval7 needs ≥5 cards)
    if len(my_e7) + len(board_e7) >= 5:
        score        = eval7.evaluate(my_e7 + board_e7)
        hand_category = _HAND_TYPE_MAP.get(eval7.handtype(score), "high_card")
    else:
        hand_category = "high_card"

    return {"equity": equity, "hand_category": hand_category}


def get_pot_odds(amount_owed, pot):
    """Minimum equity needed to break even on a call."""
    total = pot + amount_owed
    return amount_owed / total if total > 0 else 0.0


def board_texture(community_cards):
    """
    Analyse the community cards and return texture flags.
    flush_draw  : 2+ cards same suit on flop; 3+ on turn/river
    straight_draw: 3+ cards within any 4-rank window
    paired      : any rank appears 2+ times
    monotone    : all cards the same suit
    """
    if not community_cards:
        return {"flush_draw": False, "straight_draw": False, "paired": False, "monotone": False}

    suits = [c[1] for c in community_cards]
    ranks = [c[0] for c in community_cards]
    n     = len(community_cards)

    suit_max   = max(suits.count(s) for s in set(suits))
    flush_draw = suit_max >= 2 if n <= 3 else suit_max >= 3
    monotone   = len(set(suits)) == 1

    paired = any(ranks.count(r) >= 2 for r in set(ranks))

    rank_nums = sorted(set(RANK_ORDER.index(r) for r in ranks))
    straight_draw = any(
        sum(1 for r in rank_nums if low <= r <= low + 3) >= 3
        for low in rank_nums
    )

    return {
        "flush_draw":   flush_draw,
        "straight_draw": straight_draw,
        "paired":       paired,
        "monotone":     monotone,
    }


# ---------------------------------------------------------------------------
# Opponent modelling
# ---------------------------------------------------------------------------

OPPONENT_STATS = {}  # bot_id → stats dict

_opp_processed   = 0    # match_action_log entries already processed
_opp_hand_pos    = {}   # hand_num → {bot_id: action_count_seen}
_opp_preflop_ctx = {}   # hand_num → preflop action count (populated by decide())
_opp_hands_seen  = {}   # bot_id → set of hand_nums observed


def _opp_ensure(bot_id):
    if bot_id not in OPPONENT_STATS:
        OPPONENT_STATS[bot_id] = {
            "hands": 0,
            "vpip": 0,
            "pfr": 0,
            "cbet_faced": 0,
            "cbet_folded": 0,
            "total_actions": 0,
            "aggressive_actions": 0,
        }


def update_opponent_model(match_action_log):
    """
    Process new match_action_log entries and update OPPONENT_STATS.

    Preflop detection relies on _opp_preflop_ctx[hand_num] which decide()
    sets (from game_state["action_log"]) before calling this function.
    """
    global _opp_processed
    new_entries = match_action_log[_opp_processed:]
    _opp_processed = len(match_action_log)

    for entry in new_entries:
        bot_id   = entry.get("bot_id", "")
        action   = entry.get("action", "")
        hand_num = entry.get("hand_num", -1)
        if not bot_id or not action:
            continue

        _opp_ensure(bot_id)
        stats = OPPONENT_STATS[bot_id]

        # Position of this entry within the hand for this bot (0-indexed)
        if hand_num not in _opp_hand_pos:
            _opp_hand_pos[hand_num] = {}
        pos = _opp_hand_pos[hand_num].get(bot_id, 0)
        _opp_hand_pos[hand_num][bot_id] = pos + 1

        # Count new hands observed per bot
        if bot_id not in _opp_hands_seen:
            _opp_hands_seen[bot_id] = set()
        if hand_num not in _opp_hands_seen[bot_id]:
            _opp_hands_seen[bot_id].add(hand_num)
            stats["hands"] += 1

        # Is this entry preflop? pos < preflop_count means it came before
        # we acted (so it's an earlier preflop action by this opponent).
        preflop_count = _opp_preflop_ctx.get(hand_num, 0)
        is_preflop = pos < preflop_count

        # Aggression stats (all streets)
        if action in ("fold", "call", "raise", "all_in", "check"):
            stats["total_actions"] += 1
        if action in ("raise", "all_in"):
            stats["aggressive_actions"] += 1

        # Preflop stats (VPIP / PFR) — only voluntary actions, not blinds
        if is_preflop:
            if action in ("call", "raise", "all_in"):
                stats["vpip"] += 1
            if action in ("raise", "all_in"):
                stats["pfr"] += 1


def classify_opponent(bot_id):
    """Return player type: nit / TAG / LAG / calling_station / maniac / unknown."""
    if bot_id not in OPPONENT_STATS:
        return "unknown"
    stats = OPPONENT_STATS[bot_id]
    hands = stats["hands"]
    if hands < 5:
        return "unknown"

    vpip_pct = stats["vpip"] / hands
    pfr_pct  = stats["pfr"]  / hands
    total    = stats["total_actions"]
    agg      = stats["aggressive_actions"] / total if total > 0 else 0.0

    if vpip_pct < 0.15:
        return "nit"
    if vpip_pct > 0.6 and agg > 0.5:
        return "maniac"
    if vpip_pct > 0.5 and pfr_pct < 0.1:
        return "calling_station"
    if vpip_pct > 0.4 and pfr_pct > 0.25:
        return "LAG"
    return "TAG"


def get_opponent_types(state):
    """Return {bot_id: classification} for all non-folded, non-busted opponents."""
    my_seat = state["seat_to_act"]
    result  = {}
    for p in state.get("players", []):
        if p["state"] not in ("folded", "busted") and p["seat"] != my_seat:
            bid = p.get("bot_id", "")
            if bid:
                result[bid] = classify_opponent(bid)
    return result


# ---------------------------------------------------------------------------
# Main decide function
# ---------------------------------------------------------------------------

def _decide_core(game_state):
    """Inner logic — see decide() wrapper for timing enforcement."""
    try:
        # ── CFR blueprint (GTO layer) ─────────────────────────────────────────
        # Try the blueprint first; fall through to heuristics on any miss.
        cfr_action = cfr_lookup(game_state)
        if cfr_action is not None:
            return cfr_action

        street      = game_state["street"]
        amount_owed = game_state["amount_owed"]
        pot         = game_state["pot"]
        can_check   = game_state["can_check"]
        my_cards    = game_state["your_cards"]
        min_raise   = game_state["min_raise_to"]
        my_stack    = game_state["your_stack"]
        my_bet      = game_state["your_bet_this_street"]

        # ── Opponent modelling ────────────────────────────────────────────────
        hand_id = game_state.get("hand_id", "")
        try:
            hand_num = int(hand_id.rsplit("_h", 1)[-1])
        except (ValueError, IndexError):
            hand_num = -1

        # Snapshot the number of preflop actions already in the engine log so
        # update_opponent_model() can correctly tag entries as preflop.
        if street == "preflop":
            _opp_preflop_ctx[hand_num] = sum(
                1 for e in game_state.get("action_log", [])
                if e.get("action") not in ("small_blind", "big_blind")
            )

        update_opponent_model(game_state.get("match_action_log", []))

        opp_types  = get_opponent_types(game_state)
        type_vals  = list(opp_types.values())
        has_maniac          = "maniac"          in type_vals
        has_lag             = "LAG"             in type_vals
        has_nit             = "nit"             in type_vals
        has_tag             = "TAG"             in type_vals
        has_calling_station = "calling_station" in type_vals

        # High fold-to-cbet pattern (mathematician / ref_bot_2)
        high_fold_to_cbet = any(
            OPPONENT_STATS.get(bid, {}).get("cbet_faced", 0) > 5
            and OPPONENT_STATS[bid]["cbet_folded"] / OPPONENT_STATS[bid]["cbet_faced"] > 0.8
            for bid in opp_types
        )

        # ── Preflop ──────────────────────────────────────────────────────────
        if street == "preflop":
            position = get_position(game_state)
            hand_str = cards_to_hand_str(my_cards[0], my_cards[1])
            bb       = _find_bb(game_state)

            # Detect if we are facing a raise (someone put in more than just the BB)
            facing_raise = (not can_check) and (amount_owed > bb)

            # ── Facing a raise: 3bet / call / fold logic ──────────────────────
            if facing_raise:
                max_chips = my_stack + my_bet

                # Premium hands → always 3bet to 3x their raise
                if hand_str in PREMIUM_HANDS:
                    threbet = max(int(3 * amount_owed), min_raise)
                    threbet = min(threbet, max_chips)
                    if threbet >= max_chips:
                        return {"action": "all_in"}
                    return {"action": "raise", "amount": threbet}

                # Top ~15% → call
                if hand_str in _TOP_15:
                    return {"action": "call"}

                # vs maniac: widen call range to top ~40%
                if has_maniac and hand_str in _TOP_40:
                    return {"action": "call"}

                # Everything else → fold (or check if free)
                if can_check:
                    return {"action": "check"}
                return {"action": "fold"}

            # ── Not facing a raise: open or steal ────────────────────────────
            in_range = hand_in_range(hand_str, position)

            # Steal logic: BTN/CO/SB with top 55% — exploits tight folders
            # (If already in standard range this fires naturally; the steal
            #  extension catches hands that are in _STEAL_55 but not in_range)
            if position in ("BTN", "CO", "SB") and hand_str in _STEAL_55:
                in_range = True

            if in_range:
                # vs nit: steal MORE (they fold to opens) — use full range
                # vs maniac/LAG: raise tighter to avoid bloating pot OOP
                if (has_maniac or has_lag) and hand_str not in _TOP_15 and position in ("UTG", "MP"):
                    # In early position vs aggressor, tighten to avoid 3bet bluffs
                    if can_check:
                        return {"action": "check"}
                    return {"action": "fold"}

                multiplier = 2.5 if position in ("BTN", "CO") else 3.0
                raise_to = int(multiplier * bb)
                raise_to = max(raise_to, min_raise)
                max_chips = my_stack + my_bet
                raise_to  = min(raise_to, max_chips)
                if raise_to >= max_chips:
                    return {"action": "all_in"}
                return {"action": "raise", "amount": raise_to}

            # vs maniac / LAG: call only with decent hands (top ~40%)
            # Don't call garbage just because an aggressor is at the table
            if (has_maniac or has_lag) and amount_owed > 0 and hand_str in _TOP_40:
                return {"action": "call"}

            if can_check:
                return {"action": "check"}
            return {"action": "fold"}

        # ── Postflop ─────────────────────────────────────────────────────────
        community = game_state.get("community_cards", [])
        my_seat   = game_state["seat_to_act"]
        players   = game_state.get("players", [])
        num_opps  = sum(
            1 for p in players
            if p["state"] not in ("folded", "busted") and p["seat"] != my_seat
        )
        num_opps = max(num_opps, 1)

        result   = estimate_equity(my_cards, community, num_opps)
        equity   = result["equity"]
        pot_odds = get_pot_odds(amount_owed, pot)
        texture  = board_texture(community)

        # ── Stack-to-pot ratio (SPR) awareness ───────────────────────────────
        spr = my_stack / pot if pot > 0 else 999.0

        # Scale all-in equity thresholds by number of opponents (multi-way = harder to win)
        _spr_threshold_short = 0.35 + 0.04 * (num_opps - 1)  # 0.35 HU → 0.55 4-way
        _spr_threshold_low   = 0.40 + 0.04 * (num_opps - 1)  # 0.40 HU → 0.60 4-way

        # Extremely short stack: go all_in whenever equity is decent
        if spr < 1 and equity > _spr_threshold_short:
            return {"action": "all_in"}

        # Low SPR: commit — all_in if equity justifies it
        if spr < 3 and equity > _spr_threshold_low:
            return {"action": "all_in"}

        # ── Opponent-type postflop adjustments ───────────────────────────────

        # vs nit / passive opponents (e.g. Shark folds to >25% pot, Template
        # never bets): always bet 33% pot when we have initiative.
        # 33% pot → owed/pot = 0.33 > 0.25 threshold → guaranteed fold.
        if has_nit and not has_calling_station and not has_maniac:
            if can_check or amount_owed == 0:
                bet = max(int(pot * 0.33), min_raise)
                bet = min(bet, my_stack + my_bet)
                if bet >= my_stack + my_bet:
                    return {"action": "all_in"}
                return {"action": "raise", "amount": bet}
            # Facing a bet from a nit — they likely have real strength; need equity
            if equity > pot_odds:
                return {"action": "call"}
            if can_check:
                return {"action": "check"}
            return {"action": "fold"}

        # vs calling_station / maniac: never bluff, value-bet only (equity > 0.55)
        if has_calling_station or has_maniac:
            if equity > 0.55 and (can_check or amount_owed < pot * 0.35):
                bet = max(int(pot * 0.65), min_raise)
                bet = min(bet, my_stack + my_bet)
                if bet >= my_stack + my_bet:
                    return {"action": "all_in"}
                return {"action": "raise", "amount": bet}
            if equity > pot_odds:
                return {"action": "call"}
            if can_check:
                return {"action": "check"}
            return {"action": "fold"}

        # vs high fold-to-cbet (mathematician / ref_bot_2): bet 40% pot every street
        if high_fold_to_cbet:
            if can_check or amount_owed == 0:
                bet = max(int(pot * 0.4), min_raise)
                bet = min(bet, my_stack + my_bet)
                if bet >= my_stack + my_bet:
                    return {"action": "all_in"}
                return {"action": "raise", "amount": bet}
            if equity > pot_odds:
                return {"action": "call"}
            if can_check:
                return {"action": "check"}
            return {"action": "fold"}

        # ── River bluff logic ─────────────────────────────────────────────────
        # Bluff on dry boards vs nit/TAG when we have a weak hand (air)
        if street == "river" and equity < 0.3:
            board_dry = not texture["flush_draw"] and not texture["straight_draw"]
            bluff_target = (has_nit or has_tag) and not has_calling_station and not has_maniac
            if board_dry and bluff_target and (can_check or amount_owed == 0):
                bet = max(int(pot * 0.5), min_raise)
                bet = min(bet, my_stack + my_bet)
                if bet >= my_stack + my_bet:
                    return {"action": "all_in"}
                return {"action": "raise", "amount": bet}

        # ── High SPR: play cautiously, need strong equity to raise ────────────
        raise_threshold = 0.6 if spr > 10 else 0.65

        # ── Default postflop logic ────────────────────────────────────────────
        # Strong hand → raise/bet
        if equity > raise_threshold and (can_check or amount_owed < pot * 0.35):
            bet = max(int(pot * 0.65), min_raise)
            bet = min(bet, my_stack + my_bet)
            if bet >= my_stack + my_bet:
                return {"action": "all_in"}
            return {"action": "raise", "amount": bet}

        # Profitable call
        if equity > pot_odds:
            return {"action": "call"}

        # Safe fallback: free check
        if can_check:
            return {"action": "check"}

        return {"action": "fold"}

    except Exception:
        return {"action": "fold"}


def decide(game_state):
    """
    Called once per action. Returns a valid action dict within 2 seconds.
    Measures wall time and tightens MC budget if any action exceeds 1.5 s.
    """
    global _MC_BUDGET
    _t0 = time.time()
    result = _decide_core(game_state)
    elapsed = time.time() - _t0
    if elapsed > 1.5 and _MC_BUDGET > 0.3:
        _MC_BUDGET = 0.3
    return result
