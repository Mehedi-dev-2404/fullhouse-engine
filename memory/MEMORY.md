# Fullhouse Hackathon — Koda Bot Memory

## Phase 5 Final State (completed)

### Bot file: `bots/mybot/bot.py`
- Validation: PASSED
- All 5 matchups: PASSED

### Phase 5 matchup results (400 hands each)
- vs shark:         +8,650  ✅ (target was +5,000; Phase 4 was +1,500)
- vs aggressor:     +10,000 ✅
- vs mathematician: +10,000 ✅
- vs ref_bot_2:     +10,000 ✅
- vs template:      +10,000 ✅

### Demo tournament result (3-round Swiss, 6-player tables)
- #1: Koda +51,150 (dominant across all 3 rounds)
- #2: Template Bot A +2,150

---
## Key architectural decisions

### Preflop ranges
- Position-aware: UTG(14%) < MP < HJ < CO < BTN(47%)
- Steal range _STEAL_55 from BTN/CO/SB (widens to ~55%)
- vs nit: steal MORE (removed old nit-tightening that was backwards)
- vs maniac/LAG in UTG/MP: tighten to _TOP_15 only (avoid 3bet bluffs)
- vs maniac (facing non-open raise): call only if hand in _TOP_40 (not garbage)

### Postflop opponent exploit chain
1. SPR < 1: all_in if equity > 0.35 + 0.04*(num_opps-1)  [scaled for multi-way]
2. SPR < 3: all_in if equity > 0.40 + 0.04*(num_opps-1)
3. vs nit (Shark/Template): always bet 33% pot — both fold to >25% pot
4. vs calling_station/maniac: value-bet only with equity > 0.55
5. high_fold_to_cbet: bet 40% pot (note: counters never actually update — dead code)
6. River bluff on dry boards vs nit/TAG with equity < 0.3
7. Default: bet 65% pot if equity > 0.6-0.65 threshold

### Timing
- `_MC_BUDGET = 0.4s` global (auto-reduces to 0.3s if any action > 1.5s)
- MC runs 400-900+ iterations per postflop decision

### Opponent model
- Tracks: VPIP, PFR, aggression ratio per bot_id
- Classify: nit(<15% VPIP) / TAG / LAG / calling_station / maniac
- `high_fold_to_cbet` tracking (cbet_faced/cbet_folded) has dead counters — never fires

### Demo config
- `demo.py` BOT_PATHS updated to include Koda (replaced "Template Bot C" duplicate)

---
## Opponent analysis
- **Shark**: tight (VPIP ~14%), folds postflop to >25% pot → nit exploit
- **Template**: VPIP ~1% (only AA/KK), folds to >25% pot → same nit exploit
- **Aggressor**: raises 70% randomly → classified as maniac, beat by value-only play
- **Mathematician**: folds if owed > 25% pot (3:1 pot odds) → 65% pot bets work
- **Ref_bot_2**: identical to Mathematician
