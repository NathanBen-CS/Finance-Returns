"""
tier0_config.py -- every setting the Tier 0 scripts share, in one place.

Paths are built from this file's own location, so the scripts work no matter
which folder you run them from.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data_phase2" / "processed"
COMPANIES_CSV = DATA_DIR / "companies.csv"
EVENTS_CSV = DATA_DIR / "events.csv"
MARKET_REACTIONS_CSV = DATA_DIR / "market_reactions.csv"
EVENTS_PKL = DATA_DIR / "tier0_events.pkl"        # written by prepare, read by train

# ---------------------------------------------------------------- history
MAX_HISTORY_LEN = 128      # the model sees at most this many of a company's most recent past events

# ---------------------------------------------------------------- leakage rule
# THE QUESTION: when the model predicts the reaction to filing B, which of the
# company's earlier filings may it see the MARKET REACTION of?
#
# Filing B's own reaction, CAR[-1,+1], is measured over B's trading days -1, 0, +1.
# So the model must not see any price move from B's day -1 onward: those days ARE
# the answer. An earlier filing A's reaction is fair game only if it was fully
# measured BEFORE B's day -1.
#
# Count the gap in TRADING days (weekends and holidays don't count) from A's day 0
# to B's day 0. Then A's day k is B's day (k - gap). A's value is known in time if
# the last day it uses is before B's day -1:
#
#   value                          last day it uses   known if (last - gap) < -1
#   abnormal_return, volume change    A's day 0        0 - gap < -1  ->  gap >= 2
#   CAR[-1,+1], reaction_category     A's day +1       1 - gap < -1  ->  gap >= 3
#   CAR[-1,+5]                        A's day +5       5 - gap < -1  ->  gap >= 7
#   (reaction_category is just CAR[-1,+1] put into buckets, so it follows CAR[-1,+1])
#
# Worked example: A on Monday (gap counted to B's day 0)
#   B on Wednesday (gap 2): A's day 0 = Mon is before B's day -1 = Tue -> A's abnormal
#       return is shown; A's CAR[-1,+1] needs Tue, which IS B's day -1 -> hidden.
#   B the following Monday (gap 5): A's CAR[-1,+1] is shown; A's CAR[-1,+5] runs
#       to the following Monday -> hidden until gap >= 7.
#   B on the same day as A (gap 0): every reaction value of A is hidden -- they
#       share one price move, so A's CAR would literally be B's answer.
#
# Hidden values are set to 0 and come with a *_known = 0 flag, so the model can tell
# "not knowable yet" apart from "the move was genuinely zero".
MIN_GAP_DAY0 = 2
MIN_GAP_CAR1 = 3
MIN_GAP_CAR5 = 7

# ---------------------------------------------------------------- time features (ablation)
# Every past event always carries log(1 + trading days ago): a plain "how recent".
# USE_TIME_CYCLES adds sin/cos of trading-days-ago at each period below. These let
# attention pick up cyclical timing (e.g. "the same point in the last quarterly cycle")
# directly instead of having to build it out of the single recency number.
# Run once with False and once with True and compare: whether the model learns
# sensible time behaviour is part of the research question.
USE_TIME_CYCLES = True
TIME_CYCLE_PERIODS = (5, 21, 63, 252)   # trading days: week, month, quarter (earnings rhythm), year

# ---------------------------------------------------------------- how the history is summarised (A/B test)
# False: no causal mask. Every position attends to every position (the whole history
#        at once) and the SUMMARY TOKEN at position 0 is read out -- like BERT's [CLS].
# True:  causal mask. Each position attends only to itself and earlier positions (an
#        event can't see what came after it, like a model reading left to right), and
#        the LAST REAL POSITION is read out (the most recent past event, which has seen
#        everything before it). The token at position 0 is then only a start token: it
#        is read out only when the history is empty, so attention always has one valid
#        position.
# Leakage is identical either way -- the history only ever contains past events.
# A/B test it like the others: same SEED, data and settings, only this flag changed.
CAUSAL_READOUT = True

# ---------------------------------------------------------------- input scaling
CLIP_RETURNS = 0.50        # past returns/CARs are clipped to +/-50% before use as INPUTS (targets stay raw)

# ---------------------------------------------------------------- look-ahead switch
# The company profile (market cap, volatility, tiers) is one October-2026 snapshot
# attached to events back to 2016. trailing_volatility_1y is the most circular:
# for test-period events it is partly computed from those events' own returns.
# True = drop volatility_tier and trailing_volatility_1y from the inputs.
# Run once each way; if results barely move, the look-ahead is not driving them.
DROP_LOOKAHEAD_FEATURES = False

# ---------------------------------------------------------------- training
TRAIN_END = "2025-01-01"   # events before this date train the model
VAL_END = "2025-07-01"     # events before this (and after TRAIN_END) are validation; the rest is test
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
REACTION_LOSS_WEIGHT = 0.3 # weight of the secondary task (reaction_category) next to the CAR loss
TASK = "direction"        # "regression": predict the CAR[-1,+1] value itself (Huber loss)
                           # "direction":  predict only whether CAR[-1,+1] > 0 -- the same single
                           #               output becomes a logit, trained with BCEWithLogitsLoss
SEED = 0                   # fixes the starting weights and shuffle order, so two runs that differ
                           # only in a setting (e.g. USE_TIME_CYCLES) differ only because of that
                           # setting. For an ablation, repeat each side with 2-3 seeds (0, 1, 2):
                           # if the gap between settings is smaller than the gap between seeds,
                           # it is noise.
