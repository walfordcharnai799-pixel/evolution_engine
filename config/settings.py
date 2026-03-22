"""
Central configuration — every tunable lives here.
No magic numbers anywhere else in the codebase.
"""

# --- Timeframes ---
ALLOWED_TIMEFRAMES = ["H4", "H1", "M30"]   # M15 = visual inspection only, not used in logic

# MT5 timeframe constants (imported lazily to avoid hard dep at import time)
MT5_TF_NAMES = {
    "H4":  "TIMEFRAME_H4",
    "H1":  "TIMEFRAME_H1",
    "M30": "TIMEFRAME_M30",
    "M15": "TIMEFRAME_M15",
}

# --- Genome constraints ---
NUM_INDICATORS = 3
MIN_POPULATION_SIZE = 40
ELITE_FRACTION = 0.20        # top 20 % pass as elites
MUTATION_BASE_RATE = 0.15

# --- Survival filter thresholds (THE REFEREE — progressive) ---
# Gates tighten each phase. Full pressure only at gen 61+.
SURVIVAL = {
    "min_win_rate":           0.74,    # Final target: 74% win rate
    "min_profit_factor":      2.50,    # Final target PF
    "max_drawdown":           0.10,    # Final target: 10% max DD
    "min_trades":             22,      # Min trades per backtest window
    "min_expectancy":         0.50,    # Min 0.5R expectancy
    "max_stability_var":      0.30,    # Stability threshold
    "min_rr_ratio":           3.0,     # Min 1:3 R/R
    "spread_pips":            3.0,     # 3-pip slippage per trade
    "recovery_bars":          5,       # 5-bar recovery rule
}

# Progressive gate schedule — gates tighten as species mature
# Format: (min_gen, max_gen): {threshold overrides}
PROGRESSIVE_GATES = [
    (0,  10, {"min_win_rate": 0.45, "max_drawdown": 0.20, "min_profit_factor": 1.30, "min_expectancy": 0.0,  "recovery_bars": 999}),
    (11, 30, {"min_win_rate": 0.55, "max_drawdown": 0.15, "min_profit_factor": 1.80, "min_expectancy": 0.20, "recovery_bars": 999}),
    (31, 60, {"min_win_rate": 0.65, "max_drawdown": 0.12, "min_profit_factor": 2.20, "min_expectancy": 0.35, "recovery_bars": 10}),
    (61, 999,{"min_win_rate": 0.74, "max_drawdown": 0.10, "min_profit_factor": 2.50, "min_expectancy": 0.50, "recovery_bars": 5}),
]

# --- Fitness composite weights (must sum to 1.0) ---
FITNESS_WEIGHTS = {
    "profit_factor": 0.30,
    "win_rate":      0.20,
    "drawdown":      0.20,
    "expectancy":    0.15,
    "stability":     0.15,
}

# --- Data ---
SYMBOLS         = ["XAUUSD", "XAGUSD", "BTCUSD"]   # Gold, Silver, Bitcoin
LOOKBACK_BARS   = 5000                              # bars per TF per symbol
DATA_CACHE_DIR  = "evolution_engine/data/cache"

# --- Species population split (fractions must sum to 1.0) ---
SPECIES_DISTRIBUTION = {
    "davinci":    0.125,
    "newton":     0.125,
    "tesla":      0.125,
    "aristotle":  0.125,
    "hypatia":    0.125,
    "archimedes": 0.125,
    "hawking":    0.125,
    "turing":     0.125,
}

# --- Evolution ---
MAX_GENERATIONS = 100
RANDOM_SEED     = 42           # determinism anchor

# --- MT5 connection (override via CLI / config JSON) ---
MT5_LOGIN    = 0
MT5_PASSWORD = ""
MT5_SERVER   = ""
MT5_PATH     = ""              # path to terminal64.exe (Wine)

# --- Risk limits ---
MAX_RISK_PER_TRADE = 0.05      # absolute cap regardless of species
MIN_RISK_PER_TRADE = 0.001

# --- Indicator parameter ranges (global defaults) ---
INDICATOR_PARAM_RANGES = {
    "EMA":        {"period": (5, 200)},
    "SMA":        {"period": (5, 200)},
    "WMA":        {"period": (5, 200)},
    "MACD":       {"fast": (8, 16), "slow": (21, 50), "signal": (7, 12)},
    "ADX":        {"period": (7, 28)},
    "RSI":        {"period": (7, 21)},
    "STOCHASTIC": {"k_period": (5, 21), "d_period": (3, 9)},
    "CCI":        {"period": (10, 30)},
    "MOMENTUM":   {"period": (10, 20)},
    "BOLLINGER":  {"period": (10, 30), "std_dev": (1.5, 3.0)},
    "ATR":        {"period": (7, 21)},
    "WILLIAMS_R": {"period": (7, 21)},
    "OBV":        {},
}
