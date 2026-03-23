"""
Central configuration — every tunable lives here.
No magic numbers anywhere else in the codebase.
"""

# --- Timeframes ---
ALLOWED_TIMEFRAMES = ["H4", "H1", "M30"]   # Allow full seeded timeframes

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
SYMBOLS         = ["XAGUSD"]   # Focused symbol
LOOKBACK_BARS   = 5000                              # bars per TF per symbol
DATA_CACHE_DIR  = "evolution_engine/data/cache"

# --- Species population split (fractions must sum to 1.0) ---
SPECIES_DISTRIBUTION = {
    "newton":     0.50,
    "turing":     0.50,
}

# --- Seed genomes (optional) ---
# Seeded at generation 0 to bootstrap evolution. Timeframes aligned to ALLOWED_TIMEFRAMES.
SEED_GENOMES = [
    {
        "genome_id": "4cda1f26-c5b0-4b99-b5d3-73c76e9eb4df",
        "species": "newton",
        "generation": 0,
        "indicators": [
            {"name": "RSI", "params": {"period": 7}, "timeframe": "M30", "role": "entry_trigger"},
            {"name": "EMA", "params": {"period": 126}, "timeframe": "H1", "role": "entry_filter"},
            {"name": "ADX", "params": {"period": 18}, "timeframe": "M30", "role": "exit_trigger"},
        ],
        "entry_logic": {
            "direction": "both",
            "logic_type": "OR",
            "conditions": [
                {"indicator_idx": 2, "comparison": "crossover", "reference_idx": 1, "threshold": None, "n_bars": 4, "sub_key": None},
                {"indicator_idx": 0, "comparison": "above_threshold", "reference_idx": None, "threshold": 60.20559044891423, "n_bars": 5, "sub_key": None},
                {"indicator_idx": 2, "comparison": "slope_negative", "reference_idx": None, "threshold": 40.88188540457086, "n_bars": 2, "sub_key": None},
            ],
        },
        "exit_logic": {"exit_type": "trailing_stop", "take_profit_r": 4.21, "stop_loss_atr_mult": 1.78},
        "risk": {"risk_per_trade": 0.0057, "max_concurrent_trades": 1, "daily_loss_limit": 0.0093},
        "parent_ids": ["80b461c7-1589-4235-9372-bb94cb3b23ec"],
        "fitness_score": 0.732523,
    },
    {
        "genome_id": "80b461c7-1589-4235-9372-bb94cb3b23ec",
        "species": "newton",
        "generation": 0,
        "indicators": [
            {"name": "RSI", "params": {"period": 7}, "timeframe": "M30", "role": "entry_trigger"},
            {"name": "EMA", "params": {"period": 126}, "timeframe": "H1", "role": "entry_filter"},
            {"name": "ADX", "params": {"period": 18}, "timeframe": "M30", "role": "exit_trigger"},
        ],
        "entry_logic": {
            "direction": "both",
            "logic_type": "OR",
            "conditions": [
                {"indicator_idx": 2, "comparison": "crossover", "reference_idx": 1, "threshold": None, "n_bars": 4, "sub_key": None},
                {"indicator_idx": 0, "comparison": "above_threshold", "reference_idx": None, "threshold": 60.20559044891423, "n_bars": 5, "sub_key": None},
                {"indicator_idx": 2, "comparison": "slope_negative", "reference_idx": None, "threshold": 40.88188540457086, "n_bars": 2, "sub_key": None},
            ],
        },
        "exit_logic": {"exit_type": "trailing_stop", "take_profit_r": 4.18, "stop_loss_atr_mult": 1.78},
        "risk": {"risk_per_trade": 0.0057, "max_concurrent_trades": 1, "daily_loss_limit": 0.0093},
        "parent_ids": ["ade7e5bc-8ba2-4ee4-b609-788392eb0ddb"],
        "fitness_score": 0.731736,
    },
    {
        "genome_id": "ade7e5bc-8ba2-4ee4-b609-788392eb0ddb",
        "species": "newton",
        "generation": 0,
        "indicators": [
            {"name": "RSI", "params": {"period": 7}, "timeframe": "M30", "role": "entry_trigger"},
            {"name": "EMA", "params": {"period": 126}, "timeframe": "H1", "role": "entry_filter"},
            {"name": "ADX", "params": {"period": 18}, "timeframe": "M30", "role": "exit_trigger"},
        ],
        "entry_logic": {
            "direction": "both",
            "logic_type": "OR",
            "conditions": [
                {"indicator_idx": 2, "comparison": "crossover", "reference_idx": 1, "threshold": None, "n_bars": 4, "sub_key": None},
                {"indicator_idx": 0, "comparison": "above_threshold", "reference_idx": None, "threshold": 60.20559044891423, "n_bars": 5, "sub_key": None},
                {"indicator_idx": 2, "comparison": "slope_negative", "reference_idx": None, "threshold": 40.88188540457086, "n_bars": 2, "sub_key": None},
            ],
        },
        "exit_logic": {"exit_type": "trailing_stop", "take_profit_r": 4.18, "stop_loss_atr_mult": 1.80},
        "risk": {"risk_per_trade": 0.0057, "max_concurrent_trades": 1, "daily_loss_limit": 0.0093},
        "parent_ids": ["4dd51ab7-9345-4c2c-9210-2041d3c5ac31"],
        "fitness_score": 0.731666,
    },
    {
        "genome_id": "b8d45af4-1733-403d-8ae8-1f9e94439aea",
        "species": "newton",
        "generation": 0,
        "indicators": [
            {"name": "RSI", "params": {"period": 7}, "timeframe": "M30", "role": "entry_trigger"},
            {"name": "EMA", "params": {"period": 126}, "timeframe": "H1", "role": "entry_filter"},
            {"name": "ADX", "params": {"period": 18}, "timeframe": "M30", "role": "exit_trigger"},
        ],
        "entry_logic": {
            "direction": "both",
            "logic_type": "OR",
            "conditions": [
                {"indicator_idx": 2, "comparison": "crossover", "reference_idx": 1, "threshold": None, "n_bars": 4, "sub_key": None},
                {"indicator_idx": 0, "comparison": "above_threshold", "reference_idx": None, "threshold": 60.20559044891423, "n_bars": 5, "sub_key": None},
                {"indicator_idx": 2, "comparison": "slope_negative", "reference_idx": None, "threshold": 40.88188540457086, "n_bars": 2, "sub_key": None},
            ],
        },
        "exit_logic": {"exit_type": "trailing_stop", "take_profit_r": 4.16, "stop_loss_atr_mult": 1.81},
        "risk": {"risk_per_trade": 0.0057, "max_concurrent_trades": 1, "daily_loss_limit": 0.0093},
        "parent_ids": ["893f8687-efc1-40d4-bf84-425d4ae9486b"],
        "fitness_score": 0.731107,
    },
    {
        "genome_id": "ede4d2ea-429c-4cc6-8035-29f36348be9d",
        "species": "newton",
        "generation": 0,
        "indicators": [
            {"name": "RSI", "params": {"period": 7}, "timeframe": "M30", "role": "entry_trigger"},
            {"name": "EMA", "params": {"period": 126}, "timeframe": "H1", "role": "entry_filter"},
            {"name": "ADX", "params": {"period": 18}, "timeframe": "M30", "role": "exit_trigger"},
        ],
        "entry_logic": {
            "direction": "both",
            "logic_type": "OR",
            "conditions": [
                {"indicator_idx": 2, "comparison": "crossover", "reference_idx": 1, "threshold": None, "n_bars": 4, "sub_key": None},
                {"indicator_idx": 0, "comparison": "above_threshold", "reference_idx": None, "threshold": 60.20559044891423, "n_bars": 5, "sub_key": None},
                {"indicator_idx": 2, "comparison": "slope_negative", "reference_idx": None, "threshold": 40.88188540457086, "n_bars": 2, "sub_key": None},
            ],
        },
        "exit_logic": {"exit_type": "trailing_stop", "take_profit_r": 4.09, "stop_loss_atr_mult": 1.83},
        "risk": {"risk_per_trade": 0.0057, "max_concurrent_trades": 1, "daily_loss_limit": 0.0093},
        "parent_ids": ["d944ce63-3397-438d-9347-40f3c2053466"],
        "fitness_score": 0.729203,
    },
    {
        "genome_id": "548b0244-7721-4952-af0f-f3b165d89e2c",
        "species": "newton",
        "generation": 0,
        "indicators": [
            {"name": "RSI", "params": {"period": 7}, "timeframe": "M30", "role": "entry_trigger"},
            {"name": "EMA", "params": {"period": 126}, "timeframe": "M30", "role": "entry_filter"},
            {"name": "ADX", "params": {"period": 18}, "timeframe": "M30", "role": "exit_trigger"},
        ],
        "entry_logic": {
            "direction": "both",
            "logic_type": "OR",
            "conditions": [
                {"indicator_idx": 2, "comparison": "crossover", "reference_idx": 1, "threshold": None, "n_bars": 4, "sub_key": None},
                {"indicator_idx": 0, "comparison": "above_threshold", "reference_idx": None, "threshold": 60.20559044891423, "n_bars": 5, "sub_key": None},
                {"indicator_idx": 2, "comparison": "above_threshold", "reference_idx": None, "threshold": 40.88188540457086, "n_bars": 4, "sub_key": None},
            ],
        },
        "exit_logic": {"exit_type": "trailing_stop", "take_profit_r": 4.23, "stop_loss_atr_mult": 1.89},
        "risk": {"risk_per_trade": 0.005, "max_concurrent_trades": 1, "daily_loss_limit": 0.0099},
        "parent_ids": ["f0834ba4-ab89-4dc8-8df7-f62b4eda0ad4"],
        "fitness_score": 0.498847,
    },
    {
        "genome_id": "f0834ba4-ab89-4dc8-8df7-f62b4eda0ad4",
        "species": "newton",
        "generation": 0,
        "indicators": [
            {"name": "RSI", "params": {"period": 7}, "timeframe": "M30", "role": "entry_trigger"},
            {"name": "EMA", "params": {"period": 126}, "timeframe": "M30", "role": "entry_filter"},
            {"name": "ADX", "params": {"period": 18}, "timeframe": "H4", "role": "exit_trigger"},
        ],
        "entry_logic": {
            "direction": "both",
            "logic_type": "OR",
            "conditions": [
                {"indicator_idx": 2, "comparison": "crossover", "reference_idx": 1, "threshold": None, "n_bars": 4, "sub_key": None},
                {"indicator_idx": 0, "comparison": "above_threshold", "reference_idx": None, "threshold": 60.20559044891423, "n_bars": 5, "sub_key": None},
                {"indicator_idx": 2, "comparison": "above_threshold", "reference_idx": None, "threshold": 40.88188540457086, "n_bars": 4, "sub_key": None},
            ],
        },
        "exit_logic": {"exit_type": "trailing_stop", "take_profit_r": 4.19, "stop_loss_atr_mult": 1.90},
        "risk": {"risk_per_trade": 0.005, "max_concurrent_trades": 1, "daily_loss_limit": 0.0099},
        "parent_ids": ["ee66e2d9-abd3-492f-a9e4-f4194812ca6f"],
        "fitness_score": 0.423105,
    },
    {
        "genome_id": "0068d934-cc2f-4823-ade0-96b3f39cc7c7",
        "species": "newton",
        "generation": 0,
        "indicators": [
            {"name": "RSI", "params": {"period": 7}, "timeframe": "M30", "role": "entry_trigger"},
            {"name": "EMA", "params": {"period": 126}, "timeframe": "M30", "role": "entry_filter"},
            {"name": "ADX", "params": {"period": 18}, "timeframe": "M30", "role": "exit_trigger"},
        ],
        "entry_logic": {
            "direction": "both",
            "logic_type": "OR",
            "conditions": [
                {"indicator_idx": 2, "comparison": "crossover", "reference_idx": 1, "threshold": None, "n_bars": 4, "sub_key": None},
                {"indicator_idx": 0, "comparison": "above_threshold", "reference_idx": None, "threshold": 60.20559044891423, "n_bars": 5, "sub_key": None},
                {"indicator_idx": 2, "comparison": "above_threshold", "reference_idx": None, "threshold": 40.88188540457086, "n_bars": 4, "sub_key": None},
            ],
        },
        "exit_logic": {"exit_type": "trailing_stop", "take_profit_r": 4.01, "stop_loss_atr_mult": 1.87},
        "risk": {"risk_per_trade": 0.005, "max_concurrent_trades": 1, "daily_loss_limit": 0.0099},
        "parent_ids": ["0d3bcd6f-5028-4de7-8b93-403d022d96c6"],
        "fitness_score": 0.502954,
    }
]

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
