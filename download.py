import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================
#  CONFIG
# ============================================================
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
STATE_FILE = os.path.join(DATA_DIR, "download_state.json")

ASSETS = {
    "Gold":    "GC=F",
    "Bitcoin": "BTC-USD",
}

INTERVAL_CONFIG = {
    "15m": ("15m",  "60d",   None,  15),
    "30m": ("30m",  "60d",   None,  30),
    "1h":  ("1h",   "730d",  None,  60),
    "4h":  ("1h",   "730d",   "4h",  240),
    "1d":  ("1d",   "max",   None,  1440),
}
# ============================================================


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


# ── Build XAUBTC ratio CSVs from Gold + Bitcoin ──────────────
print("  Building XAUBTC ratio...")
for interval_label in INTERVAL_CONFIG.keys():
    try:
        gold_path = os.path.join(DATA_DIR, f"Gold_{interval_label}.csv")
        btc_path  = os.path.join(DATA_DIR, f"Bitcoin_{interval_label}.csv")
        if not (os.path.exists(gold_path) and os.path.exists(btc_path)):
            continue

        gold = pd.read_csv(gold_path, index_col=0, parse_dates=True)
        btc  = pd.read_csv(btc_path,  index_col=0, parse_dates=True)

        # Align on Gold timestamps (drops weekends from BTC)
        merged = gold[["Close"]].rename(columns={"Close": "G"}).join(
            btc[["Close"]].rename(columns={"Close": "B"}), how="inner"
        ).dropna()

        if merged.empty:
            continue

        ratio = pd.DataFrame(index=merged.index)
        ratio["Open"]   = merged["G"] / merged["B"]
        ratio["High"]   = merged["G"] / merged["B"]
        ratio["Low"]    = merged["G"] / merged["B"]
        ratio["Close"]  = merged["G"] / merged["B"]
        ratio["Volume"] = 0
        ratio.index.name = "Datetime"

        out_path = os.path.join(DATA_DIR, f"XAUBTC_{interval_label}.csv")
        ratio.to_csv(out_path)
        print(f"  OK    XAUBTC_{interval_label:<14} {len(ratio):>6,} rows")
    except Exception as e:
        print(f"  ERROR XAUBTC_{interval_label:<14} {e}")
        
def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_close_df(df, symbol):
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(symbol, axis=1, level=1)
    cols = [c for c in ["Open","High","Low","Close","Volume"] if c in df.columns]
    return df[cols]


def to_bangkok(index):
    if index.tzinfo is not None:
        return index.tz_convert("Asia/Bangkok")
    return index.tz_localize("UTC").tz_convert("Asia/Bangkok")


def should_fetch(state, key, min_minutes):
    last = state.get(key)
    if not last:
        return True
    return datetime.now() - datetime.fromisoformat(last) >= timedelta(minutes=min_minutes)


def download_and_save(asset_name, symbol, interval_label):
    yf_iv, yf_period, resample_to, _ = INTERVAL_CONFIG[interval_label]
    df = yf.download(symbol, period=yf_period, interval=yf_iv,
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None, 0

    ohlcv = get_close_df(df, symbol)
    if resample_to:
        ohlcv = ohlcv.resample(resample_to).agg({
            "Open": "first", "High": "max",
            "Low": "min", "Close": "last", "Volume": "sum"
        }).dropna()

    ohlcv.index = to_bangkok(ohlcv.index)
    ohlcv.index = ohlcv.index.tz_localize(None)
    ohlcv.index.name = "Datetime"

    filepath = os.path.join(DATA_DIR, f"{asset_name}_{interval_label}.csv")
    if os.path.exists(filepath):
        existing = pd.read_csv(filepath, index_col=0, parse_dates=True)
        combined = pd.concat([existing, ohlcv])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        ohlcv = combined

    ohlcv.to_csv(filepath)
    return filepath, len(ohlcv)


# ── Main ─────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
state = load_state()
now   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[{now}] download.py starting...")

for asset_name, symbol in ASSETS.items():
    for interval_label, (_, _, _, min_min) in INTERVAL_CONFIG.items():
        key = f"{asset_name}_{interval_label}"
        if not should_fetch(state, key, min_min):
            print(f"  SKIP  {key:<20} (next in {min_min}min)")
            continue
        try:
            filepath, rows = download_and_save(asset_name, symbol, interval_label)
            if filepath:
                state[key] = datetime.now().isoformat()
                print(f"  OK    {key:<20} {rows:>6,} rows -> {os.path.basename(filepath)}")
            else:
                print(f"  WARN  {key:<20} no data returned")
        except Exception as e:
            print(f"  ERROR {key:<20} {e}")

save_state(state)
print(f"[{datetime.now().strftime('%H:%M:%S')}] download.py done.\n")
