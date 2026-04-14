import pandas as pd
import os
import json
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================
#  CONFIG
# ============================================================
BASE_DIR      = os.path.dirname(__file__)
DATA_DIR      = os.path.join(BASE_DIR, "data")
OUTPUT_HTML   = os.path.join(DATA_DIR, "ema_cross_report.html")
STATE_FILE    = os.path.join(DATA_DIR, "report_state.json")

GMAIL_SENDER   = "payotorn@gmail.com"
GMAIL_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
ALERT_TO       = "payotorn@gmail.com"

MAX_ROWS       = 200
LOOKBACK_DAYS  = 30
MIN_EMAIL_GAP  = 60

ASSETS = {
    "Gold":    "GC=F",
    "Bitcoin": "BTC-USD",
    "XAUBTC":  "XAUBTC",
}

EMA_PAIRS = [
    (12,  26,  "S"),
    (20,  50,  "M"),
    (50, 200,  "L"),
]

INTERVALS = ["15m", "30m", "1h", "4h", "1d"]
# ============================================================


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"sent_events": [], "last_email": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def calc_ema(close, p):
    return close.ewm(span=p, adjust=False).mean()


def detect_cross(ema_fast, ema_slow, i):
    if i < 1:
        return None
    prev = ema_fast.iloc[i-1] - ema_slow.iloc[i-1]
    curr = ema_fast.iloc[i]   - ema_slow.iloc[i]
    if prev < 0 and curr >= 0:
        return "GOLDEN"
    if prev > 0 and curr <= 0:
        return "DEATH"
    return None


def load_csv(asset_name, interval):
    filepath = os.path.join(DATA_DIR, f"{asset_name}_{interval}.csv")
    if not os.path.exists(filepath):
        return None
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df["Close"].dropna()


def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def collect_rsi(asset_name):
    """Build RSI Series per interval for nearest-timestamp lookup."""
    rsi_series = {}
    for interval in INTERVALS:
        close = load_csv(asset_name, interval)
        if close is None or len(close) < 20:
            continue
        rsi_series[interval] = calc_rsi(close)
    return rsi_series


def lookup_rsi(rsi_series, date_str, time_str):
    """Look up RSI for each interval at nearest timestamp <= event time."""
    result = {}
    try:
        ts = pd.Timestamp(f"{date_str} {time_str}")
    except Exception:
        return result
    for iv, rsi in rsi_series.items():
        val = rsi.asof(ts)
        if pd.notna(val):
            result[iv] = round(float(val), 1)
    return result


def collect_events(asset_name):
    all_events = {}

    for interval in INTERVALS:
        close = load_csv(asset_name, interval)
        if close is None:
            continue

        for fast, slow, label in EMA_PAIRS:
            if len(close) < slow + 10:
                continue
            ema_f = calc_ema(close, fast)
            ema_s = calc_ema(close, slow)

            for i in range(1, len(close)):
                ts = close.index[i]
                cross = detect_cross(ema_f, ema_s, i)
                if cross:
                    date_str  = ts.strftime("%Y-%m-%d")
                    time_str  = ts.strftime("%H:%M")
                    price_val = float(close.iloc[i])
                    key = (date_str, time_str)
                    if key not in all_events:
                        all_events[key] = {"price": price_val, "crosses": {}}
                    all_events[key]["crosses"].setdefault(interval, {})[label] = cross

    return all_events


def cell_html(cross, iv_sep=False, last_signal=None, last_price=None, cur_price=None, iv="", lbl=""):
    classes = []
    if iv_sep:
        classes.append("iv-sep")
    attr = f' data-iv="{iv}" data-lbl="{lbl}"'
    if cross == "GOLDEN":
        cls_str = f' class="{" ".join(classes)}"' if classes else ''
        return f'<td{cls_str}{attr}><span class="g">G</span></td>'
    elif cross == "DEATH":
        cls_str = f' class="{" ".join(classes)}"' if classes else ''
        return f'<td{cls_str}{attr}><span class="d">D</span></td>'
    else:
        if last_signal and last_price and cur_price:
            if last_signal == "GOLDEN":
                classes.append("bg-g")
            elif last_signal == "DEATH":
                classes.append("bg-d")
            cls_str = f' class="{" ".join(classes)}"' if classes else ''
            if cur_price > last_price:
                return f'<td{cls_str}{attr}><span class="dot-g">●</span></td>'
            elif cur_price < last_price:
                return f'<td{cls_str}{attr}><span class="dot-d">●</span></td>'
            else:
                return f'<td{cls_str}{attr}><span class="n">—</span></td>'
        cls_str = f' class="{" ".join(classes)}"' if classes else ''
        return f'<td{cls_str}{attr}><span class="n">—</span></td>'


def fmt_price(price):
    if price >= 1000:
        return f"${price:,.0f}"
    elif price >= 1:
        return f"${price:,.2f}"
    else:
        return f"{price:.4f}"
        
def rsi_cell(val, iv_sep=False, iv=""):
    cls = "iv-sep " if iv_sep else ""
    attr = f' data-iv="{iv}" data-lbl="R"'
    if val is None:
        return f'<td class="{cls}rsi"{attr}><span class="n">—</span></td>'
    if val >= 70:
        return f'<td class="{cls}rsi rsi-hi"{attr}>{val:.0f}</td>'
    elif val <= 30:
        return f'<td class="{cls}rsi rsi-lo"{attr}>{val:.0f}</td>'
    else:
        return f'<td class="{cls}rsi"{attr}>{val:.0f}</td>'


def build_indicator_html(all_events, display_keys, rsi_data):
    """Build compact min-max indicator: horizontal price bar + 5 vertical RSI bars."""
    if not display_keys:
        return ""

    # Price range from displayed event rows
    prices = [all_events[k]["price"] for k in display_keys]
    p_min, p_max = min(prices), max(prices)
    p_cur = all_events[display_keys[-1]]["price"]
    p_pct = ((p_cur - p_min) / (p_max - p_min) * 100) if p_max > p_min else 50

    # RSI for each interval at latest timestamp
    last_key = display_keys[-1]
    rsi_row = lookup_rsi(rsi_data, last_key[0], last_key[1]) if rsi_data else {}

    # Color helper for price (red→yellow→green by position)
    def price_color(pct):
        if pct < 33:   return "#ef4444"
        if pct < 66:   return "#eab308"
        return "#22c55e"

    # RSI bars (vertical)
    rsi_bars = ""
    for iv in INTERVALS:
        v = rsi_row.get(iv)
        if v is None:
            rsi_bars += f'<div class="rsi-bar"><div class="rsi-track"></div><div class="rsi-lbl">{iv}</div><div class="rsi-val">—</div></div>'
            continue
        # marker position from bottom: 0 = bottom, 100 = top
        pos = max(0, min(100, v))
        if v >= 70:    mcolor = "#ef4444"
        elif v <= 30:  mcolor = "#22c55e"
        else:          mcolor = "#eab308"
        rsi_bars += f"""<div class="rsi-bar">
            <div class="rsi-track">
              <div class="rsi-marker" style="bottom:{pos}%;background:{mcolor};"></div>
            </div>
            <div class="rsi-lbl">{iv}</div>
            <div class="rsi-val" style="background:{mcolor};">{int(v)}</div>
          </div>"""

    return f"""
    <div class="indicator-box">
      <div class="ind-price">
        <div class="ind-price-label">Price</div>
        <div class="ind-price-bar">
          <span class="ind-min">{fmt_price(p_min)}</span>
          <div class="ind-track">
            <div class="ind-marker" style="left:{p_pct:.0f}%;background:{price_color(p_pct)};"></div>
            <div class="ind-cur" style="left:{p_pct:.0f}%;background:{price_color(p_pct)};">{fmt_price(p_cur)}</div>
          </div>
          <span class="ind-max">{fmt_price(p_max)}</span>
        </div>
      </div>
      <div class="ind-rsi">
        <div class="ind-rsi-label">RSI</div>
        <div class="ind-rsi-bars">{rsi_bars}</div>
      </div>
    </div>
    """

def build_heatmap_html(all_events, display_keys):
    """Build SVG heatmap: 200 rows x 15 cols (5 tf x 3 EMA). Cell 5x2 px."""
    if not display_keys:
        return ""

    CELL_W = 12
    CELL_H = 2
    GAP    = 1  # gap between timeframe groups
    n_tf   = len(INTERVALS)
    n_ema  = len(EMA_PAIRS)
    cols   = n_tf * n_ema
    rows   = len(display_keys)

    width  = cols * CELL_W + (n_tf - 1) * GAP
    height = rows * CELL_H

    # Track state per (iv, lbl) across full history (for bg tint)
    col_state = {}
    all_sorted = sorted(all_events.keys())
    pre_keys = [k for k in all_sorted if k not in display_keys]
    for k in pre_keys:
        ev = all_events[k]
        for iv in INTERVALS:
            for _, _, lbl in EMA_PAIRS:
                c = ev["crosses"].get(iv, {}).get(lbl)
                if c:
                    col_state[(iv, lbl)] = c

    rects = ""
    for r_idx, key in enumerate(display_keys):
        ev = all_events[key]
        y = r_idx * CELL_H
        for iv_idx, iv in enumerate(INTERVALS):
            iv_data = ev["crosses"].get(iv, {})
            for lbl_idx, (_, _, lbl) in enumerate(EMA_PAIRS):
                x = (iv_idx * n_ema + lbl_idx) * CELL_W + iv_idx * GAP
                cross = iv_data.get(lbl)
                if cross:
                    col_state[(iv, lbl)] = cross
                    fill = "#16a34a" if cross == "GOLDEN" else "#dc2626"
                else:
                    state_c = col_state.get((iv, lbl))
                    if state_c == "GOLDEN":
                        fill = "#dcfce7"
                    elif state_c == "DEATH":
                        fill = "#fee2e2"
                    else:
                        fill = "#f1efe8"
                rects += f'<rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}" fill="{fill}"/>'

    return f"""
    <div class="heatmap-box">
      <div class="heatmap-label">HEATMAP · {rows} rows</div>
      <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">
        {rects}
      </svg>
    </div>
    """

def build_full_heatmap_html(asset_name, all_events, years=4):
    """Build full-history heatmap for modal. Cell 8x1 px."""
    CELL_W = 8
    CELL_H = 1
    GAP    = 1
    n_tf   = len(INTERVALS)
    n_ema  = len(EMA_PAIRS)
    cols   = n_tf * n_ema

    # Filter events within N years
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365 * years)
    all_sorted = sorted(all_events.keys())
    display_keys = [k for k in all_sorted if pd.Timestamp(f"{k[0]} {k[1]}") >= cutoff]

    if not display_keys:
        return ""

    rows = len(display_keys)
    width  = cols * CELL_W + (n_tf - 1) * GAP
    height = rows * CELL_H

    # Track state across pre-cutoff events
    col_state = {}
    pre_keys = [k for k in all_sorted if k not in display_keys]
    for k in pre_keys:
        ev = all_events[k]
        for iv in INTERVALS:
            for _, _, lbl in EMA_PAIRS:
                c = ev["crosses"].get(iv, {}).get(lbl)
                if c:
                    col_state[(iv, lbl)] = c

    rects = ""
    for r_idx, key in enumerate(display_keys):
        ev = all_events[key]
        y = r_idx * CELL_H
        for iv_idx, iv in enumerate(INTERVALS):
            iv_data = ev["crosses"].get(iv, {})
            for lbl_idx, (_, _, lbl) in enumerate(EMA_PAIRS):
                x = (iv_idx * n_ema + lbl_idx) * CELL_W + iv_idx * GAP
                cross = iv_data.get(lbl)
                if cross:
                    col_state[(iv, lbl)] = cross
                    fill = "#16a34a" if cross == "GOLDEN" else "#dc2626"
                else:
                    state_c = col_state.get((iv, lbl))
                    if state_c == "GOLDEN":
                        fill = "#dcfce7"
                    elif state_c == "DEATH":
                        fill = "#fee2e2"
                    else:
                        fill = "#f1efe8"
                rects += f'<rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}" fill="{fill}"/>'

    first_date = display_keys[0][0]
    last_date  = display_keys[-1][0]

    return f"""
    <div class="modal-overlay" id="modal-{asset_name}" onclick="if(event.target===this)closeModal('{asset_name}')">
      <div class="modal-content">
        <div class="modal-header">
          <div class="modal-title">{asset_name} · Full Heatmap · {rows} events · {first_date} → {last_date}</div>
          <button class="modal-close" onclick="closeModal('{asset_name}')">×</button>
        </div>
        <div class="modal-body">
          <svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">
            {rects}
          </svg>
        </div>
      </div>
    </div>
    """
    
def build_table_html(asset_name, all_events, rsi_data=None):
    n_ema       = len(EMA_PAIRS)
    n_sub       = n_ema + 1   # S, M, L, R per interval
    total_cols  = 2 + len(INTERVALS) * n_sub
    all_sorted  = sorted(all_events.keys())
    display_keys = all_sorted[-MAX_ROWS:]
    
    # Pre-scan ALL events to build col_state before display window
    col_state = {}
    col_price = {}
    pre_keys = [k for k in all_sorted if k not in display_keys]
    for (date_str, time_str) in pre_keys:
        ev = all_events[(date_str, time_str)]
        for iv in INTERVALS:
            iv_data = ev["crosses"].get(iv, {})
            for _, _, lbl in EMA_PAIRS:
                cross = iv_data.get(lbl)
                if cross:
                    col_state[(iv, lbl)] = cross
                    col_price[(iv, lbl)] = ev["price"]

    # Latest event summary
    indicator_html = build_indicator_html(all_events, display_keys, rsi_data)
    heatmap_html = build_heatmap_html(all_events, display_keys)
    full_heatmap_html = build_full_heatmap_html(asset_name, all_events, years=4)
    
    summary_html = ""
    if display_keys:
        last_key  = display_keys[-1]
        last_ev   = all_events[last_key]
        last_date, last_time = last_key
        chips = []
        for iv in INTERVALS:
            iv_data = last_ev["crosses"].get(iv, {})
            for fast, slow, lbl in EMA_PAIRS:
                cross = iv_data.get(lbl)
                if cross:
                    label_full = {"S": f"Short {fast}/{slow}", "M": f"Mid {fast}/{slow}", "L": f"Long {fast}/{slow}"}.get(lbl, lbl)
                    cls  = "chip-g" if cross == "GOLDEN" else "chip-d"
                    icon = "G" if cross == "GOLDEN" else "D"
                    chips.append(f'<span class="chip {cls}"><span class="chip-badge">{icon}</span>{cross.capitalize()} · {iv} · {label_full}</span>')
        if chips:
            chips_str = "\n".join(chips)
            summary_html = f"""
            <div class="summary-box">
              <span class="summary-label">Latest event &nbsp;·&nbsp; {last_date} {last_time} &nbsp;·&nbsp; {fmt_price(last_ev['price'])}</span>
              <div class="summary-chips">{chips_str}</div>
            </div>"""

    # Header: By Interval (default)
    h1  = '<tr class="hdr-iv hdr-r1">'
    h1 += '<th rowspan="2" class="sticky s0 left th-fix">Time</th>'
    h1 += '<th rowspan="2" class="sticky s1 left th-fix">Price</th>'
    for iv in INTERVALS:
        h1 += f'<th colspan="{n_sub}" class="iv-sep">{iv}</th>'
    h1 += '</tr>'

    h2 = '<tr class="hdr-iv hdr-r2">'
    for idx, iv in enumerate(INTERVALS):
        for j, (_, _, lbl) in enumerate(EMA_PAIRS):
            cls = ' class="iv-sep"' if j == 0 and idx > 0 else ''
            h2 += f'<th{cls} data-iv="{iv}" data-lbl="{lbl}">{lbl}</th>'
        h2 += f'<th class="rsi-hdr" data-iv="{iv}" data-lbl="R">R</th>'
    h2 += '</tr>'

    # Header: By EMA (hidden)
    h1b  = '<tr class="hdr-ema hdr-r1" style="display:none;">'
    h1b += '<th rowspan="2" class="sticky s0 left th-fix">Time</th>'
    h1b += '<th rowspan="2" class="sticky s1 left th-fix">Price</th>'
    for _, _, lbl in EMA_PAIRS:
        full = {"S":"Short","M":"Mid","L":"Long"}[lbl]
        h1b += f'<th colspan="{len(INTERVALS)}" class="iv-sep">{full}</th>'
    h1b += f'<th colspan="{len(INTERVALS)}" class="iv-sep rsi-hdr">RSI</th>'
    h1b += '</tr>'

    h2b = '<tr class="hdr-ema hdr-r2" style="display:none;">'
    for j, (_, _, lbl) in enumerate(EMA_PAIRS):
        for idx, iv in enumerate(INTERVALS):
            cls = ' class="iv-sep"' if idx == 0 and j > 0 else ''
            h2b += f'<th{cls} data-iv="{iv}" data-lbl="{lbl}">{iv}</th>'
    for idx, iv in enumerate(INTERVALS):
        cls = ' class="iv-sep"' if idx == 0 else ''
        h2b += f'<th{cls} data-iv="{iv}" data-lbl="R">{iv}</th>'
    h2b += '</tr>'

    rows_html = ""
    prev_date = None
    for (date_str, time_str) in display_keys:
        if prev_date != date_str:
            rows_html += f'<tr class="day-sep"><td class="sticky s0 day-label">{date_str}</td><td colspan="{total_cols-1}"></td></tr>'
        prev_date = date_str

        ev   = all_events[(date_str, time_str)]
        cur_price = ev["price"]
        rsi_row = lookup_rsi(rsi_data, date_str, time_str) if rsi_data else {}
        row  = '<tr class="data">'
        row += f'<td class="sticky s0 left tm">{time_str}</td>'
        row += f'<td class="sticky s1 left price">{fmt_price(cur_price)}</td>'
        for iv_idx, iv in enumerate(INTERVALS):
            iv_data = ev["crosses"].get(iv, {})
            for j, (_, _, lbl) in enumerate(EMA_PAIRS):
                first_in_group = (iv_idx > 0 and j == 0)
                cross = iv_data.get(lbl)
                col_key = (iv, lbl)
                if cross:
                    col_state[col_key] = cross
                    col_price[col_key] = cur_price
                row += cell_html(cross, first_in_group, col_state.get(col_key), col_price.get(col_key), cur_price, iv, lbl)
            row += rsi_cell(rsi_row.get(iv), iv_sep=False, iv=iv)
        row += '</tr>'
        rows_html += row

    if not rows_html:
        rows_html = f'<tr><td colspan="{total_cols}" class="empty">No EMA cross events</td></tr>'

    return f"""
    <div class="asset-block">
      <div class="asset-title">{asset_name}</div>
      {summary_html}
      <div class="ind-heatmap-wrap">
        {indicator_html}
        <div class="heatmap-row">
          {heatmap_html}
          <button class="heatmap-expand" onclick="openModal('{asset_name}')" title="View 4-year heatmap">⛶</button>
        </div>
      </div>
      {full_heatmap_html}
      <div class="table-scroll">
        <table>
          <thead>{h1}{h2}{h1b}{h2b}</thead>
          <tbody>{rows_html}</tbody>
        </table>
      </div>
    </div>
    """


def build_html(sections):
    now       = datetime.now().strftime("%Y-%m-%d %H:%M")
    body      = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="th" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>EMA Cross Report</title>
<style>
  :root {{
    --bg:#0f0f0f;--bg2:#1a1a1a;--bg3:#111;--bg4:#161616;
    --border:#2a2a2a;--border2:#333;
    --text:#e8e0d0;--text2:#aaa;--text3:#666;--text4:#444;
    --gold:#f5d06e;--blue:#60a5fa;
    --g-bg:#0d3d22;--g-fg:#4ade80;
    --d-bg:#3d0d0d;--d-fg:#f87171;
    --price:#7dd3fc;--tog-bg:#2a2a2a;
    --shadow:4px 0 8px rgba(0,0,0,.5);
  }}
  [data-theme="light"] {{
    --bg:#f5f5f0;--bg2:#fff;--bg3:#efefea;--bg4:#f0f0eb;
    --border:#ddd;--border2:#ccc;
    --text:#1a1a1a;--text2:#444;--text3:#888;--text4:#bbb;
    --gold:#92400e;--blue:#1d4ed8;
    --g-bg:#dcfce7;--g-fg:#166534;
    --d-bg:#fee2e2;--d-fg:#991b1b;
    --price:#0369a1;--tog-bg:#e5e5e5;
    --shadow:4px 0 6px rgba(0,0,0,.1);
  }}
  *{{box-sizing:border-box;margin:0;padding:0;}}
  body{{font-family:'Segoe UI',Arial,sans-serif;background:var(--bg);color:var(--text);padding:14px 10px;transition:background .2s,color .2s;}}
  .top-bar{{display:flex;justify-content:space-between;align-items:flex-start;gap:10px;margin-bottom:12px;}}
  .page-title{{font-size:15px;font-weight:700;color:var(--gold);margin-bottom:2px;}}
  .page-sub{{font-size:11px;color:var(--text3);}}
  .toggle-btn{{flex-shrink:0;background:var(--tog-bg);border:1px solid var(--border2);border-radius:20px;padding:4px 12px;font-size:11px;font-weight:700;color:var(--text2);cursor:pointer;display:flex;align-items:center;gap:5px;transition:all .2s;white-space:nowrap;}}
  .toggle-btn:hover{{border-color:var(--gold);color:var(--gold);}}
  .legend{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px;font-size:11px;color:var(--text3);}}
  .leg{{display:flex;align-items:center;gap:4px;}}
  .lg{{background:var(--g-bg);color:var(--g-fg);border-radius:3px;padding:1px 7px;font-weight:700;}}
  .ld{{background:var(--d-bg);color:var(--d-fg);border-radius:3px;padding:1px 7px;font-weight:700;}}
  .asset-header{{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;margin-bottom:8px;}}
  .asset-header-left{{flex:1;min-width:0;}}
  .ind-heatmap-wrap{{display:inline-flex;flex-direction:column;gap:6px;margin-bottom:8px;}}
  .indicator-box{{display:inline-flex;flex-direction:column;gap:6px;padding:8px 10px;border-radius:8px;background:var(--bg2);border:1.5px solid var(--border2);min-width:240px;}}
  .heatmap-box{{padding:6px 8px;border-radius:8px;background:var(--bg2);border:1.5px solid var(--border2);display:inline-block;}}
  .heatmap-label{{font-size:9px;color:var(--text3);font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;}}
  .ind-price{{display:flex;align-items:center;gap:8px;padding-bottom:14px;}}
  .ind-price-label{{font-size:9px;color:var(--text3);font-weight:700;text-transform:uppercase;letter-spacing:.5px;width:28px;}}
  .ind-price-bar{{flex:1;display:flex;align-items:center;gap:6px;}}
  .ind-min,.ind-max{{font-size:9px;color:var(--text3);font-weight:700;font-family:monospace;white-space:nowrap;}}
  .ind-track{{flex:1;height:8px;border-radius:3px;background:linear-gradient(to right,#ef4444 0%,#eab308 50%,#22c55e 100%);position:relative;opacity:.35;}}
  .ind-marker{{position:absolute;top:-3px;width:4px;height:14px;border-radius:2px;transform:translateX(-2px);border:1px solid var(--text);box-shadow:0 0 0 1.5px var(--bg2);z-index:2;}}
  .ind-cur{{position:absolute;top:14px;font-size:9px;font-weight:700;font-family:monospace;transform:translateX(-50%);white-space:nowrap;color:#fff!important;padding:2px 6px;border-radius:4px;}}
  .ind-rsi{{display:flex;align-items:flex-start;gap:8px;}}
  .ind-rsi-label{{font-size:9px;color:var(--text3);font-weight:700;text-transform:uppercase;letter-spacing:.5px;width:28px;padding-top:4px;}}
  .ind-rsi-bars{{flex:1;display:flex;justify-content:space-between;gap:4px;}}
  .rsi-bar{{display:flex;flex-direction:column;align-items:center;gap:2px;flex:1;}}
  .rsi-track{{width:7px;height:36px;border-radius:3px;background:linear-gradient(to top,#22c55e 0%,#eab308 50%,#ef4444 100%);position:relative;opacity:.35;}}
  .rsi-marker{{position:absolute;left:-2px;width:10px;height:3px;border-radius:1px;border:1px solid var(--text);box-shadow:0 0 0 1px var(--bg2);}}
  .rsi-lbl{{font-size:8px;color:var(--text3);font-weight:700;}}
  .rsi-val{{font-size:9px;font-weight:700;font-family:monospace;color:#fff!important;padding:2px 5px;border-radius:4px;min-width:18px;text-align:center;}}
  .summary-box{{margin-bottom:8px;padding:8px 10px;border-radius:8px;background:var(--bg2);border:1px solid var(--border);}}
  .summary-label{{font-size:11px;color:var(--text3);font-weight:600;display:block;margin-bottom:6px;}}
  .summary-chips{{display:flex;flex-wrap:wrap;gap:6px;}}
  .chip{{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700;padding:3px 8px;border-radius:5px;}}
  .chip-g{{background:var(--g-bg);color:var(--g-fg);}}
  .chip-d{{background:var(--d-bg);color:var(--d-fg);}}
  .chip-badge{{font-weight:700;font-size:11px;}}
  .asset-block{{margin-bottom:22px;}}
  .asset-title{{font-size:13px;font-weight:700;color:var(--gold);margin-bottom:6px;}}
  .table-scroll{{overflow-x:auto;overflow-y:auto;max-height:420px;border-radius:8px;border:1px solid var(--border);}}
  table{{border-collapse:collapse;font-size:12px;font-weight:700;font-family:'Segoe UI',Arial,sans-serif;background:var(--bg2);width:auto;}}
  th,td{{padding:4px 6px;text-align:center;white-space:nowrap;border-bottom:.5px solid var(--border);font-weight:700;}}
  th{{background:var(--bg3);color:var(--text3);font-size:11px;}}
  .left{{text-align:left!important;}}
  thead th{{position:sticky;z-index:3;background:var(--bg3);}}
  thead .hdr-r1 th{{top:0;color:var(--blue);font-size:12px;padding-top:7px;padding-bottom:2px;border-bottom:none;}}
  thead .hdr-r2 th{{top:26px;color:var(--text3);font-size:11px;padding-top:1px;padding-bottom:5px;border-bottom:2px solid var(--border2);box-shadow:0 2px 4px rgba(0,0,0,.15);}}
  th.iv-sep{{border-left:2px solid var(--border2);}}
  td.iv-sep{{border-left:2px solid var(--border2);}}
  .sticky{{position:sticky;z-index:2;background:var(--bg2);}}
  thead .sticky{{z-index:5;background:var(--bg3);}}
  .s0{{left:0;}} .s1{{left:52px;box-shadow:var(--shadow);}}
  thead .s1{{box-shadow:var(--shadow);}}
  tr.data:hover .sticky{{background:var(--bg4);}}
  td.tm{{color:var(--text2);}} td.price{{color:var(--price);min-width:72px;}}
  .th-fix{{color:var(--text3)!important;font-size:11px!important;}}
  tr.data:hover td{{background:var(--bg4);}}
  tr.day-sep td{{background:var(--bg4)!important;color:var(--gold);font-size:11px;text-align:left;padding:4px 8px;letter-spacing:.5px;border-top:1px solid var(--border2);border-bottom:1px solid var(--border2);font-weight:700;}}
  .day-label{{font-weight:700!important;color:var(--gold)!important;}}
  .g{{display:inline-block;background:var(--g-bg);color:var(--g-fg);border-radius:3px;padding:2px 5px;font-weight:700;font-size:11px;min-width:16px;}}
  .d{{display:inline-block;background:var(--d-bg);color:var(--d-fg);border-radius:3px;padding:2px 5px;font-weight:700;font-size:11px;min-width:16px;}}
  .n{{color:var(--border2);font-size:11px;font-weight:400;}}
  .bg-g{{background:var(--g-bg);}}
  .bg-d{{background:var(--d-bg);}}
  .bg-g .n{{color:var(--g-fg);opacity:.4;}}
  .bg-d .n{{color:var(--d-fg);opacity:.4;}}
  .dot-g{{color:#22c55e;font-size:14px;font-weight:700;}}
  .dot-d{{color:#ef4444;font-size:14px;font-weight:700;}}
  .rsi{{font-size:11px;color:var(--text3);font-weight:700;}}
  .rsi-hi{{background:var(--d-bg);color:var(--d-fg);}}
  .rsi-lo{{background:var(--g-bg);color:var(--g-fg);}}
  .rsi-hdr{{color:var(--gold)!important;}}
  .empty{{text-align:center;color:var(--text4);padding:14px;font-weight:400;}}
  .footer{{font-size:10px;color:var(--text4);text-align:center;margin-top:6px;}}
  .heatmap-row{{display:flex;align-items:flex-start;gap:6px;}}
  .heatmap-expand{{flex-shrink:0;width:28px;height:28px;border-radius:6px;background:var(--bg2);border:1.5px solid var(--border2);color:var(--text2);font-size:16px;font-weight:700;cursor:pointer;display:flex;align-items:center;justify-content:center;transition:all .2s;}}
  .heatmap-expand:hover{{border-color:var(--gold);color:var(--gold);}}
  .modal-overlay{{display:none;position:fixed;top:0;left:0;right:0;bottom:0;background:rgba(0,0,0,.7);z-index:1000;align-items:center;justify-content:center;padding:20px;}}
  .modal-overlay.open{{display:flex;}}
  .modal-content{{background:var(--bg2);border:1.5px solid var(--border2);border-radius:12px;max-width:95vw;max-height:90vh;display:flex;flex-direction:column;overflow:hidden;}}
  .modal-header{{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid var(--border);gap:12px;}}
  .modal-title{{font-size:12px;font-weight:700;color:var(--gold);}}
  .modal-close{{background:none;border:none;color:var(--text);font-size:24px;font-weight:700;cursor:pointer;width:32px;height:32px;border-radius:6px;display:flex;align-items:center;justify-content:center;}}
  .modal-close:hover{{background:var(--bg4);}}
  .modal-body{{overflow:auto;padding:12px;flex:1;}}
</style>
</head>
<body>
<div class="top-bar">
  <div>
    <p class="page-title">EMA Cross Report</p>
    <p class="page-sub">max {MAX_ROWS} rows · Bangkok (ICT) · Updated: {now} · <span id="cd" style="color:var(--blue);font-weight:700;">--:--</span></p>
  </div>
  <div style="display:flex;gap:6px;">
    <button class="toggle-btn" onclick="toggleGroup()">
      <span id="gl">By Interval</span>
    </button>
    <button class="toggle-btn" onclick="toggleTheme()">
      <span id="ti">🌙</span><span id="tl">Dark</span>
    </button>
  </div>
</div>
<div class="legend">
  <div class="leg"><span class="lg">G</span> Golden (Buy)</div>
  <div class="leg"><span class="ld">D</span> Death (Sell)</div>
  <div class="leg" style="color:var(--text4);font-weight:400">S=12/26 M=20/50 L=50/200 R=RSI-14</div>
</div>
{body}
<p class="footer">Auto-generated every 15 min · GitHub Actions</p>
<script>
  let groupMode='iv';
  const ivs=['15m','30m','1h','4h','1d'];
  const lbls=['S','M','L','R'];
  function openModal(name){{
    const m=document.getElementById('modal-'+name);
    if(m)m.classList.add('open');
  }}
  function closeModal(name){{
    const m=document.getElementById('modal-'+name);
    if(m)m.classList.remove('open');
  }}
  function toggleTheme(){{
    const h=document.documentElement,dark=h.getAttribute('data-theme')==='dark';
    h.setAttribute('data-theme',dark?'light':'dark');
    document.getElementById('ti').textContent=dark?'🌙':'☀️';
    document.getElementById('tl').textContent=dark?'Dark':'Light';
  }}
  function toggleGroup(){{
    groupMode=groupMode==='iv'?'ema':'iv';
    document.getElementById('gl').textContent=groupMode==='iv'?'By Interval':'By EMA';
    document.querySelectorAll('.hdr-iv').forEach(r=>r.style.display=groupMode==='iv'?'':'none');
    document.querySelectorAll('.hdr-ema').forEach(r=>r.style.display=groupMode==='ema'?'':'none');
    let order=[];
    if(groupMode==='ema'){{
      lbls.forEach(l=>ivs.forEach(v=>order.push(v+'-'+l)));
    }}else{{
      ivs.forEach(v=>lbls.forEach(l=>order.push(v+'-'+l)));
    }}
    document.querySelectorAll('tr.data').forEach(row=>{{
      const cells=Array.from(row.querySelectorAll('td[data-iv]'));
      const map={{}};
      cells.forEach(c=>map[c.dataset.iv+'-'+c.dataset.lbl]=c);
      order.forEach((k,i)=>{{
        const c=map[k];
        if(c){{
          c.classList.remove('iv-sep');
          if(groupMode==='ema'&&i%5===0&&i>0)c.classList.add('iv-sep');
          if(groupMode==='iv'&&i%4===0&&i>0)c.classList.add('iv-sep');
          row.appendChild(c);
        }}
      }});
    }});
  }}
  window.addEventListener('DOMContentLoaded',()=>{{
    document.querySelectorAll('table').forEach(tbl=>{{
      tbl.querySelectorAll('tr').forEach(row=>{{
        let off=0;
        row.querySelectorAll('.sticky').forEach(cell=>{{
          cell.style.left=off+'px';off+=cell.offsetWidth;
        }});
      }});
    }});
    function tick(){{
      const now=new Date();
      const m=now.getMinutes(),s=now.getSeconds();
      const left=((14-(m%15))*60)+(60-s);
      const mm=String(Math.floor(left/60)).padStart(2,'0');
      const ss=String(left%60).padStart(2,'0');
      document.getElementById('cd').textContent='Next: '+mm+':'+ss;
      if(left<=0)location.reload();
    }}
    tick();setInterval(tick,1000);
  }});
</script>
</body></html>"""


def event_key(asset, date_str, time_str, interval, label, cross):
    return f"{asset}|{date_str}|{time_str}|{interval}|{label}|{cross}"


def send_email(subject, body_html):
    if not GMAIL_PASSWORD:
        print("  Email SKIP: GMAIL_APP_PASSWORD not set")
        return False
    try:
        msg = MIMEText(body_html, "html")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_SENDER
        msg["To"]      = ALERT_TO
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_SENDER, GMAIL_PASSWORD)
            s.sendmail(GMAIL_SENDER, ALERT_TO, msg.as_string())
        return True
    except Exception as e:
        print(f"  Email ERROR: {e}")
        return False


def build_email_body(new_events_list, now_str, folder_id="", asset_states=None):
    # Group by asset
    by_asset = {}
    for ev in new_events_list:
        by_asset.setdefault(ev["asset"], []).append(ev)

    pages_link = "https://payotorn-droid.github.io/ema-cross-alert/"
    link_html = f'<a href="{pages_link}" style="color:#1d4ed8;font-weight:700;font-size:14px;">Open Dashboard</a><br>'

    # State badges
    state_html = ""
    if asset_states:
        badges = []
        for a, s in asset_states.items():
            badges.append(f'<span style="font-size:13px;font-weight:700;">{a}: {s}</span>')
        state_html = '<div style="margin:8px 0;">' + ' &nbsp;|&nbsp; '.join(badges) + '</div>'

    sections = ""
    for asset, events in by_asset.items():
        rows = ""
        prev_date = ""
        for ev in sorted(events, key=lambda e: (e["date"], e["time"])):
            if ev["date"] != prev_date:
                rows += f'<tr><td colspan="4" style="padding:4px 4px 1px;font-size:10px;color:#888;border-top:1px solid #eee;">{ev["date"]}</td></tr>'
                prev_date = ev["date"]
            icon  = "🟢" if ev["cross"] == "GOLDEN" else "🔴"
            color = "#166534" if ev["cross"] == "GOLDEN" else "#991b1b"
            bg    = "#dcfce7" if ev["cross"] == "GOLDEN" else "#fee2e2"
            rows += f"""<tr>
              <td style="padding:3px 4px;font-size:12px;">{ev['time']}</td>
              <td style="padding:3px 4px;font-size:12px;">{ev['interval']}·{ev['label']}</td>
              <td style="padding:3px 4px;"><span style="background:{bg};color:{color};border-radius:3px;padding:1px 5px;font-weight:700;font-size:11px;">{icon}{ev['cross'][0]}</span></td>
              <td style="padding:3px 4px;font-size:12px;text-align:right;">{fmt_price(ev['price'])}</td>
            </tr>"""

        sections += f"""
        <div style="margin-bottom:12px;">
          <div style="font-weight:700;color:#92400e;font-size:13px;margin-bottom:4px;">{asset} ({len(events)})</div>
          <table style="width:100%;border-collapse:collapse;">
            <tr style="color:#888;font-size:10px;">
              <th style="text-align:left;padding:2px 4px;">Time</th>
              <th style="text-align:left;padding:2px 4px;">Iv·EMA</th>
              <th style="text-align:left;padding:2px 4px;">Sig</th>
              <th style="text-align:right;padding:2px 4px;">Price</th>
            </tr>
            {rows}
          </table>
        </div>"""

    return f"""<html><body style="font-family:Arial,sans-serif;margin:0;padding:8px;background:#f5f5f0;">
  <div style="max-width:360px;margin:auto;background:#fff;border-radius:8px;border-left:4px solid #f5d06e;padding:12px;">
    <div style="font-size:14px;font-weight:700;color:#92400e;margin-bottom:2px;">⚡ EMA Cross Alert</div>
    <div style="font-size:11px;color:#888;margin-bottom:8px;">{now_str}</div>
    {link_html}
    {state_html}
    {sections}
    <div style="font-size:10px;color:#aaa;margin-top:8px;">Auto-generated · GitHub Actions</div>
  </div>
</body></html>"""


def upload_to_drive(filepath, folder_id):
    """Upload HTML file to Google Drive using Service Account."""
    sa_key = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
    if not sa_key:
        print("  Drive SKIP: GOOGLE_SERVICE_ACCOUNT_KEY not set")
        return False
    try:
        from google.oauth2 import service_account
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload

        creds = service_account.Credentials.from_service_account_info(
            json.loads(sa_key),
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build("drive", "v3", credentials=creds)
        filename = os.path.basename(filepath)

        # Check if file already exists in folder
        results = service.files().list(
            q=f"name='{filename}' and '{folder_id}' in parents and trashed=false",
            fields="files(id)"
        ).execute()
        existing = results.get("files", [])

        media = MediaFileUpload(filepath, mimetype="text/html")
        if existing:
            service.files().update(
                fileId=existing[0]["id"],
                media_body=media
            ).execute()
            print(f"  Drive updated: {filename}")
        else:
            file_metadata = {"name": filename, "parents": [folder_id]}
            service.files().create(
                body=file_metadata,
                media_body=media
            ).execute()
            print(f"  Drive created: {filename}")
        return True
    except Exception as e:
        print(f"  Drive ERROR: {e}")
        return False


# Label mapping
LABEL_FULL = {"S": "Short 12/26", "M": "Mid 20/50", "L": "Long 50/200"}


def analyze_market_state(all_events, rsi_data=None):
    """Analyze all EMA crosses to classify market state for email subject."""
    cs = {}
    all_sorted = sorted(all_events.keys())
    for (d, t) in all_sorted:
        ev = all_events[(d, t)]
        for iv in INTERVALS:
            for _, _, lbl in EMA_PAIRS:
                cross = ev["crosses"].get(iv, {}).get(lbl)
                if cross:
                    cs[(iv, lbl)] = cross

    def sig(iv, lbl):
        return cs.get((iv, lbl))

    big   = [sig("1d","S"), sig("1d","M"), sig("1d","L"), sig("4h","S"), sig("4h","M"), sig("4h","L")]
    mid   = [sig("1h","S"), sig("1h","M"), sig("1h","L")]
    small = [sig("15m","S"), sig("15m","M"), sig("15m","L"), sig("30m","S"), sig("30m","M"), sig("30m","L")]

    big_g   = sum(1 for s in big if s == "GOLDEN")
    big_d   = sum(1 for s in big if s == "DEATH")
    mid_g   = sum(1 for s in mid if s == "GOLDEN")
    mid_d   = sum(1 for s in mid if s == "DEATH")
    small_g = sum(1 for s in small if s == "GOLDEN")
    small_d = sum(1 for s in small if s == "DEATH")

    rsi_ctx = ""
    if rsi_data:
        latest_key = all_sorted[-1] if all_sorted else None
        if latest_key:
            rsi_row = lookup_rsi(rsi_data, latest_key[0], latest_key[1])
            for iv in ["4h", "1d"]:
                v = rsi_row.get(iv)
                if v and v >= 70:
                    rsi_ctx = f" RSI {iv}={int(v)}"
                    break
                elif v and v <= 30:
                    rsi_ctx = f" RSI {iv}={int(v)}"
                    break

    if sig("1d", "L") == "DEATH":
        latest_1d_l = None
        for (d, t) in reversed(all_sorted):
            if all_events[(d, t)]["crosses"].get("1d", {}).get("L") == "DEATH":
                latest_1d_l = (d, t)
                break
        if latest_1d_l and latest_1d_l == all_sorted[-1]:
            return f"🔴 1d Death Cross 50/200{rsi_ctx}"

    if sig("1d", "L") == "GOLDEN":
        latest_1d_l = None
        for (d, t) in reversed(all_sorted):
            if all_events[(d, t)]["crosses"].get("1d", {}).get("L") == "GOLDEN":
                latest_1d_l = (d, t)
                break
        if latest_1d_l and latest_1d_l == all_sorted[-1]:
            return f"🟢 1d Golden Cross 50/200{rsi_ctx}"

    if big_g >= 5 and mid_g >= 2:
        return f"🟢 Full Bull{rsi_ctx}"
    if big_d >= 5 and mid_d >= 2:
        return f"🔴 Full Bear{rsi_ctx}"
    if big_g >= 4 and small_g >= 3:
        return f"🟢 Bull Wave{rsi_ctx}"
    if big_d >= 4 and small_d >= 3:
        return f"🔴 Bear Wave{rsi_ctx}"
    if small_g >= 4 and big_d >= 4:
        return f"⚠️ Divergence: Short↑ Long↓{rsi_ctx}"
    if small_d >= 4 and big_g >= 4:
        return f"⚠️ Divergence: Short↓ Long↑{rsi_ctx}"
    if big_g >= 3 and mid_d >= 2 and small_d >= 3:
        return f"⚠️ Momentum Fading{rsi_ctx}"
    if big_d >= 3 and mid_g >= 2 and small_g >= 3:
        return f"⚠️ Recovering{rsi_ctx}"

    total_g = big_g + mid_g + small_g
    total_d = big_d + mid_d + small_d
    if total_g > total_d:
        return f"🟡 Lean Bull{rsi_ctx}"
    elif total_d > total_g:
        return f"🟡 Lean Bear{rsi_ctx}"
    return f"🟡 Mixed{rsi_ctx}"


def find_last_4h_cross(all_events):
    """Find timestamp of latest 4h interval cross event."""
    last_ts = None
    for (date_str, time_str), ev in all_events.items():
        if "4h" in ev["crosses"]:
            ts = pd.Timestamp(f"{date_str} {time_str}")
            if last_ts is None or ts > last_ts:
                last_ts = ts
    return last_ts


def find_last_1d_cross(all_events):
    """Find timestamp of latest 1d interval cross event."""
    last_ts = None
    for (date_str, time_str), ev in all_events.items():
        if "1d" in ev["crosses"]:
            ts = pd.Timestamp(f"{date_str} {time_str}")
            if last_ts is None or ts > last_ts:
                last_ts = ts
    return last_ts

# ── Main ─────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
state = load_state()
now   = datetime.now()
now_str = now.strftime("%Y-%m-%d %H:%M:%S")

print(f"[{now_str}] report.py starting...")

# 1. Collect events from CSV
sections      = []
all_new_evs   = []
asset_states  = {}

for asset_name in ASSETS:
    all_events = collect_events(asset_name)
    rsi_data   = collect_rsi(asset_name)
    sections.append(build_table_html(asset_name, all_events, rsi_data))
    asset_states[asset_name] = analyze_market_state(all_events, rsi_data)
    print(f"  {asset_name}: {asset_states[asset_name]}")

    # Find cutoff = last 4h cross
    last_4h = find_last_4h_cross(all_events)
    email_cutoff_dt = last_4h if last_4h else now - timedelta(hours=4)
    for (date_str, time_str), ev in all_events.items():
        ts_event = pd.Timestamp(f"{date_str} {time_str}")
        if ts_event < email_cutoff_dt:
            continue
        for interval, pairs in ev["crosses"].items():
            for label, cross in pairs.items():
                ek = event_key(asset_name, date_str, time_str, interval, label, cross)
                ev_data = {
                    "asset":      asset_name,
                    "date":       date_str,
                    "time":       time_str,
                    "interval":   interval,
                    "label":      label,
                    "label_full": LABEL_FULL.get(label, label),
                    "cross":      cross,
                    "price":      ev["price"],
                    "event_key":  ek,
                }
                all_new_evs.append(ev_data)

# 2. Generate HTML
html = build_html(sections)
with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"  HTML saved: {OUTPUT_HTML}")

# 3. Upload to Google Drive
DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "")
if DRIVE_FOLDER_ID:
    upload_to_drive(OUTPUT_HTML, DRIVE_FOLDER_ID)

# 4. Send email if new events + cooldown OK
# Split: truly_new triggers email, all_new_evs shown in content
truly_new = [e for e in all_new_evs if e["event_key"] not in state["sent_events"]]

if truly_new:
    last_email = state.get("last_email")
    can_send   = True
    if last_email:
        elapsed = (now - datetime.fromisoformat(last_email)).total_seconds() / 60
        if elapsed < MIN_EMAIL_GAP:
            can_send = False
            print(f"  Email cooldown: {MIN_EMAIL_GAP - int(elapsed)} min remaining")

    if can_send:
        state_parts = [f"{a}: {s}" for a, s in asset_states.items()]
        subject = f"⚡ {' | '.join(state_parts)} — {len(truly_new)} new"
        # Email body shows ALL events from 4h cutoff, not just new
        body_html = build_email_body(all_new_evs, now_str, DRIVE_FOLDER_ID, asset_states)
        if send_email(subject, body_html):
            state["last_email"] = now.isoformat()
            print(f"  Email sent: {len(truly_new)} new, {len(all_new_evs)} total shown")

    # Mark only truly new as sent
    for ev in truly_new:
        state["sent_events"].append(ev["event_key"])

    state["sent_events"] = state["sent_events"][-2000:]
else:
    print("  No new events.")

save_state(state)
print(f"[{datetime.now().strftime('%H:%M:%S')}] report.py done.\n")
