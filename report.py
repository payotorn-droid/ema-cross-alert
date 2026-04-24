import pandas as pd
import os
import json
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_HTML = os.path.join(DATA_DIR, "ema_cross_report.html")
STATE_FILE = os.path.join(DATA_DIR, "report_state.json")
GMAIL_SENDER = "payotorn@gmail.com"
GMAIL_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
ALERT_TO = "payotorn@gmail.com"
MAX_ROWS = 200
LOOKBACK_DAYS = 30
MIN_EMAIL_GAP = 60
# ASSETS: all assets with data pipeline
# MAIN_ASSETS: subset shown in indicator panels, tables, and email alerts
# Assets in ASSETS but not MAIN_ASSETS appear only in the state map
ASSETS = {"Gold": "GC=F", "Bitcoin": "BTC-USD", "XAUBTC": "XAUBTC", "SPY": "SPY", "QQQ": "QQQ", "DXY": "DX-Y.NYB"}
MAIN_ASSETS = ["Gold", "Bitcoin", "XAUBTC"]
EMA_PAIRS = [(12, 26, "S"), (20, 50, "M"), (50, 200, "L")]
INTERVALS = ["15m", "30m", "1h", "4h", "1d"]
LABEL_FULL = {"S": "Short 12/26", "M": "Mid 20/50", "L": "Long 50/200"}

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f: return json.load(f)
    return {"sent_events": [], "last_email": None}

def save_state(state):
    with open(STATE_FILE, "w") as f: json.dump(state, f, indent=2)

def calc_ema(close, p): return close.ewm(span=p, adjust=False).mean()

def detect_cross(ema_fast, ema_slow, i):
    if i < 1: return None
    prev = ema_fast.iloc[i-1] - ema_slow.iloc[i-1]
    curr = ema_fast.iloc[i] - ema_slow.iloc[i]
    if prev < 0 and curr >= 0: return "GOLDEN"
    if prev > 0 and curr <= 0: return "DEATH"
    return None

def load_csv(asset_name, interval):
    fp = os.path.join(DATA_DIR, f"{asset_name}_{interval}.csv")
    if not os.path.exists(fp): return None
    return pd.read_csv(fp, index_col=0, parse_dates=True)["Close"].dropna()

def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    ag = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    al = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    return 100 - (100 / (1 + ag / al))

def collect_rsi(asset_name):
    r = {}
    for iv in INTERVALS:
        c = load_csv(asset_name, iv)
        if c is not None and len(c) >= 20: r[iv] = calc_rsi(c)
    return r

def lookup_rsi(rsi_series, date_str, time_str):
    result = {}
    try: ts = pd.Timestamp(f"{date_str} {time_str}")
    except: return result
    for iv, rsi in rsi_series.items():
        v = rsi.asof(ts)
        if pd.notna(v): result[iv] = round(float(v), 1)
    return result

def collect_events(asset_name):
    all_events = {}
    for interval in INTERVALS:
        close = load_csv(asset_name, interval)
        if close is None: continue
        for fast, slow, label in EMA_PAIRS:
            if len(close) < slow + 10: continue
            ema_f, ema_s = calc_ema(close, fast), calc_ema(close, slow)
            for i in range(1, len(close)):
                cross = detect_cross(ema_f, ema_s, i)
                if cross:
                    ts = close.index[i]
                    ds, tm = ts.strftime("%Y-%m-%d"), ts.strftime("%H:%M")
                    key = (ds, tm)
                    if key not in all_events:
                        all_events[key] = {"price": float(close.iloc[i]), "crosses": {}}
                    all_events[key]["crosses"].setdefault(interval, {})[label] = cross
    return all_events

def cell_html(cross, iv_sep=False, last_signal=None, last_price=None, cur_price=None, iv="", lbl=""):
    classes = []
    if iv_sep: classes.append("iv-sep")
    attr = f' data-iv="{iv}" data-lbl="{lbl}"'
    if cross == "GOLDEN":
        cs = f' class="{" ".join(classes)}"' if classes else ''
        return f'<td{cs}{attr}><span class="g">G</span></td>'
    elif cross == "DEATH":
        cs = f' class="{" ".join(classes)}"' if classes else ''
        return f'<td{cs}{attr}><span class="d">D</span></td>'
    else:
        if last_signal and last_price and cur_price:
            if last_signal == "GOLDEN": classes.append("bg-g")
            elif last_signal == "DEATH": classes.append("bg-d")
            cs = f' class="{" ".join(classes)}"' if classes else ''
            if cur_price > last_price: return f'<td{cs}{attr}><span class="dot-g">●</span></td>'
            elif cur_price < last_price: return f'<td{cs}{attr}><span class="dot-d">●</span></td>'
            else: return f'<td{cs}{attr}><span class="n">—</span></td>'
        cs = f' class="{" ".join(classes)}"' if classes else ''
        return f'<td{cs}{attr}><span class="n">—</span></td>'

def fmt_price(price):
    if price >= 1000: return f"${price:,.0f}"
    elif price >= 1: return f"${price:,.2f}"
    else: return f"{price:.4f}"

def rsi_cell(val, iv_sep=False, iv=""):
    cls = "iv-sep " if iv_sep else ""
    attr = f' data-iv="{iv}" data-lbl="R"'
    if val is None: return f'<td class="{cls}rsi"{attr}><span class="n">—</span></td>'
    if val >= 70: return f'<td class="{cls}rsi rsi-hi"{attr}>{val:.0f}</td>'
    elif val <= 30: return f'<td class="{cls}rsi rsi-lo"{attr}>{val:.0f}</td>'
    else: return f'<td class="{cls}rsi"{attr}>{val:.0f}</td>'

def build_indicator_html(asset_name, all_events, display_keys, rsi_data):
    if not display_keys: return "", ""
    prices = [all_events[k]["price"] for k in display_keys]
    p_min, p_max = min(prices), max(prices)
    frames = []
    for k in display_keys:
        price = all_events[k]["price"]
        p_pct = ((price - p_min) / (p_max - p_min) * 100) if p_max > p_min else 50
        rr = lookup_rsi(rsi_data, k[0], k[1]) if rsi_data else {}
        frames.append({"ts": f"{k[0]} {k[1]}", "price": round(p_pct, 1), "priceVal": price,
                       "rsi": [round(rr.get(iv), 1) if rr.get(iv) is not None else None for iv in INTERVALS]})
    cur = frames[-1]
    def pc(p):
        if p < 33: return "#ef4444"
        if p < 66: return "#eab308"
        return "#22c55e"
    def rc(v):
        if v is None: return "#888"
        if v >= 70: return "#ef4444"
        if v <= 30: return "#22c55e"
        return "#eab308"
    rb = ""
    for idx, iv in enumerate(INTERVALS):
        v = cur["rsi"][idx]
        if v is None:
            rb += f'<div class="rsi-bar"><div class="rsi-track"></div><div class="rsi-lbl">{iv}</div><div class="rsi-val" id="rsi-val-{asset_name}-{idx}">—</div></div>'
            continue
        pos = max(0, min(100, v))
        mc = rc(v)
        rb += f'<div class="rsi-bar"><div class="rsi-track"><div class="rsi-marker" id="rsi-mk-{asset_name}-{idx}" style="bottom:{pos}%;background:{mc};"></div></div><div class="rsi-lbl">{iv}</div><div class="rsi-val" id="rsi-val-{asset_name}-{idx}" style="background:{mc};">{int(v)}</div></div>'
    ih = f"""
<div class="indicator-box" data-asset="{asset_name}">
  <div class="ind-timeline"><div class="ind-ts" id="ts-{asset_name}">{cur['ts']}</div><div class="ind-progress"><div class="ind-progress-fill" id="prog-{asset_name}"></div></div></div>
  <div class="ind-price"><div class="ind-price-label">Price</div><div class="ind-price-bar"><span class="ind-min">{fmt_price(p_min)}</span><div class="ind-track"><div class="ind-marker" id="price-mk-{asset_name}" style="left:{cur['price']:.0f}%;background:{pc(cur['price'])};"></div><div class="ind-cur" id="price-cur-{asset_name}" style="left:{cur['price']:.0f}%;background:{pc(cur['price'])};">{fmt_price(cur['priceVal'])}</div></div><span class="ind-max">{fmt_price(p_max)}</span></div></div>
  <div class="ind-rsi"><div class="ind-rsi-label">RSI</div><div class="ind-rsi-bars">{rb}</div></div>
</div>"""
    import json as _j
    return ih, f'<script>window.timeline_{asset_name}={_j.dumps(frames)};</script>'

def build_heatmap_html(all_events, display_keys):
    if not display_keys: return ""
    CW, CH, GAP = 12, 2, 1
    nt, ne = len(INTERVALS), len(EMA_PAIRS)
    rows = len(display_keys)
    w = nt * ne * CW + (nt - 1) * GAP
    h = rows * CH
    cs = {}
    al = sorted(all_events.keys())
    for k in [k for k in al if k not in display_keys]:
        ev = all_events[k]
        for iv in INTERVALS:
            for _, _, lb in EMA_PAIRS:
                c = ev["crosses"].get(iv, {}).get(lb)
                if c: cs[(iv, lb)] = c
    rects = ""
    for ri, key in enumerate(display_keys):
        ev = all_events[key]
        y = ri * CH
        for ii, iv in enumerate(INTERVALS):
            ivd = ev["crosses"].get(iv, {})
            for li, (_, _, lb) in enumerate(EMA_PAIRS):
                x = (ii * ne + li) * CW + ii * GAP
                cr = ivd.get(lb)
                if cr:
                    cs[(iv, lb)] = cr
                    fill = "#16a34a" if cr == "GOLDEN" else "#dc2626"
                else:
                    sc = cs.get((iv, lb))
                    fill = "#dcfce7" if sc == "GOLDEN" else "#fee2e2" if sc == "DEATH" else "#f1efe8"
                rects += f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" fill="{fill}"/>'
    return f'<div class="heatmap-box"><div class="heatmap-label">HEATMAP · {rows} rows</div><svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">{rects}</svg></div>'

def build_full_heatmap_html(asset_name, all_events, years=4):
    CW, CH, GAP = 8, 1, 1
    nt, ne = len(INTERVALS), len(EMA_PAIRS)
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=365 * years)
    al = sorted(all_events.keys())
    dk = [k for k in al if pd.Timestamp(f"{k[0]} {k[1]}") >= cutoff]
    if not dk: return ""
    rows = len(dk)
    w = nt * ne * CW + (nt - 1) * GAP
    h = rows * CH
    cs = {}
    for k in [k for k in al if k not in dk]:
        ev = all_events[k]
        for iv in INTERVALS:
            for _, _, lb in EMA_PAIRS:
                c = ev["crosses"].get(iv, {}).get(lb)
                if c: cs[(iv, lb)] = c
    rects = ""
    for ri, key in enumerate(dk):
        ev = all_events[key]
        y = ri * CH
        for ii, iv in enumerate(INTERVALS):
            ivd = ev["crosses"].get(iv, {})
            for li, (_, _, lb) in enumerate(EMA_PAIRS):
                x = (ii * ne + li) * CW + ii * GAP
                cr = ivd.get(lb)
                if cr:
                    cs[(iv, lb)] = cr
                    fill = "#16a34a" if cr == "GOLDEN" else "#dc2626"
                else:
                    sc = cs.get((iv, lb))
                    fill = "#dcfce7" if sc == "GOLDEN" else "#fee2e2" if sc == "DEATH" else "#f1efe8"
                rects += f'<rect x="{x}" y="{y}" width="{CW}" height="{CH}" fill="{fill}"/>'
    fd, ld = dk[0][0], dk[-1][0]
    return f"""<div class="modal-overlay" id="modal-{asset_name}" onclick="if(event.target===this)closeModal('{asset_name}')"><div class="modal-content"><div class="modal-header"><div class="modal-title">{asset_name} · Full Heatmap · {rows} events · {fd} → {ld}</div><button class="modal-close" onclick="closeModal('{asset_name}')">×</button></div><div class="modal-body"><svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" shape-rendering="crispEdges">{rects}</svg></div></div></div>"""

def build_state_timeline_frames(asset_data, N=50):
    """Build N-frame timeline: state of all assets at each Gold cross timestamp.
    Uses Gold as master clock; queries other assets via asof-style running state.
    """
    # Precompute running cross-state snapshots per asset (tuples sorted by time)
    running = {}
    for asset in ASSETS:
        events = asset_data.get(asset, {}).get("events", {})
        snapshots = []
        cs = {}
        for key in sorted(events.keys()):
            for iv in INTERVALS:
                for _, _, lb in EMA_PAIRS:
                    cr = events[key]["crosses"].get(iv, {}).get(lb)
                    if cr: cs[(iv, lb)] = "G" if cr == "GOLDEN" else "D"
            snapshots.append((pd.Timestamp(f"{key[0]} {key[1]}"), dict(cs)))
        running[asset] = snapshots

    # Master timeline = last N Gold cross timestamps
    gold_events = asset_data.get("Gold", {}).get("events", {})
    sorted_gold = sorted(gold_events.keys())
    if not sorted_gold: return []
    pivots = sorted_gold[-N:]

    frames = []
    for (d, t) in pivots:
        ts = pd.Timestamp(f"{d} {t}")
        frame = {"ts": f"{d} {t}"}
        for asset in ASSETS:
            # Find last snapshot with ts <= pivot_ts
            snaps = running[asset]
            cs = {}
            for s_ts, s_cs in snaps:
                if s_ts > ts: break
                cs = s_cs
            # RSI lookup at ts
            rsi_data = asset_data.get(asset, {}).get("rsi", {})
            rr = {}
            for iv, rsi in rsi_data.items():
                v = rsi.asof(ts)
                if pd.notna(v): rr[iv] = round(float(v), 0)
            # Build state dict for this asset at this frame
            td = {}
            for iv in INTERVALS:
                td[iv] = {
                    "S": cs.get((iv, "S"), "G"),
                    "M": cs.get((iv, "M"), "G"),
                    "L": cs.get((iv, "L"), "G"),
                    "rsi": rr.get(iv, 50),
                }
            frame[asset] = td
        frames.append(frame)
    return frames

def build_state_map_html(frames):
    import json as _j
    controls = '''<div class="state-map-controls">
  <button class="sm-btn" id="smStepBack10" title="-10 frames">⏮</button>
  <button class="sm-btn" id="smStepBack" title="-1 frame">◀</button>
  <button class="sm-btn sm-play" id="smPlayBtn" title="Play/Pause (Space)">⏸</button>
  <button class="sm-btn" id="smStepFwd" title="+1 frame">▶</button>
  <button class="sm-btn" id="smStepFwd10" title="+10 frames">⏭</button>
  <span class="sm-frame-count" id="stateMapFrameCount">1 / 0</span>
  <span class="sm-speed-sep">·</span>
  <span class="sm-speed-label">Speed</span>
  <button class="sm-speed-btn active" data-speed="1">1x</button>
  <button class="sm-speed-btn" data-speed="0.5">0.5x</button>
  <button class="sm-speed-btn" data-speed="0.25">0.25x</button>
  <button class="sm-speed-btn" data-speed="0.1">0.1x</button>
  <button class="sm-speed-btn" data-speed="0.05">0.05x</button>
</div>'''
    legend = '''<div class="state-map-legend">
  <button class="sm-legend-pill active" data-asset="Gold">
    <span class="sm-legend-swatch sm-sw-gold"></span>Gold
  </button>
  <button class="sm-legend-pill active" data-asset="Bitcoin">
    <span class="sm-legend-swatch sm-sw-btc">₿</span>Bitcoin
  </button>
  <button class="sm-legend-pill active" data-asset="XAUBTC">
    <span class="sm-legend-swatch sm-sw-xb">X/B</span>XAU/BTC
  </button>
  <button class="sm-legend-pill active" data-asset="SPY">
    <span class="sm-legend-swatch sm-sw-spy"></span>SPY
  </button>
  <button class="sm-legend-pill active" data-asset="QQQ">
    <span class="sm-legend-swatch sm-sw-qqq">Q</span>QQQ
  </button>
  <button class="sm-legend-pill active" data-asset="DXY">
    <span class="sm-legend-swatch sm-sw-dxy">$</span>DXY
  </button>
</div>'''
    if not frames:
        return f'<div class="state-map-box"><div class="state-map-header"><div class="state-map-label">STATE MAP</div><div class="state-map-ts" id="stateMapTs">—</div></div><canvas id="stateMap" width="540" height="320" style="max-width:100%;"></canvas>{legend}<div class="state-map-progress"><div class="state-map-progress-fill" id="stateMapProg"></div></div>{controls}</div><script>window._stateMapFrames=[];</script>'
    latest_ts = frames[-1]["ts"]
    return f'<div class="state-map-box"><div class="state-map-header"><div class="state-map-label">STATE MAP</div><div class="state-map-ts" id="stateMapTs">{latest_ts}</div></div><canvas id="stateMap" width="540" height="320" style="max-width:100%;"></canvas>{legend}<div class="state-map-progress"><div class="state-map-progress-fill" id="stateMapProg"></div></div>{controls}</div><script>window._stateMapFrames={_j.dumps(frames)};</script>'

def build_table_html(asset_name, all_events, rsi_data=None):
    ne = len(EMA_PAIRS)
    ns = ne + 1
    tc = 2 + len(INTERVALS) * ns
    al = sorted(all_events.keys())
    dk = al[-MAX_ROWS:]
    cs, cp = {}, {}
    for k in [k for k in al if k not in dk]:
        ev = all_events[k]
        for iv in INTERVALS:
            for _, _, lb in EMA_PAIRS:
                c = ev["crosses"].get(iv, {}).get(lb)
                if c: cs[(iv, lb)] = c; cp[(iv, lb)] = ev["price"]
    ih, ts = build_indicator_html(asset_name, all_events, dk, rsi_data)
    hm = build_heatmap_html(all_events, dk)
    fh = build_full_heatmap_html(asset_name, all_events, years=4)
    sh = ""
    if dk:
        lk = dk[-1]; le = all_events[lk]; ld, lt = lk
        chips = []
        for iv in INTERVALS:
            ivd = le["crosses"].get(iv, {})
            for f, s, lb in EMA_PAIRS:
                cr = ivd.get(lb)
                if cr:
                    lf = {"S": f"Short {f}/{s}", "M": f"Mid {f}/{s}", "L": f"Long {f}/{s}"}.get(lb, lb)
                    cl = "chip-g" if cr == "GOLDEN" else "chip-d"
                    ic = "G" if cr == "GOLDEN" else "D"
                    chips.append(f'<span class="chip {cl}"><span class="chip-badge">{ic}</span>{cr.capitalize()} · {iv} · {lf}</span>')
        if chips:
            sh = f'<div class="summary-box"><span class="summary-label">Latest event &nbsp;·&nbsp; {ld} {lt} &nbsp;·&nbsp; {fmt_price(le["price"])}</span><div class="summary-chips">{chr(10).join(chips)}</div></div>'
    h1 = '<tr class="hdr-iv hdr-r1"><th rowspan="2" class="sticky s0 left th-fix">Time</th><th rowspan="2" class="sticky s1 left th-fix">Price</th>'
    for iv in INTERVALS: h1 += f'<th colspan="{ns}" class="iv-sep">{iv}</th>'
    h1 += '</tr>'
    h2 = '<tr class="hdr-iv hdr-r2">'
    for idx, iv in enumerate(INTERVALS):
        for j, (_, _, lb) in enumerate(EMA_PAIRS):
            c = ' class="iv-sep"' if j == 0 and idx > 0 else ''
            h2 += f'<th{c} data-iv="{iv}" data-lbl="{lb}">{lb}</th>'
        h2 += f'<th class="rsi-hdr" data-iv="{iv}" data-lbl="R">R</th>'
    h2 += '</tr>'
    h1b = '<tr class="hdr-ema hdr-r1" style="display:none;"><th rowspan="2" class="sticky s0 left th-fix">Time</th><th rowspan="2" class="sticky s1 left th-fix">Price</th>'
    for _, _, lb in EMA_PAIRS:
        fl = {"S":"Short","M":"Mid","L":"Long"}[lb]
        h1b += f'<th colspan="{len(INTERVALS)}" class="iv-sep">{fl}</th>'
    h1b += f'<th colspan="{len(INTERVALS)}" class="iv-sep rsi-hdr">RSI</th></tr>'
    h2b = '<tr class="hdr-ema hdr-r2" style="display:none;">'
    for j, (_, _, lb) in enumerate(EMA_PAIRS):
        for idx, iv in enumerate(INTERVALS):
            c = ' class="iv-sep"' if idx == 0 and j > 0 else ''
            h2b += f'<th{c} data-iv="{iv}" data-lbl="{lb}">{iv}</th>'
    for idx, iv in enumerate(INTERVALS):
        c = ' class="iv-sep"' if idx == 0 else ''
        h2b += f'<th{c} data-iv="{iv}" data-lbl="R">{iv}</th>'
    h2b += '</tr>'
    rh = ""
    pd_ = None
    for (ds, tm) in dk:
        if pd_ != ds:
            rh += f'<tr class="day-sep"><td class="sticky s0 day-label">{ds}</td><td colspan="{tc-1}"></td></tr>'
            pd_ = ds
        ev = all_events[(ds, tm)]
        pr = ev["price"]
        rr = lookup_rsi(rsi_data, ds, tm) if rsi_data else {}
        row = f'<tr class="data"><td class="sticky s0 left tm">{tm}</td><td class="sticky s1 left price">{fmt_price(pr)}</td>'
        for ii, iv in enumerate(INTERVALS):
            ivd = ev["crosses"].get(iv, {})
            for j, (_, _, lb) in enumerate(EMA_PAIRS):
                fg = (ii > 0 and j == 0)
                cr = ivd.get(lb)
                ck = (iv, lb)
                if cr: cs[ck] = cr; cp[ck] = pr
                row += cell_html(cr, fg, cs.get(ck), cp.get(ck), pr, iv, lb)
            row += rsi_cell(rr.get(iv), iv_sep=False, iv=iv)
        row += '</tr>'
        rh += row
    if not rh: rh = f'<tr><td colspan="{tc}" class="empty">No EMA cross events</td></tr>'
    return f"""{ts}
<div class="asset-block"><div class="asset-title">{asset_name}</div>{sh}
<div class="ind-heatmap-wrap">{ih}<div class="heatmap-row">{hm}<button class="heatmap-expand" onclick="openModal('{asset_name}')" title="View 4-year heatmap">⛶</button></div></div>{fh}
<div class="table-scroll"><table><thead>{h1}{h2}{h1b}{h2b}</thead><tbody>{rh}</tbody></table></div></div>"""

# ──────────────────────────────────────────────────────────────
# JS block — kept as a raw string (NOT an f-string) so we don't
# need to escape { and } as {{ }}. Easy to edit, no brace traps.
# If you need to inject dynamic values, concatenate or .format()
# this string. Current JS is fully static.
# ──────────────────────────────────────────────────────────────
JS_BLOCK = r"""
let groupMode='iv';
const ivs=['15m','30m','1h','4h','1d'];
const lbls=['S','M','L','R'];

function openModal(n){const m=document.getElementById('modal-'+n);if(m)m.classList.add('open');}
function closeModal(n){const m=document.getElementById('modal-'+n);if(m)m.classList.remove('open');}

function toggleTheme(){
  const h=document.documentElement,d=h.getAttribute('data-theme')==='dark';
  h.setAttribute('data-theme',d?'light':'dark');
  document.getElementById('ti').textContent=d?'🌙':'☀️';
  document.getElementById('tl').textContent=d?'Dark':'Light';
}

function toggleGroup(){
  groupMode=groupMode==='iv'?'ema':'iv';
  document.getElementById('gl').textContent=groupMode==='iv'?'By Interval':'By EMA';
  document.querySelectorAll('.hdr-iv').forEach(r=>r.style.display=groupMode==='iv'?'':'none');
  document.querySelectorAll('.hdr-ema').forEach(r=>r.style.display=groupMode==='ema'?'':'none');
  let o=[];
  if(groupMode==='ema')lbls.forEach(l=>ivs.forEach(v=>o.push(v+'-'+l)));
  else ivs.forEach(v=>lbls.forEach(l=>o.push(v+'-'+l)));
  document.querySelectorAll('tr.data').forEach(row=>{
    const cells=Array.from(row.querySelectorAll('td[data-iv]'));
    const map={};
    cells.forEach(c=>map[c.dataset.iv+'-'+c.dataset.lbl]=c);
    o.forEach((k,i)=>{
      const c=map[k];
      if(c){
        c.classList.remove('iv-sep');
        if(groupMode==='ema'&&i%5===0&&i>0)c.classList.add('iv-sep');
        if(groupMode==='iv'&&i%4===0&&i>0)c.classList.add('iv-sep');
        row.appendChild(c);
      }
    });
  });
}

function priceColor(p){if(p<33)return'#ef4444';if(p<66)return'#eab308';return'#22c55e';}
function rsiColor(v){if(v==null)return'#888';if(v>=70)return'#ef4444';if(v<=30)return'#22c55e';return'#eab308';}
function fmtPrice(p){if(p>=1000)return'$'+p.toLocaleString('en-US',{maximumFractionDigits:0});if(p>=1)return'$'+p.toFixed(2);return p.toFixed(4);}

function animateIndicator(asset){
  const tl=window['timeline_'+asset];
  if(!tl||!tl.length)return;
  const dur=8000,frameMs=dur/tl.length,pauseMs=2000;
  let i=0,paused=false;
  function step(){
    if(paused)return;
    const f=tl[i];
    const pmk=document.getElementById('price-mk-'+asset),pcur=document.getElementById('price-cur-'+asset);
    if(pmk){
      const c=priceColor(f.price);
      pmk.style.left=f.price+'%';
      pmk.style.background=c;
      if(pcur){
        pcur.style.left=f.price+'%';
        pcur.style.background=c;
        pcur.textContent=fmtPrice(f.priceVal);
      }
    }
    for(let j=0;j<5;j++){
      const v=f.rsi[j],mk=document.getElementById('rsi-mk-'+asset+'-'+j),vl=document.getElementById('rsi-val-'+asset+'-'+j);
      if(v==null){if(vl)vl.textContent='—';continue;}
      const c=rsiColor(v);
      if(mk){mk.style.bottom=v+'%';mk.style.background=c;}
      if(vl){vl.textContent=Math.round(v);vl.style.background=c;}
    }
    const ts=document.getElementById('ts-'+asset);
    if(ts)ts.textContent=f.ts;
    const prog=document.getElementById('prog-'+asset);
    if(prog)prog.style.width=((i+1)/tl.length*100)+'%';
    if(i===tl.length-1){
      paused=true;
      setTimeout(()=>{i=0;paused=false;step();},pauseMs);
    }else{
      i++;
      setTimeout(step,frameMs);
    }
  }
  step();
}

// Row heights: [LGG-SG, LGG-SD, LGD-SG, LGD-SD, LDG-SG, LDG-SD, LDD-SG, LDD-SD]
// Order: row index r8 = (L==D)*4 + (M==D)*2 + (S==D)
// Row heights per user formula: S = 4*1/3 | 4*2/3*2/5 | 4*2/3*2/5 | 4*2/3*1/5 | ... (sum=8)
// In fractions of 15: [20, 16, 16, 8, 8, 16, 16, 20]/15
const STATE_ROW_H=[20/15,16/15,16/15,8/15,8/15,16/15,16/15,20/15];  // sum=8
const STATE_TOTAL=8;

// Per-asset visibility toggle. Updated by HTML legend click handlers.
// Animation always advances; drawing skips assets that are off.
// Re-enabling shows trail continuously (no reset) since we always look up
// past frames from the global frame array.
window._smAssetOn=window._smAssetOn||{Gold:true,Bitcoin:true,XAUBTC:true,SPY:true,QQQ:true,DXY:true};

function drawStateMap(data,idx){
  if(!data)return;
  const cv=document.getElementById('stateMap');
  if(!cv)return;
  const ctx=cv.getContext('2d');
  ctx.clearRect(0,0,cv.width,cv.height);
  const tfs=['15m','30m','1h','4h','1d'],emas=['S','M','L'];
  const cW=28,gap=16,tW=3*cW,tH=240,tP=55,sX=14,GL='#dcfce7',RL='#fee2e2';

  // Compute y-center of row r8 in a single timeframe grid
  function rowCY(r8){
    let acc=0;
    for(let r=0;r<r8;r++)acc+=STATE_ROW_H[r];
    return tP+(acc+STATE_ROW_H[r8]/2)/STATE_TOTAL*tH;
  }

  function s2r(s,m,l){
    let r=0;
    if(l==='D')r+=4;
    if(m==='D')r+=2;
    if(s==='D')r+=1;
    return r;
  }

  // Column-based cell fill: each of 3 columns shows green/red bands
  // following the row heights of that column's partition level.
  // S column: 8 alternating bands (using STATE_ROW_H directly).
  // M column: 4 bands (each = sum of 2 S-rows sharing same L,M).
  // L column: 2 bands (each = sum of 4 rows).
  function fillColumn(tx,col){
    const x=tx+col*cW;
    let y=tP;
    if(col===0){
      // S: 8 bands of varying heights, row pattern S: G,D,G,D,G,D,G,D
      for(let r=0;r<8;r++){
        const h=STATE_ROW_H[r]/STATE_TOTAL*tH;
        ctx.fillStyle=(r%2===0)?GL:RL;
        ctx.fillRect(x,y,cW,h);
        y+=h;
      }
    }else if(col===1){
      // M: 4 bands, pair of rows (r, r+1) share M
      // M pattern: G,D,G,D (rows 0-1,2-3,4-5,6-7)
      for(let p=0;p<4;p++){
        const h=(STATE_ROW_H[p*2]+STATE_ROW_H[p*2+1])/STATE_TOTAL*tH;
        ctx.fillStyle=(p%2===0)?GL:RL;
        ctx.fillRect(x,y,cW,h);
        y+=h;
      }
    }else{
      // L: 2 bands
      let hTop=0;for(let r=0;r<4;r++)hTop+=STATE_ROW_H[r];
      hTop=hTop/STATE_TOTAL*tH;
      ctx.fillStyle=GL;
      ctx.fillRect(x,y,cW,hTop);
      ctx.fillStyle=RL;
      ctx.fillRect(x,y+hTop,cW,tH-hTop);
    }
  }

  function dGB(cx,cy,w,h){
    const tw=w*0.6;
    ctx.beginPath();
    ctx.moveTo(cx-tw/2,cy-h/2);
    ctx.lineTo(cx+tw/2,cy-h/2);
    ctx.lineTo(cx+w/2,cy+h/2);
    ctx.lineTo(cx-w/2,cy+h/2);
    ctx.closePath();
    const g=ctx.createLinearGradient(cx-w/2,cy-h/2,cx+w/2,cy+h/2);
    g.addColorStop(0,'#fde68a');
    g.addColorStop(0.4,'#f59e0b');
    g.addColorStop(0.7,'#d97706');
    g.addColorStop(1,'#92400e');
    ctx.fillStyle=g;
    ctx.fill();
    ctx.strokeStyle='#78350f';
    ctx.lineWidth=0.8;
    ctx.stroke();
  }

  for(let t=0;t<5;t++){
    const tx=sX+t*(tW+gap),tf=tfs[t];

    ctx.font='700 11px sans-serif';
    ctx.fillStyle='#1d4ed8';
    ctx.textAlign='center';
    ctx.fillText(tf,tx+tW/2,16);
    ctx.font='500 9px sans-serif';
    ctx.fillStyle='#888';
    for(let i=0;i<3;i++)ctx.fillText(emas[i],tx+i*cW+cW/2,34);

    // Fill each column with its partition pattern
    for(let col=0;col<3;col++)fillColumn(tx,col);

    // Horizontal dividers between row-partitions (only S column has 8, M has 4, L has 2)
    ctx.strokeStyle='#ccc';
    ctx.lineWidth=0.5;
    // S column: 7 dividers at cumulative row boundaries
    {
      const x=tx;
      let acc=0;
      for(let r=1;r<8;r++){
        acc+=STATE_ROW_H[r-1];
        const y=tP+acc/STATE_TOTAL*tH;
        ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+cW,y);ctx.stroke();
      }
    }
    // M column: 3 dividers
    {
      const x=tx+cW;
      let acc=0;
      for(let p=1;p<4;p++){
        acc+=STATE_ROW_H[(p-1)*2]+STATE_ROW_H[(p-1)*2+1];
        const y=tP+acc/STATE_TOTAL*tH;
        ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+cW,y);ctx.stroke();
      }
    }
    // L column: 1 divider (middle)
    {
      const x=tx+2*cW;
      let acc=0;for(let r=0;r<4;r++)acc+=STATE_ROW_H[r];
      const y=tP+acc/STATE_TOTAL*tH;
      ctx.beginPath();ctx.moveTo(x,y);ctx.lineTo(x+cW,y);ctx.stroke();
    }

    // Vertical column dividers
    ctx.strokeStyle='#ccc';
    ctx.lineWidth=0.5;
    for(let col=1;col<3;col++){
      ctx.beginPath();
      ctx.moveTo(tx+col*cW,tP);
      ctx.lineTo(tx+col*cW,tP+tH);
      ctx.stroke();
    }

    // Outer border
    ctx.strokeStyle='#aaa';
    ctx.lineWidth=1;
    ctx.strokeRect(tx,tP,tW,tH);

    ctx.font='400 8px sans-serif';
    ctx.fillStyle='#aaa';
    ctx.textAlign='left';
    ctx.fillText('30',tx+2,tP+tH+12);
    ctx.textAlign='right';
    ctx.fillText('70',tx+tW-2,tP+tH+12);
    ctx.textAlign='center';
    ctx.fillText('RSI',tx+tW/2,tP+tH+12);

    const items=[['Gold','gold','#92400e'],['Bitcoin','btc','#fbbf24'],['XAUBTC','xb','#6366f1'],['SPY','spy','#991b1b'],['QQQ','qqq','#1a6fc9'],['DXY','dxy','#15803d']];

    // Draw trails (meteor-tail). Base 3 tapered segments for all TFs:
    //   n-1 → n (75% width, alpha1),  n-2 → n-1 (50%, alpha2),  n-3 → n-2 (50%, alpha3)
    // Higher TFs add constant-spec extensions equal to 3rd-segment spec:
    //   1h adds 1 seg  (n-4 → n-3)
    //   4h adds 2 segs (n-5 → n-4, n-4 → n-3)
    //   1d adds 3 segs (n-6 → n-5, n-5 → n-4, n-4 → n-3)
    // Per-asset palette: [color, alpha_n1n, alpha_n2n1, alpha_n3n2_and_ext]
    //   Gold    #92400e  80/40/26 (50/25/15%)
    //   Bitcoin #d97706  90/60/4d (57/38/30%)
    //   XAUBTC  #6366f1  80/40/26 (50/25/15%)
    //   SPY     #991b1b  80/40/26 (50/25/15%)
    //   QQQ     #1a6fc9  80/40/26 (50/25/15%)
    //   DXY     #15803d  80/40/26 (50/25/15%)
    const framesArr=window._stateMapFrames||[];
    if(typeof idx==='number'&&idx>=1){
      function posAt(fi,aN){
        if(fi<0||fi>=framesArr.length)return null;
        const f=framesArr[fi];
        if(!f||!f[aN]||!f[aN][tf])return null;
        const st=f[aN][tf],r=s2r(st.S,st.M,st.L);
        const rv=Math.max(30,Math.min(70,st.rsi));
        return {ix:tx+((rv-30)/40)*tW,iy:rowCY(r)};
      }
      const ICON=14;
      const TRAIL={
        Gold:    ['#92400e','80','40','26'],
        Bitcoin: ['#d97706','90','60','4d'],
        XAUBTC:  ['#6366f1','80','40','26'],
        SPY:     ['#991b1b','80','40','26'],
        QQQ:     ['#1a6fc9','80','40','26'],
        DXY:     ['#15803d','80','40','26'],
      };
      // How many extra extension segments per TF (beyond the 3 base)
      // 1h adds 1, 4h adds 50, 1d adds 200 (full timeline worth of trail)
      const TF_EXT={'15m':0,'30m':0,'1h':1,'4h':50,'1d':200};
      function drawSeg(a,b,clr,alphaHex,widthFrac){
        if(!a||!b)return;
        ctx.strokeStyle=clr+alphaHex;
        ctx.lineWidth=ICON*widthFrac;
        ctx.lineCap='round';
        ctx.beginPath();
        ctx.moveTo(a.ix,a.iy);
        ctx.lineTo(b.ix,b.iy);
        ctx.stroke();
      }
      const nExt=TF_EXT[tf]||0;
      const totalSegs=3+nExt;  // max look-back = idx-totalSegs
      for(const[aN] of items){
        if(!window._smAssetOn[aN])continue;  // Hidden asset: skip trail
        const tr=TRAIL[aN];if(!tr)continue;
        const clr=tr[0];
        // Cache positions for all needed frames
        const pos=[];for(let k=0;k<=totalSegs;k++)pos.push(posAt(idx-k,aN));
        // Draw oldest → newest so newer (thicker+opaque) segments render on top.
        // Extensions first (all use 3rd-segment spec: width 0.50, alpha tr[3])
        for(let k=totalSegs;k>=4;k--){
          if(idx>=k)drawSeg(pos[k],pos[k-1],clr,tr[3],0.50);
        }
        // Then 3 base segments
        if(idx>=3)drawSeg(pos[3],pos[2],clr,tr[3],0.50);  // n-3 → n-2
        if(idx>=2)drawSeg(pos[2],pos[1],clr,tr[2],0.50);  // n-2 → n-1
        drawSeg(pos[1],pos[0],clr,tr[1],0.75);            // n-1 → n
      }
      ctx.lineWidth=1;
    }

    for(const[aN,aT] of items){
      if(!window._smAssetOn[aN])continue;  // Hidden asset: skip icon
      const d=data[aN];
      if(!d||!d[tf])continue;
      const s=d[tf],r8=s2r(s.S,s.M,s.L),iy=rowCY(r8),rsi=Math.max(30,Math.min(70,s.rsi)),ix=tx+((rsi-30)/40)*tW;
      if(aT==='gold')dGB(ix,iy,14,9);
      else if(aT==='spy'){
        // SPDR-inspired concentric rings: red outer, white ring, red core dot
        ctx.beginPath();ctx.arc(ix,iy,7,0,Math.PI*2);
        ctx.fillStyle='#dc2626';ctx.fill();
        ctx.strokeStyle='#7f1d1d';ctx.lineWidth=0.8;ctx.stroke();
        ctx.beginPath();ctx.arc(ix,iy,4.5,0,Math.PI*2);
        ctx.strokeStyle='#fff';ctx.lineWidth=1.3;ctx.stroke();
        ctx.beginPath();ctx.arc(ix,iy,2,0,Math.PI*2);
        ctx.fillStyle='#fff';ctx.fill();
        ctx.strokeStyle='#7f1d1d';ctx.lineWidth=0.5;ctx.stroke();
      }
      else{
        // Generic filled-circle icon with single-char label (BTC/XAUBTC/QQQ/DXY)
        let clr,lbl,fs;
        if(aT==='btc'){clr='#f7931a';lbl='₿';fs='700 8px sans-serif';}
        else if(aT==='qqq'){clr='#1a6fc9';lbl='Q';fs='700 8px sans-serif';}
        else if(aT==='dxy'){clr='#15803d';lbl='$';fs='700 8px sans-serif';}
        else{clr='#6366f1';lbl='X/B';fs='700 5px sans-serif';}
        ctx.beginPath();
        ctx.arc(ix,iy,7,0,Math.PI*2);
        ctx.fillStyle=clr;
        ctx.fill();
        ctx.strokeStyle='#fff';
        ctx.lineWidth=1.5;
        ctx.stroke();
        ctx.strokeStyle='#333';
        ctx.lineWidth=0.5;
        ctx.stroke();
        ctx.font=fs;
        ctx.fillStyle='#fff';
        ctx.textAlign='center';
        ctx.textBaseline='middle';
        ctx.fillText(lbl,ix,iy);
      }
    }
  }

}

// ───── State map video-player controller ─────
// Single master animation: all TFs advance together. User controls via UI:
// play/pause, step ±1/±10, speed select. End of cycle = 2 sec pause then loop.
const SPEED_OPTIONS=[
  {label:'1x',   val:1.0},
  {label:'0.5x', val:0.5},
  {label:'0.25x',val:0.25},
  {label:'0.1x', val:0.1},
  {label:'0.05x',val:0.05},
];
const BASE_FRAME_MS=40;  // 1x speed = 40ms/frame (25 fps)
const END_PAUSE_MS=2000; // pause at end of cycle before restarting

window._smIdx=0;          // current frame index
window._smPlaying=true;    // play state
window._smSpeed=1.0;       // speed multiplier
window._smLastTick=0;      // timestamp of last frame advance
window._smPauseUntil=0;    // if >0, waiting at end-of-cycle until this timestamp

function renderStateMapFrame(idx){
  const frames=window._stateMapFrames;
  if(!frames||!frames.length)return;
  const f=frames[idx];
  // Build payload: drop 'ts' key, pass {asset: {tf: {...}}}
  const payload={};
  for(const k of Object.keys(f)){if(k!=='ts')payload[k]=f[k];}
  drawStateMap(payload,idx);
  const ts=document.getElementById('stateMapTs');
  if(ts)ts.textContent=f.ts;
  const prog=document.getElementById('stateMapProg');
  if(prog)prog.style.width=((idx+1)/frames.length*100)+'%';
  const fc=document.getElementById('stateMapFrameCount');
  if(fc)fc.textContent=(idx+1)+' / '+frames.length;
}

function animateStateMap(){
  const frames=window._stateMapFrames;
  if(!frames||!frames.length)return;
  const N=frames.length;

  function step(now){
    if(window._smPlaying){
      if(window._smPauseUntil>0){
        // End-of-cycle pause
        if(now>=window._smPauseUntil){
          window._smPauseUntil=0;
          window._smIdx=0;
          window._smLastTick=now;
          renderStateMapFrame(0);
        }
      }else{
        const frameMs=BASE_FRAME_MS/window._smSpeed;
        if(!window._smLastTick)window._smLastTick=now;
        while(now-window._smLastTick>=frameMs&&window._smPauseUntil===0){
          window._smIdx++;
          window._smLastTick+=frameMs;
          if(window._smIdx>=N){
            // Reached end — show last frame, trigger 2s pause
            window._smIdx=N-1;
            renderStateMapFrame(N-1);
            window._smPauseUntil=now+END_PAUSE_MS;
            break;
          }
          renderStateMapFrame(window._smIdx);
        }
      }
    }
    requestAnimationFrame(step);
  }
  // Initial render + start loop
  renderStateMapFrame(0);
  requestAnimationFrame(step);
}

function smStep(delta){
  const frames=window._stateMapFrames;
  if(!frames||!frames.length)return;
  const N=frames.length;
  window._smPlaying=false;
  window._smPauseUntil=0;
  window._smIdx=Math.max(0,Math.min(N-1,window._smIdx+delta));
  renderStateMapFrame(window._smIdx);
  updatePlayBtn();
}

function smTogglePlay(){
  window._smPlaying=!window._smPlaying;
  if(window._smPlaying){
    window._smLastTick=performance.now();
    // If at last frame, restart from beginning
    const N=(window._stateMapFrames||[]).length;
    if(window._smIdx>=N-1){
      window._smIdx=0;
      window._smPauseUntil=0;
      renderStateMapFrame(0);
    }
  }
  updatePlayBtn();
}

function updatePlayBtn(){
  const b=document.getElementById('smPlayBtn');
  if(b)b.textContent=window._smPlaying?'⏸':'▶';
}

function smSetSpeed(v){
  window._smSpeed=v;
  window._smLastTick=performance.now();
  document.querySelectorAll('.sm-speed-btn').forEach(btn=>{
    btn.classList.toggle('active',parseFloat(btn.dataset.speed)===v);
  });
}

function setupStateMapControls(){
  const b=id=>document.getElementById(id);
  const p=b('smPlayBtn');     if(p)p.addEventListener('click',smTogglePlay);
  const s1=b('smStepBack');   if(s1)s1.addEventListener('click',()=>smStep(-1));
  const s2=b('smStepFwd');    if(s2)s2.addEventListener('click',()=>smStep(1));
  const s10=b('smStepBack10');if(s10)s10.addEventListener('click',()=>smStep(-10));
  const sf10=b('smStepFwd10');if(sf10)sf10.addEventListener('click',()=>smStep(10));
  document.querySelectorAll('.sm-speed-btn').forEach(btn=>{
    btn.addEventListener('click',()=>smSetSpeed(parseFloat(btn.dataset.speed)));
  });
  // Legend toggle pills
  document.querySelectorAll('.sm-legend-pill').forEach(btn=>{
    btn.addEventListener('click',()=>{
      const a=btn.dataset.asset;
      window._smAssetOn[a]=!window._smAssetOn[a];
      btn.classList.toggle('active',window._smAssetOn[a]);
      // Redraw current frame immediately so change is visible even when paused
      renderStateMapFrame(window._smIdx);
    });
  });
  // Keyboard shortcuts — only when state map is in viewport
  document.addEventListener('keydown',e=>{
    // Ignore when typing in inputs/textareas
    if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
    if(e.code==='Space'){e.preventDefault();smTogglePlay();}
    else if(e.code==='ArrowLeft'){e.preventDefault();smStep(-1);}
    else if(e.code==='ArrowRight'){e.preventDefault();smStep(1);}
  });
}

window.addEventListener('DOMContentLoaded',()=>{
  document.querySelectorAll('table').forEach(tbl=>{
    tbl.querySelectorAll('tr').forEach(row=>{
      let off=0;
      row.querySelectorAll('.sticky').forEach(cell=>{
        cell.style.left=off+'px';
        off+=cell.offsetWidth;
      });
    });
  });
  function tick(){
    const now=new Date(),m=now.getMinutes(),s=now.getSeconds(),left=((14-(m%15))*60)+(60-s),mm=String(Math.floor(left/60)).padStart(2,'0'),ss=String(left%60).padStart(2,'0');
    document.getElementById('cd').textContent='Next: '+mm+':'+ss;
    if(left<=0)location.reload();
  }
  tick();
  setInterval(tick,1000);
  ['Gold','Bitcoin','XAUBTC'].forEach(animateIndicator);
  setupStateMapControls();
  animateStateMap();
});
"""

def build_html(sections, state_map_html=""):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html lang="th" data-theme="light">
<head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>EMA Cross Report</title>
<style>
:root{{--bg:#0f0f0f;--bg2:#1a1a1a;--bg3:#111;--bg4:#161616;--border:#2a2a2a;--border2:#333;--text:#e8e0d0;--text2:#aaa;--text3:#666;--text4:#444;--gold:#f5d06e;--blue:#60a5fa;--g-bg:#0d3d22;--g-fg:#4ade80;--d-bg:#3d0d0d;--d-fg:#f87171;--price:#7dd3fc;--tog-bg:#2a2a2a;--shadow:4px 0 8px rgba(0,0,0,.5);}}
[data-theme="light"]{{--bg:#f5f5f0;--bg2:#fff;--bg3:#efefea;--bg4:#f0f0eb;--border:#ddd;--border2:#ccc;--text:#1a1a1a;--text2:#444;--text3:#888;--text4:#bbb;--gold:#92400e;--blue:#1d4ed8;--g-bg:#dcfce7;--g-fg:#166534;--d-bg:#fee2e2;--d-fg:#991b1b;--price:#0369a1;--tog-bg:#e5e5e5;--shadow:4px 0 6px rgba(0,0,0,.1);}}
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
.state-map-box{{margin-bottom:14px;padding:8px;border-radius:8px;background:var(--bg2);border:1.5px solid var(--border2);display:inline-block;}}
.state-map-header{{display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:4px;}}
.state-map-label{{font-size:9px;color:var(--text3);font-weight:700;text-transform:uppercase;letter-spacing:.5px;}}
.state-map-ts{{font-size:9px;color:var(--text3);font-weight:700;font-family:monospace;white-space:nowrap;}}
.state-map-progress{{height:2px;background:var(--border);border-radius:1px;overflow:hidden;margin-top:4px;}}
.state-map-progress-fill{{height:100%;background:var(--gold);width:0%;}}
.state-map-controls{{display:flex;align-items:center;gap:4px;margin-top:6px;flex-wrap:wrap;font-size:10px;}}
.sm-btn{{background:var(--tog-bg);border:1px solid var(--border2);border-radius:4px;padding:2px 7px;font-size:11px;color:var(--text2);cursor:pointer;min-width:24px;line-height:1.4;transition:all .15s;}}
.sm-btn:hover{{border-color:var(--gold);color:var(--gold);}}
.sm-btn.sm-play{{min-width:30px;font-weight:700;}}
.sm-frame-count{{font-size:10px;color:var(--text3);font-family:monospace;font-weight:700;margin-left:4px;min-width:54px;text-align:center;}}
.sm-speed-sep{{color:var(--text4);margin:0 2px;}}
.sm-speed-label{{font-size:9px;color:var(--text3);font-weight:700;text-transform:uppercase;letter-spacing:.5px;}}
.sm-speed-btn{{background:var(--tog-bg);border:1px solid var(--border2);border-radius:4px;padding:2px 6px;font-size:10px;color:var(--text2);cursor:pointer;font-family:monospace;font-weight:700;transition:all .15s;}}
.sm-speed-btn:hover{{border-color:var(--gold);color:var(--gold);}}
.sm-speed-btn.active{{background:var(--gold);color:var(--bg);border-color:var(--gold);}}
.state-map-legend{{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap;}}
.sm-legend-pill{{display:inline-flex;align-items:center;gap:5px;background:var(--tog-bg);border:1px solid var(--border2);border-radius:14px;padding:3px 10px 3px 4px;font-size:10px;font-weight:700;color:var(--text3);cursor:pointer;transition:all .15s;opacity:.45;}}
.sm-legend-pill:hover{{border-color:var(--gold);color:var(--gold);}}
.sm-legend-pill.active{{opacity:1;color:var(--text2);border-color:var(--border2);}}
.sm-legend-swatch{{display:inline-flex;align-items:center;justify-content:center;width:16px;height:16px;border-radius:50%;font-size:9px;color:#fff;font-weight:700;}}
.sm-sw-gold{{background:linear-gradient(135deg,#fde68a 0%,#f59e0b 40%,#d97706 70%,#92400e 100%);border:1px solid #78350f;border-radius:3px;width:14px;height:10px;}}
.sm-sw-btc{{background:#f7931a;border:1px solid #333;}}
.sm-sw-xb{{background:#6366f1;border:1px solid #333;font-size:6px;}}
.sm-sw-spy{{background:radial-gradient(circle,#fff 0%,#fff 20%,#dc2626 30%,#dc2626 60%,#fff 60%,#fff 70%,#dc2626 70%,#dc2626 100%);border:1px solid #7f1d1d;}}
.sm-sw-qqq{{background:#1a6fc9;border:1px solid #333;font-size:9px;}}
.sm-sw-dxy{{background:#15803d;border:1px solid #333;font-size:9px;}}
.ind-heatmap-wrap{{display:inline-flex;flex-direction:column;gap:6px;margin-bottom:8px;}}
.indicator-box{{display:inline-flex;flex-direction:column;gap:6px;padding:8px 10px;border-radius:8px;background:var(--bg2);border:1.5px solid var(--border2);min-width:240px;}}
.heatmap-box{{padding:6px 8px;border-radius:8px;background:var(--bg2);border:1.5px solid var(--border2);display:inline-block;}}
.heatmap-label{{font-size:9px;color:var(--text3);font-weight:700;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px;}}
.ind-timeline{{display:flex;align-items:center;gap:8px;margin-bottom:2px;}}
.ind-ts{{font-size:9px;color:var(--text3);font-weight:700;font-family:monospace;white-space:nowrap;}}
.ind-progress{{flex:1;height:3px;background:var(--border);border-radius:2px;overflow:hidden;}}
.ind-progress-fill{{height:100%;background:var(--gold);width:0%;transition:width .3s linear;}}
.ind-price{{display:flex;align-items:center;gap:8px;padding-top:14px;}}
.ind-price-label{{font-size:9px;color:var(--text3);font-weight:700;text-transform:uppercase;letter-spacing:.5px;width:28px;}}
.ind-price-bar{{flex:1;display:flex;align-items:center;gap:6px;}}
.ind-min,.ind-max{{font-size:9px;color:var(--text3);font-weight:700;font-family:monospace;white-space:nowrap;}}
.ind-track{{flex:1;height:8px;border-radius:3px;background:linear-gradient(to right,#ef4444 0%,#eab308 50%,#22c55e 100%);position:relative;opacity:.35;}}
.ind-marker{{position:absolute;top:-3px;width:4px;height:14px;border-radius:2px;transform:translateX(-2px);border:1px solid var(--text);box-shadow:0 0 0 1.5px var(--bg2);z-index:2;transition:left .3s,background .3s;}}
.ind-cur{{position:absolute;bottom:14px;font-size:9px;font-weight:700;font-family:monospace;transform:translateX(-50%);white-space:nowrap;color:#fff!important;padding:2px 6px;border-radius:4px;transition:left .3s,background .3s;text-shadow:-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,1px 1px 0 #000;}}
.ind-rsi{{display:flex;align-items:flex-start;gap:8px;}}
.ind-rsi-label{{font-size:9px;color:var(--text3);font-weight:700;text-transform:uppercase;letter-spacing:.5px;width:28px;padding-top:4px;}}
.ind-rsi-bars{{flex:1;display:flex;justify-content:space-between;gap:4px;}}
.rsi-bar{{display:flex;flex-direction:column;align-items:center;gap:2px;flex:1;}}
.rsi-track{{width:7px;height:36px;border-radius:3px;background:linear-gradient(to top,#22c55e 0%,#eab308 50%,#ef4444 100%);position:relative;opacity:.35;}}
.rsi-marker{{position:absolute;left:-2px;width:10px;height:3px;border-radius:1px;border:1px solid var(--text);box-shadow:0 0 0 1px var(--bg2);transition:bottom .3s,background .3s;}}
.rsi-lbl{{font-size:8px;color:var(--text3);font-weight:700;}}
.rsi-val{{font-size:9px;font-weight:700;font-family:monospace;color:#fff!important;padding:2px 5px;border-radius:4px;min-width:18px;text-align:center;transition:background .3s;text-shadow:-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,1px 1px 0 #000;}}
.summary-box{{margin-bottom:8px;padding:8px 10px;border-radius:8px;background:var(--bg2);border:1px solid var(--border);}}
.summary-label{{font-size:11px;color:var(--text3);font-weight:600;display:block;margin-bottom:6px;}}
.summary-chips{{display:flex;flex-wrap:wrap;gap:6px;}}
.chip{{display:inline-flex;align-items:center;gap:5px;font-size:11px;font-weight:700;padding:3px 8px;border-radius:5px;}}
.chip-g{{background:var(--g-bg);color:var(--g-fg);}} .chip-d{{background:var(--d-bg);color:var(--d-fg);}}
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
th.iv-sep{{border-left:2px solid var(--border2);}} td.iv-sep{{border-left:2px solid var(--border2);}}
.sticky{{position:sticky;z-index:2;background:var(--bg2);}} thead .sticky{{z-index:5;background:var(--bg3);}}
.s0{{left:0;}} .s1{{left:52px;box-shadow:var(--shadow);}} thead .s1{{box-shadow:var(--shadow);}}
tr.data:hover .sticky{{background:var(--bg4);}}
td.tm{{color:var(--text2);}} td.price{{color:var(--price);min-width:72px;}}
.th-fix{{color:var(--text3)!important;font-size:11px!important;}}
tr.data:hover td{{background:var(--bg4);}}
tr.day-sep td{{background:var(--bg4)!important;color:var(--gold);font-size:11px;text-align:left;padding:4px 8px;letter-spacing:.5px;border-top:1px solid var(--border2);border-bottom:1px solid var(--border2);font-weight:700;}}
.day-label{{font-weight:700!important;color:var(--gold)!important;}}
.g{{display:inline-block;background:var(--g-bg);color:var(--g-fg);border-radius:3px;padding:2px 5px;font-weight:700;font-size:11px;min-width:16px;}}
.d{{display:inline-block;background:var(--d-bg);color:var(--d-fg);border-radius:3px;padding:2px 5px;font-weight:700;font-size:11px;min-width:16px;}}
.n{{color:var(--border2);font-size:11px;font-weight:400;}}
.bg-g{{background:var(--g-bg);}} .bg-d{{background:var(--d-bg);}}
.bg-g .n{{color:var(--g-fg);opacity:.4;}} .bg-d .n{{color:var(--d-fg);opacity:.4;}}
.dot-g{{color:#22c55e;font-size:14px;font-weight:700;}} .dot-d{{color:#ef4444;font-size:14px;font-weight:700;}}
.rsi{{font-size:11px;color:var(--text3);font-weight:700;}}
.rsi-hi{{background:var(--d-bg);color:var(--d-fg);}} .rsi-lo{{background:var(--g-bg);color:var(--g-fg);}}
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
</style></head><body>
<div class="top-bar"><div><p class="page-title">EMA Cross Report</p><p class="page-sub">max {MAX_ROWS} rows · Bangkok (ICT) · Updated: {now} · <span id="cd" style="color:var(--blue);font-weight:700;">--:--</span></p></div><div style="display:flex;gap:6px;"><button class="toggle-btn" onclick="toggleGroup()"><span id="gl">By Interval</span></button><button class="toggle-btn" onclick="toggleTheme()"><span id="ti">🌙</span><span id="tl">Dark</span></button></div></div>
<div class="legend"><div class="leg"><span class="lg">G</span> Golden (Buy)</div><div class="leg"><span class="ld">D</span> Death (Sell)</div><div class="leg" style="color:var(--text4);font-weight:400">S=12/26 M=20/50 L=50/200 R=RSI-14</div></div>
{state_map_html}
{body}
<p class="footer">Auto-generated every 15 min · GitHub Actions</p>
<script>{JS_BLOCK}</script>
</body></html>"""

def event_key(asset, ds, tm, iv, lb, cr): return f"{asset}|{ds}|{tm}|{iv}|{lb}|{cr}"

def send_email(subject, body_html):
    if not GMAIL_PASSWORD: print("  Email SKIP: no password"); return False
    try:
        msg = MIMEText(body_html, "html"); msg["Subject"] = subject; msg["From"] = GMAIL_SENDER; msg["To"] = ALERT_TO
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s: s.login(GMAIL_SENDER, GMAIL_PASSWORD); s.sendmail(GMAIL_SENDER, ALERT_TO, msg.as_string())
        return True
    except Exception as e: print(f"  Email ERROR: {e}"); return False

def build_email_body(evts, now_str, fid="", ast=None):
    ba = {}
    for ev in evts: ba.setdefault(ev["asset"], []).append(ev)
    lh = f'<a href="https://payotorn-droid.github.io/ema-cross-alert/" style="color:#1d4ed8;font-weight:700;font-size:14px;">Open Dashboard</a><br>'
    sh = ""
    if ast:
        badges = [f'<span style="font-size:13px;font-weight:700;">{a}: {s}</span>' for a, s in ast.items()]
        sh = '<div style="margin:8px 0;">' + ' &nbsp;|&nbsp; '.join(badges) + '</div>'
    secs = ""
    for asset, events in ba.items():
        rows, pd_ = "", ""
        for ev in sorted(events, key=lambda e: (e["date"], e["time"])):
            if ev["date"] != pd_:
                rows += f'<tr><td colspan="4" style="padding:4px 4px 1px;font-size:10px;color:#888;border-top:1px solid #eee;">{ev["date"]}</td></tr>'
                pd_ = ev["date"]
            ic = "🟢" if ev["cross"] == "GOLDEN" else "🔴"
            co = "#166534" if ev["cross"] == "GOLDEN" else "#991b1b"
            bg = "#dcfce7" if ev["cross"] == "GOLDEN" else "#fee2e2"
            rows += f'<tr><td style="padding:3px 4px;font-size:12px;">{ev["time"]}</td><td style="padding:3px 4px;font-size:12px;">{ev["interval"]}·{ev["label"]}</td><td style="padding:3px 4px;"><span style="background:{bg};color:{co};border-radius:3px;padding:1px 5px;font-weight:700;font-size:11px;">{ic}{ev["cross"][0]}</span></td><td style="padding:3px 4px;font-size:12px;text-align:right;">{fmt_price(ev["price"])}</td></tr>'
        secs += f'<div style="margin-bottom:12px;"><div style="font-weight:700;color:#92400e;font-size:13px;margin-bottom:4px;">{asset} ({len(events)})</div><table style="width:100%;border-collapse:collapse;"><tr style="color:#888;font-size:10px;"><th style="text-align:left;padding:2px 4px;">Time</th><th style="text-align:left;padding:2px 4px;">Iv·EMA</th><th style="text-align:left;padding:2px 4px;">Sig</th><th style="text-align:right;padding:2px 4px;">Price</th></tr>{rows}</table></div>'
    return f'<html><body style="font-family:Arial,sans-serif;margin:0;padding:8px;background:#f5f5f0;"><div style="max-width:360px;margin:auto;background:#fff;border-radius:8px;border-left:4px solid #f5d06e;padding:12px;"><div style="font-size:14px;font-weight:700;color:#92400e;margin-bottom:2px;">⚡ EMA Cross Alert</div><div style="font-size:11px;color:#888;margin-bottom:8px;">{now_str}</div>{lh}{sh}{secs}<div style="font-size:10px;color:#aaa;margin-top:8px;">Auto-generated · GitHub Actions</div></div></body></html>'

def upload_to_drive(filepath, folder_id):
    sa_key = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY")
    if not sa_key: print("  Drive SKIP"); return False
    try:
        from google.oauth2 import service_account; from googleapiclient.discovery import build; from googleapiclient.http import MediaFileUpload
        creds = service_account.Credentials.from_service_account_info(json.loads(sa_key), scopes=["https://www.googleapis.com/auth/drive"])
        svc = build("drive", "v3", credentials=creds); fn = os.path.basename(filepath)
        res = svc.files().list(q=f"name='{fn}' and '{folder_id}' in parents and trashed=false", fields="files(id)").execute()
        ex = res.get("files", []); media = MediaFileUpload(filepath, mimetype="text/html")
        if ex: svc.files().update(fileId=ex[0]["id"], media_body=media).execute(); print(f"  Drive updated: {fn}")
        else: svc.files().create(body={"name": fn, "parents": [folder_id]}, media_body=media).execute(); print(f"  Drive created: {fn}")
        return True
    except Exception as e: print(f"  Drive ERROR: {e}"); return False

def analyze_market_state(all_events, rsi_data=None):
    cs = {}; al = sorted(all_events.keys())
    for (d, t) in al:
        ev = all_events[(d, t)]
        for iv in INTERVALS:
            for _, _, lb in EMA_PAIRS:
                cr = ev["crosses"].get(iv, {}).get(lb)
                if cr: cs[(iv, lb)] = cr
    def sig(iv, lb): return cs.get((iv, lb))
    big = [sig("1d","S"),sig("1d","M"),sig("1d","L"),sig("4h","S"),sig("4h","M"),sig("4h","L")]
    mid = [sig("1h","S"),sig("1h","M"),sig("1h","L")]
    small = [sig("15m","S"),sig("15m","M"),sig("15m","L"),sig("30m","S"),sig("30m","M"),sig("30m","L")]
    bg, bd = sum(1 for s in big if s=="GOLDEN"), sum(1 for s in big if s=="DEATH")
    mg, md = sum(1 for s in mid if s=="GOLDEN"), sum(1 for s in mid if s=="DEATH")
    sg, sd = sum(1 for s in small if s=="GOLDEN"), sum(1 for s in small if s=="DEATH")
    rc = ""
    if rsi_data and al:
        lk = al[-1]; rr = lookup_rsi(rsi_data, lk[0], lk[1])
        for iv in ["4h","1d"]:
            v = rr.get(iv)
            if v and (v >= 70 or v <= 30): rc = f" RSI {iv}={int(v)}"; break
    if sig("1d","L")=="DEATH":
        for (d,t) in reversed(al):
            if all_events[(d,t)]["crosses"].get("1d",{}).get("L")=="DEATH":
                if (d,t)==al[-1]: return f"🔴 1d Death Cross 50/200{rc}"
                break
    if sig("1d","L")=="GOLDEN":
        for (d,t) in reversed(al):
            if all_events[(d,t)]["crosses"].get("1d",{}).get("L")=="GOLDEN":
                if (d,t)==al[-1]: return f"🟢 1d Golden Cross 50/200{rc}"
                break
    if bg>=5 and mg>=2: return f"🟢 Full Bull{rc}"
    if bd>=5 and md>=2: return f"🔴 Full Bear{rc}"
    if bg>=4 and sg>=3: return f"🟢 Bull Wave{rc}"
    if bd>=4 and sd>=3: return f"🔴 Bear Wave{rc}"
    if sg>=4 and bd>=4: return f"⚠️ Divergence: Short↑ Long↓{rc}"
    if sd>=4 and bg>=4: return f"⚠️ Divergence: Short↓ Long↑{rc}"
    if bg>=3 and md>=2 and sd>=3: return f"⚠️ Momentum Fading{rc}"
    if bd>=3 and mg>=2 and sg>=3: return f"⚠️ Recovering{rc}"
    tg, td = bg+mg+sg, bd+md+sd
    if tg > td: return f"🟡 Lean Bull{rc}"
    elif td > tg: return f"🟡 Lean Bear{rc}"
    return f"🟡 Mixed{rc}"

def find_last_4h_cross(all_events):
    lt = None
    for (d, t), ev in all_events.items():
        if "4h" in ev["crosses"]:
            ts = pd.Timestamp(f"{d} {t}")
            if lt is None or ts > lt: lt = ts
    return lt

# ── Main ─────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    state = load_state()
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now_str}] report.py starting...")
    sections, all_new_evs, asset_states = [], [], {}
    asset_data = {}  # Cache {asset: {"events": ..., "rsi": ...}} for reuse in state map build
    for asset_name in ASSETS:
        all_events = collect_events(asset_name)
        rsi_data = collect_rsi(asset_name)
        asset_data[asset_name] = {"events": all_events, "rsi": rsi_data}
        if asset_name in MAIN_ASSETS:
            # Indicator panel, table, state badge, email alerts only for main assets
            sections.append(build_table_html(asset_name, all_events, rsi_data))
            asset_states[asset_name] = analyze_market_state(all_events, rsi_data)
            print(f"  {asset_name}: {asset_states[asset_name]}")
            last_4h = find_last_4h_cross(all_events)
            ecut = last_4h if last_4h else now - timedelta(hours=4)
            for (ds, tm), ev in all_events.items():
                if pd.Timestamp(f"{ds} {tm}") < ecut: continue
                for iv, pairs in ev["crosses"].items():
                    for lb, cr in pairs.items():
                        ek = event_key(asset_name, ds, tm, iv, lb, cr)
                        all_new_evs.append({"asset": asset_name, "date": ds, "time": tm, "interval": iv, "label": lb, "label_full": LABEL_FULL.get(lb, lb), "cross": cr, "price": ev["price"], "event_key": ek})
        else:
            # State-map-only asset: just log current state for visibility
            st = analyze_market_state(all_events, rsi_data)
            print(f"  {asset_name} (state map only): {st}")

    # Build state map with animation timeline aligned to indicator (same N as MAX_ROWS)
    frames = build_state_timeline_frames(asset_data, N=MAX_ROWS)
    smh = build_state_map_html(frames)

    html = build_html(sections, smh)
    with open(OUTPUT_HTML, "w", encoding="utf-8") as f: f.write(html)
    print(f"  HTML saved: {OUTPUT_HTML}")

    DRIVE_FOLDER_ID = os.environ.get("DRIVE_FOLDER_ID", "")
    if DRIVE_FOLDER_ID: upload_to_drive(OUTPUT_HTML, DRIVE_FOLDER_ID)

    truly_new = [e for e in all_new_evs if e["event_key"] not in state["sent_events"]]
    if truly_new:
        can_send = True
        if state.get("last_email"):
            if (now - datetime.fromisoformat(state["last_email"])).total_seconds() / 60 < MIN_EMAIL_GAP:
                can_send = False; print(f"  Email cooldown")
        if can_send:
            sp = [f"{a}: {s}" for a, s in asset_states.items()]
            subj = f"⚡ {' | '.join(sp)} — {len(truly_new)} new"
            if send_email(subj, build_email_body(all_new_evs, now_str, DRIVE_FOLDER_ID, asset_states)):
                state["last_email"] = now.isoformat()
                print(f"  Email sent: {len(truly_new)} new, {len(all_new_evs)} total shown")
        for ev in truly_new: state["sent_events"].append(ev["event_key"])
        state["sent_events"] = state["sent_events"][-2000:]
    else:
        print("  No new events.")

    save_state(state)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] report.py done.")
