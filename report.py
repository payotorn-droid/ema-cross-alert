import pandas as pd
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
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

MAX_ROWS       = 30
LOOKBACK_DAYS  = 30
MIN_EMAIL_GAP  = 60

ASSETS = {
    "Gold":    "GC=F",
    "Bitcoin": "BTC-USD",
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


def cell_html(cross, iv_sep=False, last_signal=None, last_price=None, cur_price=None):
    classes = []
    if iv_sep:
        classes.append("iv-sep")
    if cross == "GOLDEN":
        return f'<td class="{" ".join(classes)}"><span class="g">G</span></td>' if classes else f'<td><span class="g">G</span></td>'
    elif cross == "DEATH":
        return f'<td class="{" ".join(classes)}"><span class="d">D</span></td>' if classes else f'<td><span class="d">D</span></td>'
    else:
        if last_signal and last_price and cur_price:
            if last_signal == "GOLDEN":
                classes.append("bg-g")
            elif last_signal == "DEATH":
                classes.append("bg-d")
            cls_str = f' class="{" ".join(classes)}"' if classes else ''
            if cur_price > last_price:
                return f'<td{cls_str}><span class="dot-g">●</span></td>'
            elif cur_price < last_price:
                return f'<td{cls_str}><span class="dot-d">●</span></td>'
            else:
                return f'<td{cls_str}><span class="n">—</span></td>'
        cls_str = f' class="{" ".join(classes)}"' if classes else ''
        return f'<td{cls_str}><span class="n">—</span></td>'


def fmt_price(price):
    return f"${price:,.0f}" if price >= 1000 else f"${price:,.2f}"


def build_table_html(asset_name, all_events):
    n_ema       = len(EMA_PAIRS)
    total_cols  = 2 + len(INTERVALS) * n_ema
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

    h1  = '<tr>'
    h1 += '<th rowspan="2" class="sticky s0 left th-fix">Time</th>'
    h1 += '<th rowspan="2" class="sticky s1 left th-fix">Price</th>'
    for iv in INTERVALS:
        h1 += f'<th colspan="{n_ema}" class="iv-sep">{iv}</th>'
    h1 += '</tr>'

    h2 = '<tr>'
    for idx in range(len(INTERVALS)):
        for j, (_, _, lbl) in enumerate(EMA_PAIRS):
            cls = ' class="iv-sep"' if j == 0 and idx > 0 else ''
            h2 += f'<th{cls}>{lbl}</th>'
    h2 += '</tr>'

    rows_html = ""
    prev_date = None
    for (date_str, time_str) in display_keys:
        if prev_date != date_str:
            rows_html += f'<tr class="day-sep"><td class="sticky s0 day-label">{date_str}</td><td colspan="{total_cols-1}"></td></tr>'
        prev_date = date_str

        ev   = all_events[(date_str, time_str)]
        cur_price = ev["price"]
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
                row += cell_html(cross, first_in_group, col_state.get(col_key), col_price.get(col_key), cur_price)
        row += '</tr>'
        rows_html += row

    if not rows_html:
        rows_html = f'<tr><td colspan="{total_cols}" class="empty">No EMA cross events</td></tr>'

    return f"""
    <div class="asset-block">
      <div class="asset-title">{asset_name}</div>
      {summary_html}
      <div class="table-scroll">
        <table>
          <thead>{h1}{h2}</thead>
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
  thead tr:first-child th{{color:var(--blue);font-size:12px;padding-top:7px;padding-bottom:2px;border-bottom:none;position:sticky;top:0;z-index:3;background:var(--bg3);}}
  thead tr:last-child th{{color:var(--text3);font-size:11px;padding-top:1px;padding-bottom:5px;border-bottom:1px solid var(--border2);position:sticky;top:26px;z-index:3;background:var(--bg3);}}
  th.iv-sep{{border-left:2px solid var(--border2);}}
  td.iv-sep{{border-left:2px solid var(--border2);}}
  .sticky{{position:sticky;z-index:2;background:var(--bg2);}}
  thead tr:first-child .sticky{{z-index:5;background:var(--bg3);}}
  thead tr:last-child .sticky{{z-index:5;background:var(--bg3);}}
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
  .empty{{text-align:center;color:var(--text4);padding:14px;font-weight:400;}}
  .footer{{font-size:10px;color:var(--text4);text-align:center;margin-top:6px;}}
</style>
</head>
<body>
<div class="top-bar">
  <div>
    <p class="page-title">EMA Cross Report</p>
    <p class="page-sub">max {MAX_ROWS} rows · Bangkok (ICT) · Updated: {now}</p>
  </div>
  <button class="toggle-btn" onclick="toggleTheme()">
    <span id="ti">🌙</span><span id="tl">Dark</span>
  </button>
</div>
<div class="legend">
  <div class="leg"><span class="lg">G</span> Golden (Buy)</div>
  <div class="leg"><span class="ld">D</span> Death (Sell)</div>
  <div class="leg" style="color:var(--text4);font-weight:400">S=12/26 M=20/50 L=50/200</div>
</div>
{body}
<p class="footer">Auto-generated every 15 min · GitHub Actions</p>
<script>
  function toggleTheme(){{
    const h=document.documentElement,dark=h.getAttribute('data-theme')==='dark';
    h.setAttribute('data-theme',dark?'light':'dark');
    document.getElementById('ti').textContent=dark?'🌙':'☀️';
    document.getElementById('tl').textContent=dark?'Dark':'Light';
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
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = GMAIL_SENDER
        msg["To"]      = ALERT_TO
        msg.attach(MIMEText(body_html, "html"))
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_SENDER, GMAIL_PASSWORD)
            s.sendmail(GMAIL_SENDER, ALERT_TO, msg.as_string())
        return True
    except Exception as e:
        print(f"  Email ERROR: {e}")
        return False


def build_email_body(new_events_list, now_str):
    rows = ""
    for ev in new_events_list:
        icon   = "🟢" if ev["cross"] == "GOLDEN" else "🔴"
        color  = "#166534" if ev["cross"] == "GOLDEN" else "#991b1b"
        bg     = "#dcfce7" if ev["cross"] == "GOLDEN" else "#fee2e2"
        rows  += f"""
        <tr>
          <td style="padding:6px 10px;font-weight:700;">{ev['asset']}</td>
          <td style="padding:6px 10px;">{ev['date']} {ev['time']}</td>
          <td style="padding:6px 10px;">{ev['interval']}</td>
          <td style="padding:6px 10px;">{ev['label_full']}</td>
          <td style="padding:6px 10px;text-align:center;">
            <span style="background:{bg};color:{color};border-radius:4px;padding:2px 8px;font-weight:700;">
              {icon} {ev['cross']}
            </span>
          </td>
          <td style="padding:6px 10px;font-family:monospace;">{fmt_price(ev['price'])}</td>
        </tr>"""

    return f"""
    <html><body style="font-family:Arial,sans-serif;padding:20px;background:#f5f5f0;">
      <div style="max-width:600px;margin:auto;background:#fff;border-radius:10px;
                  border-left:6px solid #f5d06e;padding:20px;">
        <h2 style="color:#92400e;margin:0 0 4px;">⚡ EMA Cross Alert</h2>
        <p style="color:#888;font-size:13px;margin:0 0 16px;">{now_str} (Bangkok)</p>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
          <thead>
            <tr style="background:#f5f5f0;color:#888;font-size:11px;">
              <th style="padding:6px 10px;text-align:left;">Asset</th>
              <th style="padding:6px 10px;text-align:left;">Date/Time</th>
              <th style="padding:6px 10px;text-align:left;">Interval</th>
              <th style="padding:6px 10px;text-align:left;">EMA</th>
              <th style="padding:6px 10px;text-align:left;">Signal</th>
              <th style="padding:6px 10px;text-align:left;">Price</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        <hr style="border:none;border-top:1px solid #eee;margin:16px 0;">
        <p style="font-size:11px;color:#aaa;margin:0;">EMA Cross Report · Auto-generated</p>
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
            scopes=["https://www.googleapis.com/auth/drive.file"]
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
            # Update existing file
            service.files().update(
                fileId=existing[0]["id"],
                media_body=media
            ).execute()
            print(f"  Drive updated: {filename}")
        else:
            # Create new file
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


# ── Main ─────────────────────────────────────────────────────
os.makedirs(DATA_DIR, exist_ok=True)
state = load_state()
now   = datetime.now()
now_str = now.strftime("%Y-%m-%d %H:%M:%S")

print(f"[{now_str}] report.py starting...")

# 1. Collect events from CSV
sections    = []
all_new_evs = []

for asset_name in ASSETS:
    all_events = collect_events(asset_name)
    sections.append(build_table_html(asset_name, all_events))

    # Check for new events not yet sent (only recent ones)
    email_cutoff = (now - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    for (date_str, time_str), ev in all_events.items():
        if date_str < email_cutoff:
            continue
        for interval, pairs in ev["crosses"].items():
            for label, cross in pairs.items():
                ek = event_key(asset_name, date_str, time_str, interval, label, cross)
                if ek not in state["sent_events"]:
                    all_new_evs.append({
                        "asset":      asset_name,
                        "date":       date_str,
                        "time":       time_str,
                        "interval":   interval,
                        "label":      label,
                        "label_full": LABEL_FULL.get(label, label),
                        "cross":      cross,
                        "price":      ev["price"],
                        "event_key":  ek,
                    })

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
if all_new_evs:
    last_email = state.get("last_email")
    can_send   = True
    if last_email:
        elapsed = (now - datetime.fromisoformat(last_email)).total_seconds() / 60
        if elapsed < MIN_EMAIL_GAP:
            can_send = False
            print(f"  Email cooldown: {MIN_EMAIL_GAP - int(elapsed)} min remaining")

    if can_send:
        subject   = f"⚡ EMA Cross Alert — {len(all_new_evs)} new event(s)"
        body_html = build_email_body(all_new_evs, now_str)
        if send_email(subject, body_html):
            state["last_email"] = now.isoformat()
            print(f"  Email sent: {len(all_new_evs)} new event(s)")

    # Mark all new events as sent
    for ev in all_new_evs:
        if ev["event_key"] not in state["sent_events"]:
            state["sent_events"].append(ev["event_key"])

    state["sent_events"] = state["sent_events"][-500:]
else:
    print("  No new events.")

save_state(state)
print(f"[{datetime.now().strftime('%H:%M:%S')}] report.py done.\n")
