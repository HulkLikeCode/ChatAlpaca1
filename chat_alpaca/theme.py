STALE_VALUE_COLOR = "#86A7D8"


THEME_CSS = """
<style>
:root {
  --ink: #f7f8ff;
  --muted: #8f9bb7;
  --panel: rgba(12, 17, 31, 0.92);
  --line: rgba(105, 126, 255, 0.24);
  --blue: #4d8dff;
  --cyan: #67d7ff;
  --violet: #9b72ff;
  --stale: __STALE_VALUE_COLOR__;
}
.stApp {
  background:
    radial-gradient(circle at 10% 0%, rgba(62, 83, 196, .20), transparent 32rem),
    radial-gradient(circle at 92% 12%, rgba(115, 62, 190, .18), transparent 30rem),
    #05060a;
  color: var(--ink);
}
header[data-testid="stHeader"] {
  opacity: 0;
  transition: opacity .18s ease-in-out;
}
header[data-testid="stHeader"]:hover,
header[data-testid="stHeader"]:focus-within {
  opacity: 1;
}
.block-container {max-width: 1380px; padding-top: .7rem; padding-bottom: 3rem;}
h1, h2, h3 {letter-spacing: -0.035em; color: var(--ink);}
[data-testid="stCaptionContainer"] {margin-top: -.2rem; margin-bottom: -.35rem;}
[data-testid="stCaptionContainer"], .stCaption {color: var(--muted) !important;}
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(15,22,42,.96), rgba(8,11,22,.96));
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: .72rem .85rem;
  min-height: 104px;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
  color: var(--violet) !important;
  font-size: .72rem !important;
  font-weight: 800 !important;
  letter-spacing: .06em;
  line-height: 1.15;
}
[data-testid="stMetricValue"] {
  color: var(--ink);
  font-size: 1.55rem;
  font-weight: 760;
  line-height: 1.15;
}
[data-testid="stMetricDelta"] {color: var(--cyan) !important;}
[class*="st-key-stale_metric_"] [data-testid="stMetricValue"] {
  color: var(--stale) !important;
}
[class*="st-key-compact_transaction_"] [data-testid="stVerticalBlock"] {gap: .45rem;}
[class*="st-key-compact_transaction_"] [data-testid="stForm"] {
  padding: .65rem .8rem;
}
[class*="st-key-compact_transaction_"] [data-testid="stForm"]
  [data-testid="stVerticalBlock"] {gap: .35rem;}
.st-key-master_controls {
  position: sticky;
  top: 3.75rem;
  z-index: 999;
  margin: 0 0 .2rem;
  padding: .35rem .75rem .15rem;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: rgba(7, 10, 19, .94);
  box-shadow: 0 12px 34px rgba(0, 0, 0, .34);
  backdrop-filter: blur(16px);
}
.st-key-master_controls [data-testid="stForm"] {border: 0; padding: 0;}
.portfolio-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: .75rem;
}
.portfolio-card {
  background: linear-gradient(145deg, rgba(15,22,42,.96), rgba(8,11,22,.96));
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: .72rem .85rem;
  min-height: 104px;
}
.portfolio-card .eyebrow {color: var(--violet); font-size: .72rem; font-weight: 800; letter-spacing: .06em; line-height: 1.15;}
.portfolio-card .value {color: var(--ink); font-size: 1.55rem; font-weight: 760; line-height: 1.15; margin-top: .35rem; white-space: nowrap;}
.portfolio-card .detail {color: var(--muted); font-size: .82rem; margin-top: .25rem;}
.performance-status {
  display: flex;
  flex-wrap: wrap;
  gap: .28rem .75rem;
  margin: .35rem 0 .45rem;
  color: var(--muted);
  font-size: .78rem;
  line-height: 1.25;
}
.performance-status span + span::before {
  content: "·";
  margin-right: .75rem;
  color: rgba(143, 155, 183, .58);
}
.stButton > button, .stDownloadButton > button {
  border-radius: 10px;
  border: 1px solid rgba(105,126,255,.45);
  background: linear-gradient(110deg, rgba(38,76,153,.9), rgba(93,56,154,.9));
  color: white;
}
.stButton > button:hover {border-color: var(--cyan); color: white;}
[data-baseweb="tab-list"] {gap: .25rem; border-bottom: 1px solid var(--line);}
[data-baseweb="tab"] {height: 2.65rem; padding: 0 .8rem; color: var(--muted);}
[aria-selected="true"] {color: white !important;}
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {border: 1px solid var(--line); border-radius: 12px; overflow: hidden;}
[data-testid="stAlert"] {background: rgba(19,27,52,.88); border-color: rgba(105,126,255,.4); color: white;}
.stale-symbol-alert {
  margin: .35rem 0 .75rem;
  padding: .75rem 1rem;
  border: 1px solid rgba(134, 167, 216, .58);
  border-radius: 12px;
  background: rgba(78, 116, 174, .16);
  color: #b9cbea;
}
hr {border-color: var(--line);}
</style>
""".replace("__STALE_VALUE_COLOR__", STALE_VALUE_COLOR)


PLOT_COLORS = [
    "#67D7FF",
    "#4D8DFF",
    "#9B72FF",
    "#D0B7FF",
    "#6F7DFF",
    "#FFFFFF",
    "#4662A8",
    "#8064C9",
    "#8FB8FF",
]
