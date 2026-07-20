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
}
.stApp {
  background:
    radial-gradient(circle at 10% 0%, rgba(62, 83, 196, .20), transparent 32rem),
    radial-gradient(circle at 92% 12%, rgba(115, 62, 190, .18), transparent 30rem),
    #05060a;
  color: var(--ink);
}
.block-container {max-width: 1380px; padding-top: .35rem; padding-bottom: 3rem;}
h1, h2, h3 {letter-spacing: -0.035em; color: var(--ink);}
h1 {
  font-size: clamp(1.25rem, 2.2vw, 1.65rem) !important;
  line-height: 1 !important;
  margin-top: -.2rem !important;
  margin-bottom: -.1rem !important;
  color: var(--violet) !important;
}
[data-testid="stMarkdownContainer"]:has(.mode-chip) {height: 1.05rem;}
[data-testid="stCaptionContainer"] {margin-top: -.2rem; margin-bottom: -.35rem;}
[data-testid="stCaptionContainer"], .stCaption {color: var(--muted) !important;}
[data-testid="stMetric"] {
  background: linear-gradient(145deg, rgba(15,22,42,.96), rgba(8,11,22,.96));
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: .9rem 1rem;
}
[data-testid="stMetricValue"] {font-size: 1.55rem; color: var(--ink);}
[data-testid="stMetricDelta"] {color: var(--cyan) !important;}
.st-key-master_controls {
  position: sticky;
  top: 3.75rem;
  z-index: 999;
  margin: .2rem 0 .3rem;
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
  background: linear-gradient(145deg, rgba(17,25,47,.95), rgba(8,11,22,.95));
  border: 1px solid var(--line);
  border-radius: 16px;
  padding: 1rem 1.1rem;
  min-height: 126px;
  box-shadow: 0 14px 40px rgba(0,0,0,.24);
}
.portfolio-card .eyebrow {color: var(--violet); font-size: .72rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase;}
.portfolio-card .value {font-size: clamp(1.3rem, 2.2vw, 1.75rem); font-weight: 760; margin-top: .35rem; white-space: nowrap;}
.portfolio-card .detail {color: var(--muted); font-size: .82rem; margin-top: .25rem;}
.mode-chip {display:inline-block; padding:.1rem .38rem; border:1px solid rgba(103,215,255,.4); border-radius:999px; color:var(--cyan); font-size:.56rem; font-weight:800; letter-spacing:.08em;}
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
hr {border-color: var(--line);}
</style>
"""


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
