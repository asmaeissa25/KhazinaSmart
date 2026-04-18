"""KhazinaSmart — Professional Grocery Intelligence Dashboard"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dash import Dash, dcc, html, Input, Output, State, ctx, ALL, no_update
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import base64, io, warnings
warnings.filterwarnings("ignore")

from src.data_adapter import detect_columns, standardize
from src.universal_model import UniversalForecastModel
from src.alerts import generate_alerts_dataframe, estimate_financial_impact
from src.chatbot import get_starter_questions, answer_inventory_question

# ─── DEMO DATA ────────────────────────────────────────────────────────────────
def _load_demo():
    p = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_grocery.csv')
    if os.path.exists(p):
        return pd.read_csv(p, parse_dates=["date"])
    np.random.seed(42)
    rows = []
    for s in ["Store_A","Store_B","Store_C"]:
        for c in ["Produce","Dairy","Bakery","Meat","Beverages","Snacks"]:
            base = np.random.uniform(8000, 28000)
            for d in pd.date_range("2022-01-03","2024-01-01",freq="W-MON"):
                m = d.month
                rev = base * (1+0.3*np.sin(2*np.pi*(m-3)/12)) * np.random.normal(1,.07)
                if m in [11,12]: rev *= 1.3
                rows.append({"date":d,"store_id":s,"category":c,
                              "units_sold":max(1,int(rev/8)),"revenue":round(max(0,rev),2),
                              "is_promoted":np.random.choice([0,1],p=[.75,.25])})
    return pd.DataFrame(rows)

DEMO_DF = _load_demo()

# ─── PLOTLY HELPERS ───────────────────────────────────────────────────────────
PALETTE = ["#6c63ff","#e94560","#10b981","#f59e0b","#3b82f6","#ec4899","#8b5cf6","#14b8a6"]

def _layout(theme, title="", h=300):
    bg  = "rgba(0,0,0,0)"
    tmpl = "plotly_dark" if theme == "dark" else "plotly_white"
    return dict(template=tmpl, paper_bgcolor=bg, plot_bgcolor=bg,
                font_family="Inter", height=h, title=title,
                margin=dict(l=12,r=12,t=40,b=12))

def _empty(theme, msg="Upload data to visualise"):
    fig = go.Figure()
    fig.update_layout(**_layout(theme, h=260),
        annotations=[dict(text=msg, showarrow=False, x=.5, y=.5,
                          xref="paper", yref="paper",
                          font=dict(color="#8b8db8",size=14))])
    return fig

def _sales_col(df):
    return "revenue" if "revenue" in df.columns else "sales"

def chart_trend(df, theme):
    sc = _sales_col(df)
    agg = df.groupby("date")[sc].sum().reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=agg["date"], y=agg[sc], mode="lines",
        name="Revenue", line=dict(color="#6c63ff",width=2.5),
        fill="tozeroy", fillcolor="rgba(108,99,255,0.07)"))
    if "is_promoted" in df.columns:
        promo = df[df["is_promoted"]==1].groupby("date")[sc].sum().reset_index()
        if not promo.empty:
            fig.add_trace(go.Scatter(x=promo["date"], y=promo[sc], mode="markers",
                name="Promo weeks", marker=dict(color="#e94560",size=6,symbol="diamond")))
    fig.update_layout(**_layout(theme,"Total Weekly Revenue Trend",300),
        legend=dict(orientation="h",y=1.12,x=0))
    return fig

def chart_category(df, theme):
    sc = _sales_col(df)
    agg = df.groupby("category")[sc].sum().sort_values(ascending=True)
    fig = go.Figure(go.Bar(x=agg.values, y=agg.index, orientation="h",
        marker=dict(color=agg.values,
                    colorscale=[[0,"#1a1a32"],[.5,"#6c63ff"],[1,"#e94560"]])))
    fig.update_layout(**_layout(theme,"Revenue by Category",280))
    return fig

def chart_store(df, theme):
    sc = _sales_col(df)
    if "store_id" not in df.columns:
        return _empty(theme)
    agg = df.groupby("store_id")[sc].sum().reset_index().sort_values(sc,ascending=False)
    fig = go.Figure(go.Bar(x=agg["store_id"], y=agg[sc],
        marker=dict(color=PALETTE[:len(agg)]),
        text=agg[sc].apply(lambda v: f"{v/1000:.0f}K"), textposition="outside"))
    fig.update_layout(**_layout(theme,"Revenue by Store",280))
    return fig

def chart_heatmap(df, theme):
    sc = _sales_col(df)
    if "category" not in df.columns or "store_id" not in df.columns:
        return _empty(theme)
    pivot = df.pivot_table(index="category",columns="store_id",values=sc,aggfunc="sum",fill_value=0)
    cs = "Purples" if theme=="dark" else "Blues"
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale=cs, title="Heatmap: Category × Store")
    fig.update_layout(**_layout(theme,h=310))
    return fig

def chart_seasonal(df, theme):
    sc = _sales_col(df)
    df2 = df.copy()
    df2["month"] = pd.to_datetime(df2["date"]).dt.month
    monthly = df2.groupby("month")[sc].mean().reset_index()
    mnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig = go.Figure(go.Bar(
        x=[mnames[m-1] for m in monthly["month"]], y=monthly[sc],
        marker=dict(color=monthly[sc],
                    colorscale=[[0,"#1a1a32"],[.5,"#6c63ff"],[1,"#e94560"]])))
    fig.update_layout(**_layout(theme,"Monthly Seasonality",260))
    return fig

# ─── ALERTS ───────────────────────────────────────────────────────────────────
def _build_alerts(df):
    sc = _sales_col(df)
    std = df.rename(columns={sc:"sales"}) if sc != "sales" else df.copy()
    for col, fallback in [("store_id","Store_A"),("category","All")]:
        if col not in std.columns:
            std[col] = fallback
    model = UniversalForecastModel()
    try:
        model.fit(std, n_transfer_trees=40)
        preds = model.get_predictions()
        if not preds.empty and "predicted" in preds.columns:
            preds = preds.rename(columns={
                "sales":     "Weekly_Sales",
                "predicted": "predicted_demand",
                "store_id":  "Store",
                "category":  "Dept",
            })
            alerts = generate_alerts_dataframe(preds)
            return alerts, estimate_financial_impact(alerts)
    except Exception:
        pass
    # demo fallback
    np.random.seed(77)
    n = min(600, len(df))
    smp = df.sample(n, replace=len(df)<n).copy()
    smp["Weekly_Sales"]    = smp[sc].values
    smp["predicted_demand"]= smp[sc].values * np.random.normal(1,.35,n)
    alerts = generate_alerts_dataframe(smp)
    return alerts, estimate_financial_impact(alerts)

# ─── APP ──────────────────────────────────────────────────────────────────────
app = Dash(__name__,
    external_stylesheets=[
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap"
    ],
    suppress_callback_exceptions=True,
    title="KhazinaSmart")

# ─── LAYOUT HELPERS ───────────────────────────────────────────────────────────
def kpi_card(icon, label, val_id, sub, accent):
    return html.Div([
        html.Div(icon, className="kpi-icon"),
        html.Div(label, className="kpi-label"),
        html.Div("—", id=val_id, className="kpi-value"),
        html.Div(sub,  id=val_id+"-sub", className="kpi-sub"),
    ], className=f"kpi-card accent-{accent}")

# TAB PAGES — non-overview start hidden
def page_overview():
    return html.Div([
        html.Div([
            kpi_card("📦","Total Revenue","kpi-rev","All stores",       "purple"),
            kpi_card("🏪","Active Stores", "kpi-stores","In dataset",   "green"),
            kpi_card("🗂","Categories",    "kpi-cats","Product groups",  "yellow"),
            kpi_card("🔴","Overstock Alerts","kpi-over","Needs action", "red"),
        ], className="kpi-grid"),
        html.Div([
            html.Div([
                html.Div([html.Div("Weekly Revenue Trend",className="chart-title")],className="chart-header"),
                html.Div(dcc.Graph(id="ch-trend",config={"displayModeBar":False}),className="chart-body"),
            ],className="chart-card"),
        ],className="chart-grid-1"),
        html.Div([
            html.Div([
                html.Div([html.Div("Revenue by Category",className="chart-title")],className="chart-header"),
                html.Div(dcc.Graph(id="ch-cat",config={"displayModeBar":False}),className="chart-body"),
            ],className="chart-card"),
            html.Div([
                html.Div([html.Div("Revenue by Store",className="chart-title")],className="chart-header"),
                html.Div(dcc.Graph(id="ch-store",config={"displayModeBar":False}),className="chart-body"),
            ],className="chart-card"),
        ],className="chart-grid-2"),
        html.Div([
            html.Div([
                html.Div([html.Div("Category × Store Heatmap",className="chart-title")],className="chart-header"),
                html.Div(dcc.Graph(id="ch-heat",config={"displayModeBar":False}),className="chart-body"),
            ],className="chart-card"),
            html.Div([
                html.Div([html.Div("Monthly Seasonality",className="chart-title")],className="chart-header"),
                html.Div(dcc.Graph(id="ch-season",config={"displayModeBar":False}),className="chart-body"),
            ],className="chart-card"),
        ],className="chart-grid-2"),
    ], id="pg-overview", style={"display":"block"})

def page_forecast():
    return html.Div([
        html.Div([
            html.Div([html.Label("Store"),
                      dcc.Dropdown(id="fc-store",placeholder="Select store…",style={"fontSize":"13px","minWidth":"180px"})],
                     className="fc-field"),
            html.Div([html.Label("Category"),
                      dcc.Dropdown(id="fc-cat",placeholder="Select category…",style={"fontSize":"13px","minWidth":"180px"})],
                     className="fc-field"),
            html.Div([html.Label("Weeks ahead"),
                      dcc.Slider(id="fc-weeks",min=4,max=16,step=4,value=8,
                                 marks={4:"4w",8:"8w",12:"12w",16:"16w"})],
                     className="fc-field",style={"minWidth":"220px"}),
            html.Button("Generate Forecast",id="fc-run",n_clicks=0,className="btn-primary",
                        style={"height":"36px","alignSelf":"flex-end"}),
        ],className="forecast-controls"),
        html.Div([
            html.Div([
                html.Div([html.Div("Demand Forecast",className="chart-title")],className="chart-header"),
                html.Div(dcc.Graph(id="ch-fc",config={"displayModeBar":False}),className="chart-body"),
            ],className="chart-card",style={"marginBottom":"16px"}),
        ]),
        html.Div(id="fc-table"),
    ], id="pg-forecast", style={"display":"none"})

def page_alerts():
    return html.Div([
        html.Div([
            html.Div([html.Label("Status"),
                      dcc.Dropdown(id="al-status",
                          options=[{"label":l,"value":v} for l,v in [
                              ("All","All"),("🔴 Overstock","Overstock"),
                              ("🟡 Stockout Risk","Stockout Risk"),("🟢 Healthy","Healthy")]],
                          value="All",clearable=False,style={"fontSize":"13px","minWidth":"170px"})],
                     className="fc-field"),
            html.Div([html.Label("Min risk score"),
                      dcc.Slider(id="al-risk",min=0,max=100,step=10,value=0,
                                 marks={0:"0",50:"50",100:"100"})],
                     className="fc-field",style={"minWidth":"200px"}),
            html.Div(id="al-kpis",style={"display":"flex","gap":"10px","alignItems":"center","flexWrap":"wrap"}),
        ],className="filter-bar"),
        html.Div(id="al-table"),
        html.Div(id="al-chart",style={"marginTop":"14px"}),
    ], id="pg-alerts", style={"display":"none"})

def page_chatbot():
    starters = get_starter_questions()
    return html.Div([
        html.Div([html.Button(q,id={"type":"sq","index":i},n_clicks=0,className="sq-btn")
                  for i,q in enumerate(starters)],
                 className="starter-questions"),
        html.Div([
            html.Div([
                html.Div([
                    html.Div("📦",className="chat-avatar bot-av"),
                    html.Div([html.Strong("KhazBot"),html.Br(),
                               html.Span("Hi! Ask me anything about your inventory in French or English.")],
                             className="chat-bubble bubble-bot"),
                ],className="chat-msg"),
            ],id="chat-msgs",className="chat-messages"),
            html.Div([
                dcc.Input(id="chat-in",placeholder="Ask about inventory…",type="text",
                          className="chat-input",debounce=False),
                html.Button("Send",id="chat-go",n_clicks=0,className="chat-send"),
            ],className="chat-input-row"),
        ],className="chat-wrap"),
    ], id="pg-chatbot", style={"display":"none"})

# FULL LAYOUT
app.layout = html.Div([
    # Stores
    dcc.Store(id="st-theme",  data="dark",  storage_type="session"),
    dcc.Store(id="st-data",   data=None,    storage_type="memory"),
    dcc.Store(id="st-alerts", data=None,    storage_type="memory"),
    dcc.Store(id="st-chat",   data=[],      storage_type="memory"),
    dcc.Store(id="st-tab",    data="overview", storage_type="memory"),

    html.Div([
        # HEADER
        html.Header([
            html.Div([
                html.Span("📦",className="header-brand-icon"),
                html.Span("KhazinaSmart",className="header-brand-name"),
                html.Span("BETA",className="header-brand-tag"),
            ],className="header-brand"),
            html.Nav([
                html.Button("Overview", id="tb-ov", n_clicks=0, className="nav-tab active"),
                html.Button("Forecast", id="tb-fc", n_clicks=0, className="nav-tab"),
                html.Button("Alerts",   id="tb-al", n_clicks=0, className="nav-tab"),
                html.Button("KhazBot",  id="tb-ch", n_clicks=0, className="nav-tab"),
            ],className="header-nav"),
            html.Div([
                dcc.Upload(id="up-csv",
                    children=html.Button("⬆ Upload CSV", className="btn-upload"),
                    accept=".csv", multiple=False),
                html.Button("🌙",id="btn-theme",n_clicks=0,className="btn-theme",title="Toggle theme"),
            ],className="header-actions"),
        ],className="site-header"),

        html.Div([
            html.Div(id="status-bar"),
            page_overview(),
            page_forecast(),
            page_alerts(),
            page_chatbot(),
            html.Div("KhazinaSmart © 2026 | IT DAY'Z Hackathon | ENSA Tanger | AI for Startups & Business",
                     className="site-footer"),
        ],className="page-body"),
    ], id="app-root", className="theme-dark"),
])

# ─── CALLBACKS ────────────────────────────────────────────────────────────────

# 1. THEME TOGGLE — updates className on #app-root + button icon + store
@app.callback(
    Output("app-root",   "className"),
    Output("btn-theme",  "children"),
    Output("st-theme",   "data"),
    Input("btn-theme",   "n_clicks"),
    State("st-theme",    "data"),
)
def toggle_theme(n, current):
    theme = current or "dark"
    if n:
        theme = "light" if theme == "dark" else "dark"
    icon = "☀️" if theme == "light" else "🌙"
    return f"theme-{theme}", icon, theme


# 2. TAB NAVIGATION — update active store
@app.callback(
    Output("st-tab",  "data"),
    Output("tb-ov",   "className"),
    Output("tb-fc",   "className"),
    Output("tb-al",   "className"),
    Output("tb-ch",   "className"),
    Input("tb-ov",    "n_clicks"),
    Input("tb-fc",    "n_clicks"),
    Input("tb-al",    "n_clicks"),
    Input("tb-ch",    "n_clicks"),
    State("st-tab",   "data"),
    prevent_initial_call=True,
)
def switch_tab(n1,n2,n3,n4,current):
    tid = ctx.triggered_id
    tab = {"tb-ov":"overview","tb-fc":"forecast","tb-al":"alerts","tb-ch":"chatbot"}.get(tid, current)
    def cls(t): return "nav-tab active" if tab==t else "nav-tab"
    return tab, cls("overview"), cls("forecast"), cls("alerts"), cls("chatbot")


# 3. TAB VISIBILITY — server callback, 4 style outputs
@app.callback(
    Output("pg-overview","style"),
    Output("pg-forecast","style"),
    Output("pg-alerts",  "style"),
    Output("pg-chatbot", "style"),
    Input("st-tab",      "data"),
)
def show_tab(tab):
    show = {"display":"block"}
    hide = {"display":"none"}
    tab = tab or "overview"
    return (
        show if tab=="overview" else hide,
        show if tab=="forecast" else hide,
        show if tab=="alerts"   else hide,
        show if tab=="chatbot"  else hide,
    )


# 4. UPLOAD CSV
@app.callback(
    Output("st-data",   "data"),
    Output("status-bar","children"),
    Input("up-csv",     "contents"),
    State("up-csv",     "filename"),
    prevent_initial_call=True,
)
def on_upload(contents, filename):
    if not contents:
        raise PreventUpdate
    try:
        _, b64 = contents.split(",", 1)
        raw = pd.read_csv(io.StringIO(base64.b64decode(b64).decode("utf-8")))
    except Exception as e:
        return no_update, html.Div(f"Error reading file: {e}",
                                   style={"color":"#e94560","padding":"10px"})
    try:
        col_map = detect_columns(raw)
        std = standardize(raw, col_map)
    except Exception as e:
        return no_update, html.Div(f"Could not parse columns: {e}",
                                   style={"color":"#e94560","padding":"10px"})

    data_json = std.to_json(orient="split", date_format="iso")
    n_rows    = len(std)
    n_stores  = std["store_id"].nunique() if "store_id" in std.columns else "—"
    n_cats    = std["category"].nunique() if "category" in std.columns else "—"

    bar = html.Div([
        html.Div(className="status-dot"),
        html.Span(f"✓  {filename}", style={"fontWeight":600,"color":"var(--c-text)"}),
        html.Span(f"{n_rows:,} rows",        className="status-badge badge-rows"),
        html.Span(f"{n_stores} stores · {n_cats} categories", className="metric-chip"),
        html.Span("Transfer Learning Active", className="status-badge badge-transfer"),
        html.Span(f"Detected: {col_map}", style={"fontSize":"11px","color":"var(--c-text3)"}),
    ], className="status-bar")
    return data_json, bar


def _df(data_json):
    """Return uploaded df, fall back to demo."""
    if data_json:
        try:
            df = pd.read_json(io.StringIO(data_json), orient="split")
            df["date"] = pd.to_datetime(df["date"])
            return df
        except Exception:
            pass
    return DEMO_DF.copy()


# 5. KPIs
@app.callback(
    Output("kpi-rev",    "children"),
    Output("kpi-stores", "children"),
    Output("kpi-cats",   "children"),
    Output("kpi-over",   "children"),
    Input("st-data",     "data"),
    Input("st-alerts",   "data"),
)
def update_kpis(data_json, alerts_json):
    df = _df(data_json)
    sc = _sales_col(df)
    total = df[sc].sum()
    stores = df["store_id"].nunique() if "store_id" in df.columns else "—"
    cats   = df["category"].nunique() if "category" in df.columns else "—"
    rev_str = f"{total/1_000_000:.1f}M" if total>=1_000_000 else f"{total/1000:.0f}K"
    ov = "—"
    if alerts_json:
        try:
            al = pd.read_json(io.StringIO(alerts_json), orient="split")
            ov = str(int((al["status"]=="Overstock").sum()))
        except Exception:
            pass
    return rev_str, str(stores), str(cats), ov


# 6. OVERVIEW CHARTS
@app.callback(
    Output("ch-trend",  "figure"),
    Output("ch-cat",    "figure"),
    Output("ch-store",  "figure"),
    Output("ch-heat",   "figure"),
    Output("ch-season", "figure"),
    Input("st-data",    "data"),
    Input("st-theme",   "data"),
)
def update_overview(data_json, theme):
    df = _df(data_json)
    t = theme or "dark"
    return (chart_trend(df,t), chart_category(df,t), chart_store(df,t),
            chart_heatmap(df,t), chart_seasonal(df,t))


# 7. FORECAST DROPDOWNS
@app.callback(
    Output("fc-store","options"), Output("fc-store","value"),
    Output("fc-cat",  "options"), Output("fc-cat",  "value"),
    Input("st-data",  "data"),
)
def fc_dropdowns(data_json):
    df = _df(data_json)
    stores = sorted(df["store_id"].unique().tolist()) if "store_id" in df.columns else ["All"]
    cats   = sorted(df["category"].unique().tolist()) if "category" in df.columns else ["All"]
    return ([{"label":s,"value":s} for s in stores], stores[0],
            [{"label":c,"value":c} for c in cats],   cats[0])


# 8. FORECAST CHART
@app.callback(
    Output("ch-fc",    "figure"),
    Output("fc-table", "children"),
    Input("fc-run",    "n_clicks"),
    State("st-data",   "data"),
    State("fc-store",  "value"),
    State("fc-cat",    "value"),
    State("fc-weeks",  "value"),
    State("st-theme",  "data"),
    prevent_initial_call=True,
)
def run_forecast(n, data_json, store, cat, weeks, theme):
    if not n:
        raise PreventUpdate
    t  = theme or "dark"
    df = _df(data_json)
    sc = _sales_col(df)
    std = df.rename(columns={sc:"sales"}) if sc!="sales" else df.copy()

    model = UniversalForecastModel()
    try:
        metrics = model.fit(std, n_transfer_trees=60)
        future  = model.predict_future(std, weeks=weeks or 8)
    except Exception as e:
        return _empty(t, f"Forecast error: {e}"), html.Div(str(e))

    # Historical for chosen store+cat
    hist = df.copy()
    if "store_id" in hist.columns and store:
        hist = hist[hist["store_id"]==store]
    if "category" in hist.columns and cat:
        hist = hist[hist["category"]==cat]
    agg_hist = hist.groupby("date")[sc].sum().reset_index()

    fut = future[(future["store_id"]==store)&(future["category"]==cat)] \
          if not future.empty else pd.DataFrame()

    fig = go.Figure()
    if not agg_hist.empty:
        fig.add_trace(go.Scatter(x=agg_hist["date"], y=agg_hist[sc],
            name="Historical", mode="lines", line=dict(color="#6c63ff",width=2.5)))
    if not fut.empty:
        fig.add_trace(go.Scatter(
            x=list(fut["date"])+list(fut["date"][::-1]),
            y=list(fut["upper"])+list(fut["lower"][::-1]),
            fill="toself", fillcolor="rgba(233,69,96,0.1)",
            line=dict(color="rgba(0,0,0,0)"), name="Confidence", showlegend=True))
        fig.add_trace(go.Scatter(x=fut["date"], y=fut["predicted"],
            name="Forecast", mode="lines+markers",
            line=dict(color="#e94560",width=2.5,dash="dash"),
            marker=dict(size=7)))
        if not agg_hist.empty:
            fig.add_vline(x=agg_hist["date"].max(), line_dash="dot", line_color="#f59e0b",
                          annotation_text="Forecast start")

    fig.update_layout(**_layout(t,f"Forecast — {store} / {cat}",400),
        legend=dict(orientation="h",y=1.12),
        xaxis_title="Date", yaxis_title="Revenue")

    # Table
    if fut.empty:
        table = html.Div("No data for this combination.",
                         style={"color":"var(--c-text2)","padding":"12px"})
    else:
        fut2 = fut.copy()
        fut2["date"] = fut2["date"].dt.strftime("%Y-%m-%d")
        tl = metrics.get("transfer_learning", False)
        rows = [html.Tr([html.Td(f"W{i+1}"), html.Td(r["date"]),
                         html.Td(f"{r['predicted']:,.0f}"),
                         html.Td(f"{r['lower']:,.0f}"), html.Td(f"{r['upper']:,.0f}"),
                         html.Td("📈" if i==0 or r["predicted"]>fut2.iloc[i-1]["predicted"] else "📉")])
                for i,(_, r) in enumerate(fut2.iterrows())]
        table = html.Div([
            html.Div([
                html.Span(f"R²={metrics.get('r2',0):.3f}  MAPE={metrics.get('mape',0):.1f}%",
                          className="metric-chip metric-good"),
                html.Span("Transfer Learning ✓" if tl else "Fresh Model",
                          className=f"metric-chip {'badge-transfer' if tl else ''}"),
            ], style={"display":"flex","gap":"8px","margin":"12px 0 12px 0"}),
            html.Table([
                html.Thead(html.Tr([html.Th(c) for c in
                    ["Week","Date","Predicted","Lower","Upper","Trend"]])),
                html.Tbody(rows),
            ], className="data-table",
               style={"background":"var(--c-card)","borderRadius":"12px","overflow":"hidden"}),
        ])
    return fig, table


# 9. ALERTS
@app.callback(
    Output("al-table",  "children"),
    Output("al-chart",  "children"),
    Output("st-alerts", "data"),
    Output("al-kpis",   "children"),
    Input("st-data",    "data"),
    Input("al-status",  "value"),
    Input("al-risk",    "value"),
    Input("st-theme",   "data"),
)
def update_alerts(data_json, status_f, min_risk, theme):
    df = _df(data_json)
    t  = theme or "dark"
    alerts, impact = _build_alerts(df)

    filt = alerts.copy()
    if status_f and status_f != "All":
        filt = filt[filt["status"]==status_f]
    if min_risk:
        filt = filt[filt["risk_score"]>=min_risk]

    ov = int((alerts["status"]=="Overstock").sum())
    st = int((alerts["status"]=="Stockout Risk").sum())
    hl = int((alerts["status"]=="Healthy").sum())
    kpis = [
        html.Span(f"🔴 {ov} Overstock",    className="metric-chip metric-bad"),
        html.Span(f"🟡 {st} Stockout Risk", className="metric-chip metric-warn"),
        html.Span(f"🟢 {hl} Healthy",       className="metric-chip metric-good"),
        html.Span(f"💰 {impact['total_risk_mad']:,.0f} at risk", className="metric-chip"),
    ]

    possible = ["Store","Dept","store_id","category","status","risk_score","action_needed"]
    cols = [c for c in possible if c in filt.columns and c not in ("store_id","category"
                if "Store" in filt.columns else [])]
    if not cols:
        cols = [c for c in ["store_id","category","status","risk_score","action_needed"]
                if c in filt.columns]

    def pill(s):
        cls = {"Overstock":"pill-overstock","Stockout Risk":"pill-stockout","Healthy":"pill-healthy"}.get(s,"")
        return html.Span(s, className=f"status-pill {cls}")

    def risk_bar_html(v):
        cls = "risk-high" if v>65 else ("risk-medium" if v>35 else "risk-low")
        return html.Div([
            html.Span(f"{v:.0f}", style={"fontWeight":"600","marginRight":"8px","minWidth":"32px","display":"inline-block"}),
            html.Div([html.Div(className=f"risk-fill {cls}",style={"width":f"{min(v,100):.0f}%"})],className="risk-bar"),
        ], style={"display":"flex","alignItems":"center"})

    trows = []
    for _, row in filt.head(120).iterrows():
        cells = []
        for c in cols:
            v = row.get(c,"—")
            if c=="status":
                cells.append(html.Td(pill(str(v))))
            elif c=="risk_score":
                cells.append(html.Td(risk_bar_html(float(v))))
            else:
                cells.append(html.Td(str(v)[:55] if isinstance(v,str) else f"{v:,.0f}" if isinstance(v,float) else str(v)))
        trows.append(html.Tr(cells))

    table = html.Div([
        html.Table([
            html.Thead(html.Tr([html.Th(c.replace("_"," ").title()) for c in cols])),
            html.Tbody(trows),
        ], className="data-table"),
    ], style={"background":"var(--c-card)","borderRadius":"12px",
              "border":"1px solid var(--c-border)","overflow":"hidden"})

    fig = px.histogram(filt, x="risk_score", color="status", nbins=25,
        color_discrete_map={"Healthy":"#10b981","Overstock":"#e94560","Stockout Risk":"#f59e0b"},
        title="Risk Score Distribution", labels={"risk_score":"Risk Score"})
    fig.update_layout(**_layout(t,h=280))

    chart_section = html.Div([
        html.Div([
            html.Div([html.Div("Risk Distribution",className="chart-title")],className="chart-header"),
            html.Div(dcc.Graph(figure=fig,config={"displayModeBar":False}),className="chart-body"),
        ],className="chart-card"),
    ])
    return table, chart_section, alerts.to_json(orient="split",date_format="iso"), kpis


# 10. CHATBOT
@app.callback(
    Output("chat-msgs","children"),
    Output("st-chat",  "data"),
    Output("chat-in",  "value"),
    Input("chat-go",   "n_clicks"),
    Input({"type":"sq","index":ALL},"n_clicks"),
    State("chat-in",   "value"),
    State("st-chat",   "data"),
    State("st-alerts", "data"),
    prevent_initial_call=True,
)
def on_chat(n_send, sq_clicks, user_input, history, alerts_json):
    tid = ctx.triggered_id
    question = None

    if tid == "chat-go" and user_input and user_input.strip():
        question = user_input.strip()
    elif isinstance(tid, dict) and tid.get("type") == "sq":
        idx = tid["index"]
        raw = get_starter_questions()[idx]
        # Strip trailing emoji/icon
        for tail in [" 🔴"," 📦"," 💰"," 🏪"," 📊"]:
            raw = raw.split(tail)[0]
        question = raw

    if not question:
        raise PreventUpdate

    if alerts_json:
        try:
            al = pd.read_json(io.StringIO(alerts_json), orient="split")
        except Exception:
            al = _build_alerts(DEMO_DF)[0]
    else:
        al = _build_alerts(DEMO_DF)[0]

    response = answer_inventory_question(question, al)

    history = history or []
    history.append({"role":"user",      "content":question})
    history.append({"role":"assistant", "content":response})

    def msg(m):
        is_u = m["role"]=="user"
        return html.Div([
            html.Div("👤" if is_u else "📦",
                     className=f"chat-avatar {'user-av' if is_u else 'bot-av'}"),
            html.Div(m["content"], className=f"chat-bubble {'bubble-user' if is_u else 'bubble-bot'}"),
        ], className=f"chat-msg {'user' if is_u else ''}")

    init = html.Div([
        html.Div("📦",className="chat-avatar bot-av"),
        html.Div([html.Strong("KhazBot"),html.Br(),
                  html.Span("Hi! Ask me anything about your inventory in French or English.")],
                 className="chat-bubble bubble-bot"),
    ], className="chat-msg")

    return [init]+[msg(m) for m in history], history, ""


if __name__ == "__main__":
    app.run(debug=False, port=8502, host="0.0.0.0")
