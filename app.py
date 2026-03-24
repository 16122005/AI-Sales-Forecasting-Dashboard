import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import datetime
import time

st.set_page_config(layout="wide")

# =========================
# THEME TOGGLE
# =========================
theme = st.sidebar.radio("Theme", ["Dark", "Light"])

if theme == "Dark":
    bg = "#020617"
    text = "white"
    card_bg = "rgba(15,23,42,0.7)"
else:
    bg = "#f8fafc"
    text = "black"
    card_bg = "white"

# =========================
# 🌌 ULTRA CSS
# =========================
st.markdown(f"""
<style>

/* Animated Background */
.stApp {{
    background: linear-gradient(-45deg, #020617, #0f172a, #020617);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
    color: {text};
}}

@keyframes gradient {{
    0% {{background-position: 0%}}
    50% {{background-position: 100%}}
    100% {{background-position: 0%}}
}}

/* Floating animation */
@keyframes float {{
    0% {{transform: translateY(0px);}}
    50% {{transform: translateY(-10px);}}
    100% {{transform: translateY(0px);}}
}}

/* KPI Cards */
.kpi {{
    padding: 25px;
    border-radius: 20px;
    text-align: center;
    background: {card_bg};
    backdrop-filter: blur(10px);
    border: 1px solid #9333ea;
    box-shadow: 0 0 20px #9333ea;
    animation: float 3s ease-in-out infinite;
    transition: 0.3s;
}}
.kpi:hover {{
    box-shadow: 0 0 40px #38bdf8;
    transform: scale(1.05);
}}

/* Chart Cards */
.card {{
    background: {card_bg};
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #334155;
    box-shadow: 0 0 15px #0ea5e9;
    margin-bottom: 15px;
    transition: 0.3s;
}}
.card:hover {{
    box-shadow: 0 0 30px #38bdf8;
    transform: scale(1.02);
}}

</style>
""", unsafe_allow_html=True)

# =========================
# LOGIN
# =========================
def login():
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "user" and p == "12345":
            st.session_state.logged_in = True
            st.session_state.user = u
            st.rerun()
        else:
            st.error("Invalid credentials")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login()
    st.stop()

# LOGOUT
if st.sidebar.button("🚪 Logout"):
    st.session_state.logged_in = False
    st.rerun()

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load():
    df = pd.read_csv("Superstore.csv", encoding="latin1")
    df['Order Date'] = pd.to_datetime(df['Order Date'])
    return df

df = load()

# =========================
# FILTERS
# =========================
st.sidebar.success(f"👤 {st.session_state.user}")

region = st.sidebar.multiselect("Region", df['Region'].unique(), df['Region'].unique())
category = st.sidebar.multiselect("Category", df['Category'].unique(), df['Category'].unique())

date_range = st.sidebar.date_input(
    "Date Range",
    [df['Order Date'].min(), df['Order Date'].max()]
)

df = df[
    (df['Region'].isin(region)) &
    (df['Category'].isin(category)) &
    (df['Order Date'] >= pd.to_datetime(date_range[0])) &
    (df['Order Date'] <= pd.to_datetime(date_range[1]))
]

# =========================
# HEADER
# =========================
st.title("🚀 AI SALES FORECASTING DASHBOARD")
st.caption(f"Updated: {datetime.datetime.now()}")

# =========================
# KPI (ANIMATED COUNTER)
# =========================
def animate_value(value, suffix=""):
    for i in range(0, 101, 10):
        st.empty().markdown(f"### {value * i / 100:.2f}{suffix}")
        time.sleep(0.01)

c1, c2, c3 = st.columns(3)

c1.markdown(f'<div class="kpi"><h2>{df["Sales"].sum()/1e6:.2f}M</h2><p>Sales</p></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi"><h2>{df["Quantity"].sum()/1000:.0f}K</h2><p>Quantity</p></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi"><h2>{df["Profit"].sum()/1000:.2f}K</h2><p>Profit</p></div>', unsafe_allow_html=True)

# =========================
# CHART THEME FIX
# =========================
def chart_theme(fig):
    fig.update_layout(
        font=dict(color=text),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# =========================
# MAIN CHARTS
# =========================
col1, col2, col3, col4 = st.columns(4)

with col1:
    fig = chart_theme(px.pie(df, names='Segment', values='Sales', hole=0.5))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = chart_theme(px.bar(df, x='Category', y='Sales', color='Category'))
    st.plotly_chart(fig, use_container_width=True)

with col3:
    monthly = df.groupby(df['Order Date'].dt.month)['Sales'].sum().reset_index()
    monthly.columns = ['Month','Sales']
    fig = chart_theme(px.line(monthly, x='Month', y='Sales'))
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = chart_theme(px.scatter_geo(df, locations="State", locationmode="USA-states",
                                     size="Sales", scope="usa"))
    st.plotly_chart(fig, use_container_width=True)

# =========================
# EXTRA CHARTS
# =========================
st.subheader("📊 Advanced Analytics")

c5, c6, c7 = st.columns(3)

with c5:
    st.plotly_chart(chart_theme(px.histogram(df, x="Sales")), use_container_width=True)

with c6:
    st.plotly_chart(chart_theme(px.box(df, y="Sales")), use_container_width=True)

with c7:
    st.plotly_chart(chart_theme(px.scatter(df, x="Sales", y="Profit",
                                           size="Quantity", color="Category")),
                    use_container_width=True)

# =========================
# FORECAST
# =========================
st.subheader("🔮 Forecast")

df2 = df.groupby('Order Date')['Sales'].sum().reset_index()
df2['lag'] = df2['Sales'].shift(1)
df2 = df2.dropna()

model = RandomForestRegressor()
model.fit(df2[['lag']], df2['Sales'])

df2['pred'] = model.predict(df2[['lag']])

st.plotly_chart(chart_theme(px.line(df2, x='Order Date', y=['Sales','pred'])))

# =========================
# MODEL COMPARISON
# =========================
lr = LinearRegression()
lr.fit(df2[['lag']], df2['Sales'])

rf_mae = mean_absolute_error(df2['Sales'], model.predict(df2[['lag']]))
lr_mae = mean_absolute_error(df2['Sales'], lr.predict(df2[['lag']]))

st.subheader("🤖 Model Comparison")

c1, c2 = st.columns(2)
c1.metric("Random Forest MAE", f"{rf_mae:.2f}")
c2.metric("Linear Regression MAE", f"{lr_mae:.2f}")

# =========================
# AI INSIGHTS
# =========================
st.subheader("🧠 AI Recommendations")

if df2['Sales'].iloc[-1] > df['Sales'].mean():
    st.success("📈 Sales increasing → Increase inventory")
else:
    st.warning("📉 Sales dropping → Offer discounts")

# =========================
# DOWNLOAD
# =========================
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("📥 Download Report", csv, "report.csv")

st.success("🔥  DASHBOARD READY")
