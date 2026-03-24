import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import datetime
import numpy as np

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Sales Forecast Dashboard",
    page_icon="🚀",
    layout="wide"
)

# =========================
# LOGIN
# =========================
def login():
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "admin" and p == "123456":
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
# THEME
# =========================
theme = st.sidebar.radio("Theme", ["Dark", "Light"])

text = "white" if theme == "Dark" else "black"
card_bg = "rgba(15,23,42,0.7)" if theme == "Dark" else "white"

# =========================
# CSS
# =========================
st.markdown(f"""
<style>
.stApp {{
    background: linear-gradient(-45deg, #020617, #0f172a, #020617);
    background-size: 400% 400%;
    animation: gradient 10s ease infinite;
    color: {text};
}}

.kpi {{
    padding: 20px;
    border-radius: 20px;
    text-align: center;
    background: {card_bg};
    box-shadow: 0 0 20px #9333ea;
}}

</style>
""", unsafe_allow_html=True)

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

df = df[(df['Region'].isin(region)) & (df['Category'].isin(category))]

# =========================
# HEADER
# =========================
st.title("🚀 AI SALES FORECASTING DASHBOARD")
st.caption(f"Updated: {datetime.datetime.now()}")

# =========================
# KPI
# =========================
c1, c2, c3 = st.columns(3)

c1.markdown(f'<div class="kpi"><h2>{df["Sales"].sum()/1e6:.2f}M</h2><p>Sales</p></div>', unsafe_allow_html=True)
c2.markdown(f'<div class="kpi"><h2>{df["Quantity"].sum()/1000:.0f}K</h2><p>Quantity</p></div>', unsafe_allow_html=True)
c3.markdown(f'<div class="kpi"><h2>{df["Profit"].sum()/1000:.2f}K</h2><p>Profit</p></div>', unsafe_allow_html=True)

# =========================
# CHARTS
# =========================
st.subheader("📊 Analytics")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(px.pie(df, names='Segment', values='Sales', hole=0.5), use_container_width=True)

with col2:
    st.plotly_chart(px.bar(df, x='Category', y='Sales', color='Category'), use_container_width=True)

# =========================
# ML MODEL
# =========================
@st.cache_resource
def train_model(data):
    df2 = data.groupby('Order Date')['Sales'].sum().reset_index()
    df2['lag1'] = df2['Sales'].shift(1)
    df2['lag2'] = df2['Sales'].shift(2)
    df2 = df2.dropna()

    X = df2[['lag1', 'lag2']]
    y = df2['Sales']

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)

    return model, df2

model, df2 = train_model(df)

df2['pred'] = model.predict(df2[['lag1', 'lag2']])

# =========================
# FORECAST
# =========================
st.subheader("🔮 Forecast")

fig = px.line(df2, x='Order Date', y=['Sales', 'pred'])

# Confidence band
df2['upper'] = df2['pred'] * 1.1
df2['lower'] = df2['pred'] * 0.9

fig.add_scatter(x=df2['Order Date'], y=df2['upper'], mode='lines', name='Upper Bound')
fig.add_scatter(x=df2['Order Date'], y=df2['lower'], mode='lines', name='Lower Bound')

st.plotly_chart(fig, use_container_width=True)

# =========================
# FEATURE IMPORTANCE
# =========================
st.subheader("📊 Feature Importance")

importance = pd.DataFrame({
    "Feature": ['lag1', 'lag2'],
    "Importance": model.feature_importances_
})

st.bar_chart(importance.set_index("Feature"))

# =========================
# MODEL COMPARISON
# =========================
lr = LinearRegression()
lr.fit(df2[['lag1', 'lag2']], df2['Sales'])

rf_mae = mean_absolute_error(df2['Sales'], model.predict(df2[['lag1', 'lag2']]))
lr_mae = mean_absolute_error(df2['Sales'], lr.predict(df2[['lag1', 'lag2']]))

st.subheader("🤖 Model Comparison")

c1, c2 = st.columns(2)
c1.metric("Random Forest MAE", f"{rf_mae:.2f}")
c2.metric("Linear Regression MAE", f"{lr_mae:.2f}")

# =========================
# FUTURE FORECAST
# =========================
st.subheader("📅 Future Prediction")

days = st.slider("Days", 1, 30, 7)

last = df2.iloc[-1]
future = []
current = last.copy()

for i in range(days):
    pred = model.predict([[current['lag1'], current['lag2']]])[0]
    future.append(pred)
    current['lag2'] = current['lag1']
    current['lag1'] = pred

future_df = pd.DataFrame({"Day": range(1, days+1), "Sales": future})
st.line_chart(future_df.set_index("Day"))

# =========================
# AI INSIGHTS
# =========================
st.subheader("🧠 AI Insights")

if df2['Sales'].iloc[-1] > df2['Sales'].mean():
    st.success("📈 Sales increasing → Increase inventory")
else:
    st.warning("📉 Sales dropping → Offer discounts")

# =========================
# DATA PREVIEW (NEW)
# =========================
with st.expander("📋 View Data"):
    st.dataframe(df.head())

# =========================
# DOWNLOAD
# =========================
st.download_button("📥 Download Report", df.to_csv(index=False), "report.csv")

st.success("🔥 INDUSTRY-LEVEL DASHBOARD READY")
