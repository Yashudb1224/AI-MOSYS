# app_modern_final_REFRESH_RULES_ONE_TEMP.py

import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- COLOR DEFINITIONS ---
SIDEBAR_BG = "#1c1c1e"
SIDEBAR_TEXT = "#ffffff"
MAIN_BG = "#f0f2f6"
MAIN_TEXT = "#1c1c1e"
PRIMARY_COLOR = "#0072b5"

# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
st.set_page_config(page_title="Memory Bus Optimizer 🚀", layout="wide")

# -------------------------------------------------------
# CSS STYLING
# -------------------------------------------------------
st.markdown(f"""
<style>
.stApp {{
    background-color: {MAIN_BG};
    color: {MAIN_TEXT};
}}
h1, h2, h3, h4, p, label {{
    color: {MAIN_TEXT} !important;
}}
.stMetric {{
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid {PRIMARY_COLOR};
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
}}
[data-testid="stMetricValue"] {{
    color: {PRIMARY_COLOR};
    font-size: 1.8rem;
}}
/* Sidebar */
.css-1lcbmhc, .e1fqkh3o10 {{
    background-color: {SIDEBAR_BG};
}}
.stSidebar * {{
    color: {SIDEBAR_TEXT} !important;
}}
/* Tables */
.stDataFrame > div {{
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #e6e6e6;
}}
.stDataFrame table {{
    background-color: #ffffff;
    color: {MAIN_TEXT};
}}
.stDataFrame thead th {{
    background-color: #f8f8f8;
    border-bottom: 2px solid #e6e6e6;
}}
.stDataFrame tbody tr:nth-child(even) {{
    background-color: #fbfbfb;
}}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# LOAD DATASET  (your full dataset goes here)
# -------------------------------------------------------
data_string = """
Frequency,Temperature,Command_Sequence,Optimal_Spacing,Optimal_Refresh_Rate (μs)
2133,30,R-R,4,7.8
2133,30,R-W,4,7.8
2133,30,W-R,8,7.8
2133,30,W-W,4,7.8
2133,30,R-R-R,4-4,7.8
2133,30,R-R-W,4-4,7.8
2133,30,R-W-R,4-8,7.8
2133,30,R-W-W,4-4,7.8
2133,30,W-R-R,8-4,7.8
2133,30,W-R-W,8-4,7.8
2133,30,W-W-R,4-8,7.8
2133,30,W-W-W,4-4,7.8
2133,30,R-R-R-R,4-4-4,7.8
2133,30,R-R-R-W,4-4-4,7.8
2133,30,R-R-W-R,4-4-8,7.8
2133,30,R-R-W-W,4-4-4,7.8
2133,30,R-W-R-R,4-8-4,7.8
2133,30,R-W-R-W,4-8-4,7.8
2133,30,R-W-W-R,4-4-8,7.8
2133,30,R-W-W-W,4-4-4,7.8
2133,30,W-R-R-R,8-4-4,7.8
2133,30,W-R-R-W,8-4-4,7.8
2133,30,W-R-W-R,8-4-8,7.8
2133,30,W-R-W-W,8-4-4,7.8
2133,30,W-W-R-R,4-8-4,7.8
2133,30,W-W-R-W,4-8-4,7.8
2133,30,W-W-W-R,4-4-8,7.8
2133,30,W-W-W-W,4-4-4,7.8
2133,50,R-R,4,7.8
2133,50,R-W,4,7.8
2133,50,W-R,8,7.8
2133,50,W-W,4,7.8
2133,50,R-R-R,4-4,7.8
2133,50,R-R-W,4-4,7.8
2133,50,R-W-R,4-8,7.8
2133,50,R-W-W,4-4,7.8
2133,50,W-R-R,8-4,7.8
2133,50,W-R-W,8-4,7.8
2133,50,W-W-R,4-8,7.8
2133,50,W-W-W,4-4,7.8
2133,50,R-R-R-R,4-4-4,7.8
2133,50,R-R-R-W,4-4-4,7.8
2133,50,R-R-W-R,4-4-8,7.8
2133,50,R-R-W-W,4-4-4,7.8
2133,50,R-W-R-R,4-8-4,7.8
2133,50,R-W-R-W,4-8-4,7.8
2133,50,R-W-W-R,4-4-8,7.8
2133,50,R-W-W-W,4-4-4,7.8
2133,50,W-R-R-R,8-4-4,7.8
2133,50,W-R-R-W,8-4-4,7.8
2133,50,W-R-W-R,8-4-8,7.8
2133,50,W-R-W-W,8-4-4,7.8
2133,50,W-W-R-R,4-8-4,7.8
2133,50,W-W-R-W,4-8-4,7.8
2133,50,W-W-W-R,4-4-8,7.8
2133,50,W-W-W-W,4-4-4,7.8
2666,30,R-R,4,7.8
2666,30,R-W,4,7.8
2666,30,W-R,8,7.8
2666,30,W-W,4,7.8
2666,30,R-R-R,4-4,7.8
2666,30,R-R-W,4-4,7.8
2666,30,R-W-R,4-8,7.8
2666,30,R-W-W,4-4,7.8
2666,30,W-R-R,8-4,7.8
2666,30,W-R-W,8-4,7.8
2666,30,W-W-R,4-8,7.8
2666,30,W-W-W,4-4,7.8
2666,30,R-R-R-R,4-4-4,7.8
2666,30,R-R-R-W,4-4-4,7.8
2666,30,R-R-W-R,4-4-8,7.8
2666,30,R-R-W-W,4-4-4,7.8
2666,30,R-W-R-R,4-8-4,7.8
2666,30,R-W-R-W,4-8-4,7.8
2666,30,R-W-W-R,4-4-8,7.8
2666,30,R-W-W-W,4-4-4,7.8
2666,30,W-R-R-R,8-4-4,7.8
2666,30,W-R-R-W,8-4-4,7.8
2666,30,W-R-W-R,8-4-8,7.8
2666,30,W-R-W-W,8-4-4,7.8
2666,30,W-W-R-R,4-8-4,7.8
2666,30,W-W-R-W,4-8-4,7.8
2666,30,W-W-W-R,4-4-8,7.8
2666,30,W-W-W-W,4-4-4,7.8
2666,50,R-R,4,7.8
2666,50,R-W,4,7.8
2666,50,W-R,8,7.8
2666,50,W-W,4,7.8
2666,50,R-R-R,4-4,7.8
2666,50,R-R-W,4-4,7.8
2666,50,R-W-R,4-8,7.8
2666,50,R-W-W,4-4,7.8
2666,50,W-R-R,8-4,7.8
2666,50,W-R-W,8-4,7.8
2666,50,W-W-R,4-8,7.8
2666,50,W-W-W,4-4,7.8
2666,50,R-R-R-R,4-4-4,7.8
2666,50,R-R-R-W,4-4-4,7.8
2666,50,R-R-W-R,4-4-8,7.8
2666,50,R-R-W-W,4-4-4,7.8
2666,50,R-W-R-R,4-8-4,7.8
2666,50,R-W-R-W,4-8-4,7.8
2666,50,R-W-W-R,4-4-8,7.8
2666,50,R-W-W-W,4-4-4,7.8
2666,50,W-R-R-R,8-4-4,7.8
2666,50,W-R-R-W,8-4-4,7.8
2666,50,W-R-W-R,8-4-8,7.8
2666,50,W-R-W-W,8-4-4,7.8
2666,50,W-W-R-R,4-8-4,7.8
2666,50,W-W-R-W,4-8-4,7.8
2666,50,W-W-W-R,4-4-8,7.8
2666,50,W-W-W-W,4-4-4,7.8
3200,30,R-R,4,7.8
3200,30,R-W,4,7.8
3200,30,W-R,8,7.8
3200,30,W-W,4,7.8
3200,30,R-R-R,4-4,7.8
3200,30,R-R-W,4-4,7.8
3200,30,R-W-R,4-8,7.8
3200,30,R-W-W,4-4,7.8
3200,30,W-R-R,8-4,7.8
3200,30,W-R-W,8-4,7.8
3200,30,W-W-R,4-8,7.8
3200,30,W-W-W,4-4,7.8
3200,30,R-R-R-R,4-4-4,7.8
3200,30,R-R-R-W,4-4-4,7.8
3200,30,R-R-W-R,4-4-8,7.8
3200,30,R-R-W-W,4-4-4,7.8
3200,30,R-W-R-R,4-8-4,7.8
3200,30,R-W-R-W,4-8-4,7.8
3200,30,R-W-W-R,4-4-8,7.8
3200,30,R-W-W-W,4-4-4,7.8
3200,30,W-R-R-R,8-4-4,7.8
3200,30,W-R-R-W,8-4-4,7.8
3200,30,W-R-W-R,8-4-8,7.8
3200,30,W-R-W-W,8-4-4,7.8
3200,30,W-W-R-R,4-8-4,7.8
3200,30,W-W-R-W,4-8-4,7.8
3200,30,W-W-W-R,4-4-8,7.8
3200,30,W-W-W-W,4-4-4,7.8
3200,50,R-R,4,7.8
3200,50,R-W,4,7.8
3200,50,W-R,8,7.8
3200,50,W-W,4,7.8
3200,50,R-R-R,4-4,7.8
3200,50,R-R-W,4-4,7.8
3200,50,R-W-R,4-8,7.8
3200,50,R-W-W,4-4,7.8
3200,50,W-R-R,8-4,7.8
3200,50,W-R-W,8-4,7.8
3200,50,W-W-R,4-8,7.8
3200,50,W-W-W,4-4,7.8
3200,50,R-R-R-R,4-4-4,7.8
3200,50,R-R-R-W,4-4-4,7.8
3200,50,R-R-W-R,4-4-8,7.8
3200,50,R-R-W-W,4-4-4,7.8
3200,50,R-W-R-R,4-8-4,7.8
3200,50,R-W-R-W,4-8-4,7.8
3200,50,R-W-W-R,4-4-8,7.8
3200,50,R-W-W-W,4-4-4,7.8
3200,50,W-R-R-R,8-4-4,7.8
3200,50,W-R-R-W,8-4-4,7.8
3200,50,W-R-W-R,8-4-8,7.8
3200,50,W-R-W-W,8-4-4,7.8
3200,50,W-W-R-R,4-8-4,7.8
3200,50,W-W-R-W,4-8-4,7.8
3200,50,W-W-W-R,4-4-8,7.8
3200,50,W-W-W-W,4-4-4,7.8
3733,30,R-R,4,7.8
3733,30,R-W,4,7.8
3733,30,W-R,8,7.8
3733,30,W-W,4,7.8
3733,30,R-R-R,4-4,7.8
3733,30,R-R-W,4-4,7.8
3733,30,R-W-R,4-8,7.8
3733,30,R-W-W,4-4,7.8
3733,30,W-R-R,8-4,7.8
3733,30,W-R-W,8-4,7.8
3733,30,W-W-R,4-8,7.8
3733,30,W-W-W,4-4,7.8
3733,30,R-R-R-R,4-4-4,7.8
3733,30,R-R-R-W,4-4-4,7.8
3733,30,R-R-W-R,4-4-8,7.8
3733,30,R-R-W-W,4-4-4,7.8
3733,30,R-W-R-R,4-8-4,7.8
3733,30,R-W-R-W,4-8-4,7.8
3733,30,R-W-W-R,4-4-8,7.8
3733,30,R-W-W-W,4-4-4,7.8
3733,30,W-R-R-R,8-4-4,7.8
3733,30,W-R-R-W,8-4-4,7.8
3733,30,W-R-W-R,8-4-8,7.8
3733,30,W-R-W-W,8-4-4,7.8
3733,30,W-W-R-R,4-8-4,7.8
3733,30,W-W-R-W,4-8-4,7.8
3733,30,W-W-W-R,4-4-8,7.8
3733,30,W-W-W-W,4-4-4,7.8
3733,50,R-R,4,7.8
3733,50,R-W,4,7.8
3733,50,W-R,8,7.8
3733,50,W-W,4,7.8
3733,50,R-R-R,4-4,7.8
3733,50,R-R-W,4-4,7.8
3733,50,R-W-R,4-8,7.8
3733,50,R-W-W,4-4,7.8
3733,50,W-R-R,8-4,7.8
3733,50,W-R-W,8-4,7.8
3733,50,W-W-R,4-8,7.8
3733,50,W-W-W,4-4,7.8
3733,50,R-R-R-R,4-4-4,7.8
3733,50,R-R-R-W,4-4-4,7.8
3733,50,R-R-W-R,4-4-8,7.8
3733,50,R-R-W-W,4-4-4,7.8
3733,50,R-W-R-R,4-8-4,7.8
3733,50,R-W-R-W,4-8-4,7.8
3733,50,R-W-W-R,4-4-8,7.8
3733,50,R-W-W-W,4-4-4,7.8
3733,50,W-R-R-R,8-4-4,7.8
3733,50,W-R-R-W,8-4-4,7.8
3733,50,W-R-W-R,8-4-8,7.8
3733,50,W-R-W-W,8-4-4,7.8
3733,50,W-W-R-R,4-8-4,7.8
3733,50,W-W-R-W,4-8-4,7.8
3733,50,W-W-W-R,4-4-8,7.8
3733,50,W-W-W-W,4-4-4,7.8
4266,30,R-R,4,7.8
4266,30,R-W,4,7.8
4266,30,W-R,8,7.8
4266,30,W-W,4,7.8
4266,30,R-R-R,4-4,7.8
4266,30,R-R-W,4-4,7.8
4266,30,R-W-R,4-8,7.8
4266,30,R-W-W,4-4,7.8
4266,30,W-R-R,8-4,7.8
4266,30,W-R-W,8-4,7.8
4266,30,W-W-R,4-8,7.8
4266,30,W-W-W,4-4,7.8
4266,30,R-R-R-R,4-4-4,7.8
4266,30,R-R-R-W,4-4-4,7.8
4266,30,R-R-W-R,4-4-8,7.8
4266,30,R-R-W-W,4-4-4,7.8
4266,30,R-W-R-R,4-8-4,7.8
4266,30,R-W-R-W,4-8-4,7.8
4266,30,R-W-W-R,4-4-8,7.8
4266,30,R-W-W-W,4-4-4,7.8
4266,30,W-R-R-R,8-4-4,7.8
4266,30,W-R-R-W,8-4-4,7.8
4266,30,W-R-W-R,8-4-8,7.8
4266,30,W-R-W-W,8-4-4,7.8
4266,30,W-W-R-R,4-8-4,7.8
4266,30,W-W-R-W,4-8-4,7.8
4266,30,W-W-W-R,4-4-8,7.8
4266,30,W-W-W-W,4-4-4,7.8
4266,50,R-R,4,7.8
4266,50,R-W,4,7.8
4266,50,W-R,8,7.8
4266,50,W-W,4,7.8
4266,50,R-R-R,4-4,7.8
4266,50,R-R-W,4-4,7.8
4266,50,R-W-R,4-8,7.8
4266,50,R-W-W,4-4,7.8
4266,50,W-R-R,8-4,7.8
4266,50,W-R-W,8-4,7.8
4266,50,W-W-R,4-8,7.8
4266,50,W-W-W,4-4,7.8
4266,50,R-R-R-R,4-4-4,7.8
4266,50,R-R-R-W,4-4-4,7.8
4266,50,R-R-W-R,4-4-8,7.8
4266,50,R-R-W-W,4-4-4,7.8
4266,50,R-W-R-R,4-8-4,7.8
4266,50,R-W-R-W,4-8-4,7.8
4266,50,R-W-W-R,4-4-8,7.8
4266,50,R-W-W-W,4-4-4,7.8
4266,50,W-R-R-R,8-4-4,7.8
4266,50,W-R-R-W,8-4-4,7.8
4266,50,W-R-W-R,8-4-8,7.8
4266,50,W-R-W-W,8-4-4,7.8
4266,50,W-W-R-R,4-8-4,7.8
4266,50,W-W-R-W,4-8-4,7.8
4266,50,W-W-W-R,4-4-8,7.8
4266,50,W-W-W-W,4-4-4,7.8
"""

# ❗ In your real file paste your FULL dataset here
data = pd.read_csv(io.StringIO(data_string.strip()))
data.columns = ['Frequency', 'Temperature', 'Command_Sequence', 'Optimal_Spacing', 'Optimal_Refresh_Rate']

freq_options = sorted(data['Frequency'].unique().tolist())

# -------------------------------------------------------
# TRANSFORM TRAINING DATA FOR ML
# -------------------------------------------------------
transformed = []
for _, row in data.iterrows():
    cmds = row['Command_Sequence'].split('-')
    spacings = str(row['Optimal_Spacing']).split('-')
    if len(cmds) - 1 != len(spacings):
        continue
    for i in range(len(cmds) - 1):
        transformed.append({
            'Frequency': row['Frequency'],
            'Temperature': row['Temperature'],
            'Prev_Cmd': cmds[i],
            'Curr_Cmd': cmds[i+1],
            'Spacing': int(spacings[i]),
            'Refresh_Rate': float(row['Optimal_Refresh_Rate'])
        })

df = pd.DataFrame(transformed)

# Train ML model for spacing
X = df[['Frequency','Temperature','Prev_Cmd','Curr_Cmd']]
y = df[['Spacing','Refresh_Rate']]

preprocessor = ColumnTransformer([
    ('num', 'passthrough', ['Frequency','Temperature']),
    ('cat', OneHotEncoder(handle_unknown='ignore'), ['Prev_Cmd','Curr_Cmd'])
])

model = Pipeline([
    ('pre', preprocessor),
    ('rf', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X, y)

# Bus capacity table
default_capacity = {
    2133: 20,
    2666: 24,
    3200: 28,
    3733: 32,
    4266: 36
}

# -------------------------------------------------------
# PAGE HEADER
# -------------------------------------------------------
st.title("Memory Bus Optimizer")
st.markdown("ML spacing + Bus packing + Your refresh rules.")

# -------------------------------------------------------
# SIDEBAR INPUTS (ONLY ONE TEMPERATURE NOW)
# -------------------------------------------------------
st.sidebar.header("Configuration")

# Frequency input
freq = st.sidebar.selectbox("Frequency (MT/s)", freq_options, index=0)

# SINGLE Temperature input (0–95°C)
temperature = st.sidebar.number_input("Temperature (0–95°C)", min_value=0, max_value=95, value=30)

# Command Sequence
seq = st.sidebar.text_input("Command Sequence", "R-W-R-W")

# REF Rule Inputs
st.sidebar.markdown("---")
st.sidebar.subheader("Refresh Rule Engine")

command_type = st.sidebar.selectbox("Select Command", ["REFab", "REFsb"])
refresh_mode = st.sidebar.selectbox("Refresh Mode", ["Normal", "Fine Granularity"])

n_banks = None
if command_type == "REFsb":
    n_banks = st.sidebar.number_input("Number of Banks (n)", min_value=1, max_value=32, value=4)

# -------------------------------------------------------
# YOUR REFRESH RULE ENGINE
# -------------------------------------------------------
def compute_refresh_value(cmd, mode, temp, n=None):
    if cmd == "REFab":
        if mode == "Normal":
            return 3.9 if temp <= 85 else 1.95
        elif mode == "Fine Granularity":
            return 1.95 if temp <= 85 else 0.975

    if cmd == "REFsb":
        if mode == "Fine Granularity":
            return (1.95 / n) if temp <= 85 else (0.975 / n)

    return None

computed_refresh_value = compute_refresh_value(command_type, refresh_mode, temperature, n_banks)

# -------------------------------------------------------
# RUN BUTTON
# -------------------------------------------------------
run = st.button("Run Optimization")

# -------------------------------------------------------
# MAIN LOGIC
# -------------------------------------------------------
if run:

    # ML spacing prediction using the SAME temperature input
    commands = [c.strip().upper() for c in seq.split('-') if c.strip()]

    pairs = []
    predicted_spacings = []
    for i in range(len(commands)-1):
        prev_cmd = commands[i]
        curr_cmd = commands[i+1]

        inp = pd.DataFrame([{
            'Frequency': freq,
            'Temperature': temperature,   # 🔥 ONE temperature used everywhere
            'Prev_Cmd': prev_cmd,
            'Curr_Cmd': curr_cmd
        }])

        pred = model.predict(inp)[0]
        predicted_spacings.append(int(round(pred[0])))
        pairs.append(f"{prev_cmd}→{curr_cmd}")

    # Build steps
    steps = []
    for i, p in enumerate(pairs):
        busy = 2 if p[0] == 'R' else 4
        steps.append({
            "Transition": p,
            "Predicted Spacing": predicted_spacings[i],
            "Busy": busy,
            "Total": busy + predicted_spacings[i]
        })

    # Pack buses
    cap = default_capacity[freq]
    buses = []
    cur_bus = []
    used = 0

    for step in steps:
        if used + step["Total"] <= cap:
            cur_bus.append(step)
            used += step["Total"]
        else:
            buses.append((cur_bus, used))
            cur_bus = [step]
            used = step["Total"]
    if cur_bus:
        buses.append((cur_bus, used))

    # -------------------------------------------------------
    # METRICS OUTPUT (ORDER FIXED)
    # -------------------------------------------------------
    st.header("Optimization Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Buses Required", len(buses))
    c2.metric("Total Cycles Consumed", sum(s["Total"] for s in steps))
    c3.metric("Computed Refresh Value", f"{computed_refresh_value:.4f}")
    c4.metric("Bus Capacity", f"{cap} cycles")

    # -------------------------------------------------------
    # TABLES
    # -------------------------------------------------------
    st.subheader("Bus Packing Breakdown")
    bus_rows = []
    for i,(b,used) in enumerate(buses):
        bus_rows.append({
            "Bus #": i+1,
            "Used Cycles": used,
            "Capacity": cap,
            "Utilization %": round((used/cap)*100,1),
            "Transitions": " | ".join([s["Transition"] for s in b])
        })
    st.dataframe(pd.DataFrame(bus_rows), hide_index=True, use_container_width=True)

    st.subheader("Per-Transition Details")
    st.dataframe(pd.DataFrame(steps), hide_index=True, use_container_width=True)

st.markdown("---")
st.caption("Made with ❤️, macha.")
