# app_modern_final_CLEANED_TABLES.py - Fixed syntax, white sidebar, and improved table styling

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
PRIMARY_COLOR = "#0072b5" # Used for highlight and button

# ---------------------------
# 0) MODERN UI STYLES (Injected CSS for a cleaner look with Sidebar Fix and Table Improvements)
# ---------------------------
st.set_page_config(page_title="Memory Bus Optimizer 🚀", layout="wide")

# Custom CSS for a modern, compact look and feel AND CONTRAST FIX
st.markdown(f"""
<style>
/* 1. Global Contrast Fix: Set main text color to dark */
.stApp {{
    background-color: {MAIN_BG}; /* Light gray background */
    color: {MAIN_TEXT}; /* Dark text for better contrast */
}}
/* Ensure all main headers are dark */
h1, h2, h3, h4, .stMarkdown, label, p, .stAlert p {{
    color: {MAIN_TEXT} !important;
}}

/* 2. Metric Card Styling */
.stMetric {{
    background-color: #ffffff;
    padding: 15px;
    border-radius: 10px;
    border-left: 5px solid {PRIMARY_COLOR}; /* Highlight color */
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
}}
/* Explicitly set metric label and value colors */
[data-testid="stMetricLabel"] > div {{
    color: #555555; /* Medium gray for labels */
    font-weight: 500;
}}
[data-testid="stMetricValue"] {{
    color: {PRIMARY_COLOR}; /* Highlight color for values */
    font-size: 1.8rem;
}}
[data-testid="stMetricDelta"] {{
    color: #28a745 !important; /* Green for delta/sub-text */
}}

/* 3. General component styling */
.block-container {{
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 1rem;
    padding-right: 1rem;
}}

/* 4. SIDEBAR FIX: Dark Background and White Text */
.css-1lcbmhc, .e1fqkh3o10 {{ /* Target the sidebar container and content area */
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
    background-color: {SIDEBAR_BG}; /* Dark Sidebar Background */
}}
/* Force all text elements in the sidebar to white */
.stSidebar h2, .stSidebar h3, .stSidebar .stMarkdown, .stSidebar label, .stSidebar p {{
    color: {SIDEBAR_TEXT} !important;
}}
/* Fix for the text input label that ignores generic markdown/p style */
.stSidebar .stTextInput label p,
.stSidebar .stSelectbox label p,
.stSidebar .stNumberInput label p {{
    color: {SIDEBAR_TEXT} !important;
}}
/* Force expander text inside sidebar to white */
.stSidebar .streamlit-expanderHeader {{
    color: {SIDEBAR_TEXT} !important;
}}

/* 5. TABLE STYLING IMPROVEMENTS */
.stDataFrame > div {{ /* Target the inner div of stDataFrame */
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    border: 1px solid #e6e6e6; /* Light border */
}}
.stDataFrame table {{
    background-color: #ffffff; /* White background for tables */
    color: {MAIN_TEXT}; /* Dark text */
}}
.stDataFrame thead th {{ /* Table header */
    background-color: #f8f8f8; /* Slightly darker header */
    color: {MAIN_TEXT};
    font-weight: bold;
    border-bottom: 2px solid #e6e6e6;
}}
.stDataFrame tbody tr:nth-child(even) {{ /* Zebra striping */
    background-color: #fbfbfb;
}}
/* Ensure wide tables scroll horizontally */
.stDataFrame {{
    overflow-x: auto;
}}

</style>
""", unsafe_allow_html=True)


# ---------------------------
# 1) DATA (paste you provided dataset here)
# ---------------------------
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
data = pd.read_csv(io.StringIO(data_string.strip()))
data.columns = ['Frequency', 'Temperature', 'Command_Sequence', 'Optimal_Spacing', 'Optimal_Refresh_Rate']

freq_options = sorted(data['Frequency'].unique().tolist())
temp_options = sorted(data['Temperature'].unique().tolist())

transformed = []
for _, row in data.iterrows():
    cmds = str(row['Command_Sequence']).strip().split('-')
    spacings = str(row['Optimal_Spacing']).strip().split('-')
    if len(cmds) - 1 != len(spacings):
        continue
    for i in range(len(cmds) - 1):
        transformed.append({
            'Frequency': int(row['Frequency']),
            'Temperature': int(row['Temperature']),
            'Prev_Cmd': cmds[i].upper(),
            'Curr_Cmd': cmds[i+1].upper(),
            'Spacing': int(spacings[i]),
            'Refresh_Rate': float(row['Optimal_Refresh_Rate'])
        })

df = pd.DataFrame(transformed)

if len(df) == 0:
    st.error("No training data available after parsing. Check data_string formatting.")
    st.stop()

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

default_capacity = {
    2133: 20,
    2666: 24,
    3200: 28,
    3733: 32,
    4266: 36
}


# ---------------------------
# 6) UI: INPUTS 
# ---------------------------

with st.container():
    st.title("🚀 Memory Bus Optimizer")
    st.markdown("An ML-enhanced tool for predicting optimal command spacing and simulating bus packing efficiency.")

st.markdown("---")

st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("**1. Operating Conditions (from dataset)**")

col1_s, col2_s = st.sidebar.columns(2)
freq = col1_s.selectbox("Frequency (MT/s)", freq_options, index=freq_options.index(freq_options[2])) 
temp = col2_s.selectbox("Temperature (°C)", temp_options, index=0)

st.sidebar.markdown("**2. Command Sequence**")
seq = st.sidebar.text_input("Sequence (e.g., R-W-R-W)", value="R-W-R-W", help="R = Read, W = Write. Use dashes to separate commands.")

with st.sidebar.expander("🛠️ Advanced: Edit LPDDR Capacity"):
    capacities = default_capacity.copy()
    st.write("Set LPDDR cycle capacity per bus for each frequency:")
    for f in sorted(freq_options):
        capacities[f] = st.number_input(f"Capacity @ **{f}** MT/s (cycles)", min_value=1, value=capacities.get(f, 20), key=f)

# ---------------------------
# NEW: Refresh command/mode/temperature inputs (user requested)
# ---------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("**3. Refresh Calculation Inputs**")

# Command dropdown - only REFab and REFsb allowed
refresh_command = st.sidebar.selectbox("Refresh Command", options=["REFab", "REFsb"], index=0, help="Choose REFab or REFsb")

# Refresh mode dropdown
refresh_mode = st.sidebar.selectbox("Refresh Mode", options=["Normal", "Fine Granularity"], index=0, help="Normal or Fine Granularity")

# Temperature input constrained between 0 and 95 inclusive
refresh_temp = st.sidebar.number_input("Temperature (°C)", min_value=0.0, max_value=95.0, value=30.0, step=0.1, format="%.1f")

# If REFsb selected, ask for number of banks in bank group (n)
refsb_n = None
if refresh_command == "REFsb":
    refsb_n = st.sidebar.number_input("Number of banks in bank-group (n)", min_value=1, value=4, step=1, help="Used only when command is REFsb and mode is Fine Granularity (division).")

st.sidebar.markdown("---")

# ---------------------------
# 7) UI: INFO & RUN BUTTON 
# ---------------------------

run_col, info_col, _ = st.columns([1, 1, 3])
if run_col.button("⚡ Run Optimization", use_container_width=True, type="primary"):
    st.session_state['run_clicked'] = True
else:
    if 'run_clicked' not in st.session_state:
        st.session_state['run_clicked'] = False

with info_col.expander("📚 About This Tool"):
    st.markdown("""
    This application utilizes a **Random Forest Regressor** trained on the provided dataset to predict the **Optimal Spacing** (idle cycles) and **Optimal Refresh Rate** for any given command transition ($Prev \\to Curr$) under specific frequency and temperature conditions.
    
    The predicted transitions are then scheduled (packed) into virtual **LPDDR Buses** based on a simple **First-Fit sequential algorithm**.
    """)

st.markdown("---")

# ---------------------------
# helper functions 
# ---------------------------
BUSY = {'R': 2, 'W': 4}

def compute_refresh_value(command: str, mode: str, temp_c: float, n_banks: int = None) -> float:
    """
    Implements the exact mapping you requested:
    1) REFab & Normal & 0 <= temp <= 85  => 3.9
    2) REFab & Normal & 85 < = temp <=95 => 1.95
    3) REFab & FineGran & 0 <= temp <=85 => 1.95
    4) REFab & FineGran & 85 < = temp <=95 => 0.975
    6) REFsb & FineGran & 0 <= temp <=85 => 1.95 / n
    7) REFsb & FineGran & 85 < = temp <=95 => 0.975 / n

    For REFsb & Normal: we apply the same mapping as REFab & Normal (not divided), because you didn't specify otherwise.
    Temperature boundaries are inclusive.
    """
    # validate temp range (should already be enforced by UI)
    if temp_c < 0 or temp_c > 95:
        raise ValueError("Temperature must be between 0 and 95 inclusive.")

    # Helper: check if temp in lower (0..85) or upper (85..95)
    # Inclusive rules: 0 <= temp <=85 lower, 85 < temp <=95 upper? 
    # You specified inclusive on both 0 and 85 and also 95 included for upper.
    # We'll treat 0..85 (inclusive) as lower, and >85..95 (inclusive) as upper.
    # BUT original user wrote "(0 and 85 included) ... 85 to 95(95 included)". To avoid overlap,
    # interpret ranges as: 0 <= temp <= 85 -> lower, 85 < temp <= 95 -> upper.
    # If user really wanted 85 included in both, this is ambiguous; we choose the practical, non-overlapping approach.
    if temp_c <= 85.0:
        temp_region = "lower"
    else:
        temp_region = "upper"

    command = (command or "").strip()
    mode = (mode or "").strip()

    # REFab handling
    if command == "REFab":
        if mode == "Normal":
            if temp_region == "lower":
                return 3.9
            else:
                return 1.95
        else:  # Fine Granularity
            if temp_region == "lower":
                return 1.95
            else:
                return 0.975

    # REFsb handling
    if command == "REFsb":
        # If Fine Granularity -> require n_banks (division)
        if mode == "Fine Granularity":
            if not n_banks or n_banks <= 0:
                # Avoid ZeroDivisionError; return NaN or raise. We'll raise to flag user input error.
                raise ValueError("For REFsb with Fine Granularity you must provide a valid n (>=1).")
            if temp_region == "lower":
                return 1.95 / float(n_banks)
            else:
                return 0.975 / float(n_banks)
        else:  # Normal mode for REFsb - apply same values as REFab Normal (no division)
            if temp_region == "lower":
                return 3.9
            else:
                return 1.95

    # fallback (should not happen)
    raise ValueError("Unsupported command/mode combination.")


@st.cache_data
def predict_for_sequence(freq_sel, temp_sel, seq_string):
    commands = [c.strip().upper() for c in seq_string.split('-') if c.strip()]
    if len(commands) < 2:
        raise ValueError("Sequence must contain at least two commands (e.g., R-W).")
    pairs = []
    predicted_spacings = []
    predicted_refreshes = []
    for i in range(len(commands)-1):
        prev_cmd = commands[i]
        curr_cmd = commands[i+1]
        inp = pd.DataFrame([{
            'Frequency': int(freq_sel),
            'Temperature': int(temp_sel),
            'Prev_Cmd': prev_cmd,
            'Curr_Cmd': curr_cmd
        }])
        pred = model.predict(inp)[0]
        spacing = int(round(pred[0]))
        refresh = float(pred[1])
        pairs.append((prev_cmd, curr_cmd))
        predicted_spacings.append(spacing)
        predicted_refreshes.append(refresh)
    avg_refresh = float(np.mean(predicted_refreshes)) if len(predicted_refreshes) else 0.0
    return commands, pairs, predicted_spacings, avg_refresh

@st.cache_data
def build_steps(pairs, spacings):
    steps = []
    for i, (prev, curr) in enumerate(pairs):
        busy = BUSY.get(prev, 2)
        spacing = spacings[i]
        step_cycles = spacing + busy
        steps.append({
            'index': i,
            'pair': f"{prev}→{curr}",
            'prev': prev,
            'curr': curr,
            'predicted_spacing': spacing,
            'busy_cycles': busy,
            'step_cycles': step_cycles
        })
    return steps

@st.cache_data
def pack_into_buses(steps, capacity):
    """
    Sequential packing: keep order. Start new bus when adding next step would exceed capacity.
    Returns list of buses where each bus is a list of steps.
    """
    buses = []
    cur_bus = []
    cur_sum = 0
    for s in steps:
        if cur_sum + s['step_cycles'] <= capacity:
            cur_bus.append(s)
            cur_sum += s['step_cycles']
        else:
            if cur_bus:
                buses.append({'steps': cur_bus, 'used_cycles': cur_sum})
            cur_bus = [s]
            cur_sum = s['step_cycles']
    if cur_bus:
        buses.append({'steps': cur_bus, 'used_cycles': cur_sum})
    return buses

def cycle_time_ns(freq_mts):
    return 1000.0 / float(freq_mts)

def summarize_buses(buses, capacity, cycle_ns, avg_refresh_us):
    summary = []
    for i, b in enumerate(buses):
        used = b['used_cycles']
        util = (used / capacity) * 100.0 if capacity > 0 else 0.0
        elapsed_ns = used * cycle_ns
        refresh_ns = avg_refresh_us * 1000.0 if avg_refresh_us > 0 else float('inf')
        refresh_events = int(elapsed_ns // refresh_ns) if refresh_ns != float('inf') else 0
        summary.append({
            'Bus': i+1,
            'Used Cycles': used,
            'Capacity': capacity,
            'Utilization (%)': round(util, 1),
            'Refresh Events (this bus)': refresh_events,
            'Transitions on Bus (in order)': " | ".join([step['pair'] for step in b['steps']])
        })
    return summary

# ---------------------------
# run prediction + packing
# ---------------------------
if 'run_clicked' not in st.session_state:
    st.session_state['run_clicked'] = False

# Show a small preview of computed refresh value before run (so user sees mapping)
try:
    preview_value = compute_refresh_value(refresh_command, refresh_mode, refresh_temp, refsb_n)
    st.sidebar.markdown(f"**Preview refresh value:** `{preview_value:.6g}`")
except Exception as e:
    st.sidebar.markdown(f"**Preview refresh value:** ❗ {e}")

if st.session_state['run_clicked']:
    try:
        # Predict using ML
        commands, pairs, predicted_spacings, avg_refresh = predict_for_sequence(freq, temp, seq)
        steps = build_steps(pairs, predicted_spacings)

        # capacity for selected frequency
        cap = capacities.get(int(freq), default_capacity.get(int(freq), 20))
        buses = pack_into_buses(steps, cap)
        cycle_ns = cycle_time_ns(freq)
        bus_summ = summarize_buses(buses, cap, cycle_ns, avg_refresh)

        # overall totals
        total_steps_cycles = sum([s['step_cycles'] for s in steps])
        total_busy = sum([s['busy_cycles'] for s in steps])
        total_buses = len(buses)

        ## Quick Summary (Metrics)
        st.header("🎯 Optimization Results")

        col_met1, col_met2, col_met3, col_met4 = st.columns(4)

        col_met1.metric(
            label="Buses Required",
            value=total_buses,
            delta=f"{len(steps)} Transitions"
        )
        col_met2.metric(
            label="Total Cycles Consumed",
            value=f"{total_steps_cycles} cycles",
            help="Sum of (Predicted Spacing + Busy Cycles) for all transitions."
        )
        col_met3.metric(
            label="Refresh Interval",
            value=f"{avg_refresh:.2f} μs",
            delta="Predicted by ML Model"
        )
        col_met4.metric(
            label="Bus Capacity",
            value=f"{cap} cycles",
            delta=f"@{freq} MT/s"
        )

        # Show the computed refresh value (from new rules)
        try:
            computed_value = compute_refresh_value(refresh_command, refresh_mode, refresh_temp, refsb_n)
            st.metric(label="Computed Refresh Value (per your rules)", value=f"{computed_value:.6g}")
        except Exception as e:
            st.error(f"Refresh value error: {e}")

        st.markdown("---")

        ## Bus-by-bus table (Now permanently visible)
        st.subheader("🚌 Bus Packing Breakdown")
        st.markdown("The sequence is packed sequentially into buses, each having a capacity of **{} cycles**.".format(cap))
        st.dataframe(pd.DataFrame(bus_summ), use_container_width=True, hide_index=True)

        st.markdown("---")

        ## Detailed Per-Transition Table (Now permanently visible)
        st.subheader("📋 Detailed Per-Transition Predictions")
        st.markdown("Predicted timing parameters for each transition in the sequence.")

        trans_df = pd.DataFrame(steps)[['index','pair','predicted_spacing','busy_cycles','step_cycles']]
        trans_df = trans_df.rename(columns={
            'index':'#',
            'pair':'Transition',
            'predicted_spacing':'Predicted Spacing (cycles)',
            'busy_cycles':'Busy Cycles',
            'step_cycles':'Step Cycles (Spacing + Busy)'
        })
        st.dataframe(trans_df, use_container_width=True, hide_index=True)

        st.markdown("---")


        ## Explanation (Retained for clarity)
        st.subheader("💡 Simple Explanation of Logic")
        st.markdown(f"""
        1.  **Prediction:** Your sequence had **{len(steps)} transitions**. The ML model determined the optimal **spacing** (idle cycles) for each.
        2.  **Cycle Consumption:** Each transition uses `Step Cycles` = `Predicted Spacing` + `Busy Cycles` (R=2, W=4). The total required cycles were **{total_steps_cycles} cycles**.
        3.  **Packing Result:** Given the **{cap} cycle** limit per bus, the sequential scheduling required **{total_buses} buses** to complete the command sequence (`{seq}`).
        """)

    except ValueError as e:
        st.error(f"Input Error: {e}")
        st.session_state['run_clicked'] = False
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.session_state['run_clicked'] = False

st.markdown("---")
st.caption("Developed using Streamlit, Scikit-learn, and your custom dataset. Data and ML model trained in-memory.")
