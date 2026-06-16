import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime

# --- 1. PAGE CONFIGURATION & THEMING ---
st.set_page_config(
    page_title="Executive Retail Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom injection for a high-end, unified dark glassmorphism interface (Ref: Image 1 style)
st.markdown("""
    <style>
        /* Base Background and Fonts */
        .stApp {
            background-color: #0b0e14;
            color: #ecf0f1;
        }
        
        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #111622 !important;
            border-right: 1px solid #1f293d;
        }
        
        /* Glassmorphic KPI Cards */
        .kpi-card {
            background: linear-gradient(135deg, #161c2a 0%, #111622 100%);
            border: 1px solid #233044;
            padding: 22px;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
            margin-bottom: 15px;
        }
        .kpi-title {
            color: #8a99ad;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .kpi-value {
            color: #ffffff;
            font-size: 1.8rem;
            font-weight: 700;
            font-family: 'Courier New', monospace;
        }
        .kpi-delta {
            font-size: 0.85rem;
            margin-top: 4px;
            font-weight: bold;
        }
        .delta-pos { color: #00e676; }
        .delta-neg { color: #ff5252; }

        /* Input Panel Wrapper */
        .input-panel {
            background: #121824;
            border: 1px solid #1f293d;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 25px;
        }

        /* Buttons Styling */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #ff4757 0%, #ff6b81 100%) !important;
            color: white !important;
            border: none !important;
            font-weight: 600 !important;
            padding: 0.6rem 2rem !important;
            border-radius: 6px !important;
            width: 100%;
            box-shadow: 0 4px 15px rgba(255, 71, 87, 0.4);
            transition: all 0.3s ease;
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 71, 87, 0.6);
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. MOCK MODEL INFERENCE ENGINE ---
# Replace this helper function with your actual backend ML model evaluation code
def run_model_inference(store_id, is_promo, assortment, holiday_type, is_weekend, expected_customers):
    # Dummy underlying logic matching operational assumptions
    base_sales = expected_customers * 14.20
    
    # Feature weights multipliers
    promo_multiplier = 1.45 if is_promo else 1.0
    weekend_multiplier = 0.85 if is_weekend else 1.15  # Retail dependent structural drop/hike
    assortment_map = {"Basic": 1.0, "Extended": 1.18, "Extra": 1.35}
    holiday_map = {"None": 1.0, "Public Holiday": 0.60, "School Holiday": 1.10, "Easter": 0.50}
    
    predicted_sales = (base_sales * promo_multiplier * weekend_multiplier * assortment_map[assortment] * holiday_map[holiday_type])
    
    # Adding a light Gaussian noise structure to keep outputs dynamic
    predicted_sales = max(0.0, predicted_sales + np.random.normal(0, 50))
    return round(predicted_sales, 2), int(expected_customers)


# --- 3. SIDEBAR (SYSTEM CONFIGURATIONS) ---
with st.sidebar:
    st.markdown("<h2 style='color:#00ffcc; margin-top:0;'>⚙️ System Settings</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("### 🤖 Model Architecture")
    model_version = st.selectbox(
        "Select Active Weights Profile", 
        ["v2.1 - XGBoost (Production Production)", "v1.0 - Random Forest Baseline"],
        label_visibility="collapsed"
    )
    
    st.markdown("### 🛠️ Strategic Overrides")
    override_assortment = st.selectbox("Global Assortment Level", ["Basic", "Extended", "Extra"], index=1)
    
    st.markdown("---")
    st.caption("Retail Serving Interface v3.0 • Status: Operational")


# --- 4. MAIN INTERFACE HEADER ---
col_header, col_status = st.columns([5, 1])
with col_header:
    st.markdown("<h1 style='margin-bottom: 0px; color:#ffffff;'>Store Performance Forecasting</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#7f8c8d; font-size:1.1rem; margin-top:4px;'>Serving multi-variant forward-looking retail projections</p>", unsafe_allow_html=True)
with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='text-align: right; background: rgba(0, 230, 118, 0.1); border: 1px solid #00e676; padding: 6px 12px; border-radius: 20px; color: #00e676; font-size: 0.85rem; font-weight: bold; display: inline-block;'>● ENGINE ONLINE</div>", unsafe_allow_html=True)

st.markdown("---")


# --- 5. CONTROL PANEL & PARAMETER INPUTS ---
st.markdown("<h3 style='color: #00f0ff; font-size:1.2rem; margin-bottom:15px;'>🎛️ Scenario Control Parameters</h3>", unsafe_allow_html=True)

# Encapsulated Input layout block
with st.container():
    col1, col2, col3 = st.columns(3)
    
    with col1:
        inp_store_id = st.number_input("Target Store ID", min_value=1, max_value=1115, value=1, step=1)
        inp_customers = st.number_input("Expected Customer Baseline Volume", min_value=10, max_value=5000, value=620)
        
    with col2:
        inp_holiday = st.selectbox("Calendar Holiday Multiplier", ["None", "Public Holiday", "School Holiday", "Easter"])
        inp_weekend = st.checkbox("Apply Weekend Correction Factor", value=False)
        
    with col3:
        inp_promo = st.checkbox("Active Promotional Run (IsPromo)", value=True)
        st.markdown("<div style='margin-top:25px;'></div>", unsafe_allow_html=True)
        execute_inference = st.button("Generate Server Inference")


# --- 6. BATCH PROCESSING LAYER (CSV FILE UPLOADS) ---
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📂 Batch Optimization Engine (Upload CSV Datasets)"):
    st.markdown("""
    Provide a structured tabular system matrix containing historical parameters. 
    **Required Columns:** `Date`, `IsHoliday`, `IsWeekend`, `IsPromo`
    """)
    uploaded_file = st.file_uploader("Drop target matrix operational logs directly here", type=["csv"], label_visibility="collapsed")


# --- 7. INFERENCE EXECUTION PIPELINE & VISUALS ---
if execute_inference or uploaded_file is not None:
    
    # ------------------ CASE A: BATCH PROCESSING EVALUATION ------------------
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            # Formulating fallback structure validation routines
            required_cols = ['Date', 'IsHoliday', 'IsWeekend', 'IsPromo']
            if not all(col in df_batch.columns for col in required_cols):
                st.error(f"Execution terminated. Uploaded CSV must accurately match structural format headers: {required_cols}")
            else:
                # Execution Matrix conversion loop parsing
                batch_sales = []
                batch_cust = []
                
                for idx, row in df_batch.iterrows():
                    # Parse dependencies gracefully
                    h_type = "School Holiday" if str(row['IsHoliday']) == '1' else "None"
                    w_flag = True if str(row['IsWeekend']) == '1' else False
                    p_flag = True if str(row['IsPromo']) == '1' else False
                    
                    # Fallback dynamic check for missing store ID
                    s_id = row['Store_id'] if 'Store_id' in df_batch.columns else inp_store_id
                    c_vol = row['Customers'] if 'Customers' in df_batch.columns else inp_customers
                    
                    s, c = run_model_inference(s_id, p_flag, override_assortment, h_type, w_flag, c_vol)
                    batch_sales.append(s)
                    batch_cust.append(c)
                
                df_batch['Predicted_Sales'] = batch_sales
                df_batch['Predicted_Customers'] = batch_cust
                
                # --- METRICS CALCULATIONS ---
                total_sales = df_batch['Predicted_Sales'].sum()
                avg_cust = df_batch['Predicted_Customers'].mean()
                efficiency = total_sales / df_batch['Predicted_Customers'].sum() if df_batch['Predicted_Customers'].sum() > 0 else 0
                
                # Display Card Container Components 
                m_col1, m_col2, m_col3 = st.columns(3)
                with m_col1:
                    st.markdown(f"""<div class='kpi-card'><div class='kpi-title'>Cumulated Batch Sales</div><div class='kpi-value'>${total_sales:,.2f}</div><div class='kpi-delta delta-pos'>↑ Batch Complete</div></div>""", unsafe_allow_html=True)
                with m_col2:
                    st.markdown(f"""<div class='kpi-card'><div class='kpi-title'>Mean Customer Flow</div><div class='kpi-value'>{avg_cust:.1f}</div><div class='kpi-delta delta-neg'>↓ Active Period</div></div>""", unsafe_allow_html=True)
                with m_col3:
                    st.markdown(f"""<div class='kpi-card'><div class='kpi-title'>Aggregated Capture Margin</div><div class='kpi-value'>${efficiency:.2f}</div><div class='kpi-delta delta-pos'>★ Nominal</div></div>""", unsafe_allow_html=True)
                
                # --- VISUALIZATION LAYER (PLOTLY HIGH-END GRAPHING) ---
                st.markdown("<h3 style='color: #ffffff; font-size:1.1rem; margin-top:20px;'>📉 Multi-Axis Operational Trend Graph</h3>", unsafe_allow_html=True)
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(x=df_batch['Date'], y=df_batch['Predicted_Sales'], name='Predicted Revenue ($)', line=dict(color='#00f0ff', width=3)), secondary_y=False)
                fig.add_trace(go.Bar(x=df_batch['Date'], y=df_batch['Predicted_Customers'], name='Customer Counts', marker_color='rgba(155, 89, 182, 0.3)', opacity=0.6), secondary_y=True)
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=20, r=20, t=40, b=20),
                    xaxis=dict(showgrid=True, gridcolor='#1f293d', color='#8a99ad'),
                    yaxis=dict(showgrid=True, gridcolor='#1f293d', color='#8a99ad', title="Revenue ($)"),
                    yaxis2=dict(color='#8a99ad', title="Customers Count", overlaying="y", side="right")
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # --- DATA STORAGE INTERACTION LAYER ---
                csv_bytes = df_batch.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Export Prediction Data Table (CSV)",
                    data=csv_bytes,
                    file_name=f"batch_predictions_{datetime.date.today()}.csv",
                    mime="text/csv"
                )
                
                st.markdown("### Preview Pipeline Sample Window")
                st.dataframe(df_batch.head(10), use_container_width=True)
                
        except Exception as e:
            st.error(f"Runtime Compilation Error: {str(e)}")

    # ------------------ CASE B: SINGLE SCENARIO EVALUATION ------------------
    else:
        sales_val, customer_val = run_model_inference(
            inp_store_id, inp_promo, override_assortment, inp_holiday, inp_weekend, inp_customers
        )
        
        # Display Card UI Components
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""<div class='kpi-card'><div class='kpi-title'>Target Predicted Revenue</div><div class='kpi-value'>${sales_val:,.2f}</div><div class='kpi-delta delta-pos'>↑ +8.4% (vs Baseline)</div></div>""", unsafe_allow_html=True)
        with m_col2:
            st.markdown(f"""<div class='kpi-card'><div class='kpi-title'>Customer Saturation Volume</div><div class='kpi-value'>{customer_val:,}</div><div class='kpi-delta delta-neg'>↓ -2.0% (Seasonal Elasticity)</div></div>""", unsafe_allow_html=True)
        with m_col3:
            st.markdown(f"""<div class='kpi-card'><div class='kpi-title'>Derived Unit Per Ticket Value</div><div class='kpi-value'>${(sales_val/customer_val):.2f}</div><div class='kpi-delta delta-pos'>★ Optimized Core Configuration</div></div>""", unsafe_allow_html=True)
        
        # --- PREDICTION PROFILE GENERATOR OVER TIME ---
        st.markdown("<h3 style='color: #ffffff; font-size:1.1rem; margin-top:20px;'>🔮 Estimated 7-Day Forward Operational Window</h3>", unsafe_allow_html=True)
        
        dates = [datetime.date.today() + datetime.timedelta(days=x) for x in range(7)]
        sim_sales = [max(0.0, sales_val * (1 + np.random.uniform(-0.12, 0.12))) for _ in range(7)]
        sim_cust = [max(10, int(customer_val * (1 + np.random.uniform(-0.06, 0.06)))) for _ in range(7)]
        
        df_single_forecast = pd.DataFrame({"Date": dates, "Predicted_Sales": sim_sales, "Predicted_Customers": sim_cust})
        
        # Plotly Express Combined Target View Chart
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=df_single_forecast['Date'], y=df_single_forecast['Predicted_Sales'], name='Revenue Waveform', line=dict(color='#ff4757', width=4)), secondary_y=False)
        fig.add_trace(go.Bar(x=df_single_forecast['Date'], y=df_single_forecast['Predicted_Customers'], name='Foot traffic Spectrum', marker_color='rgba(0, 240, 255, 0.2)'), secondary_y=True)
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=40, b=20),
            xaxis=dict(showgrid=True, gridcolor='#1f293d', color='#8a99ad'),
            yaxis=dict(showgrid=True, gridcolor='#1f293d', color='#8a99ad'),
            yaxis2=dict(overlaying="y", side="right", color='#8a99ad')
        )
        st.plotly_chart(fig, use_container_width=True)


# --- 8. FOOTER TERMINAL CONTAINER ---
st.markdown("<br><br><div style='border-top: 1px solid #1f293d; padding-top: 15px; text-align: center; color: #57606f; font-size: 0.8rem;'>Enterprise Intelligence Server Architecture Core Framework Edition v3.0 • Secure Access Nodes</div>", unsafe_allow_html=True)