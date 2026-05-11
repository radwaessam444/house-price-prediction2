import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# --- PAGE CONFIG ---
st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# --- DATA LOADING ---
@st.cache_data
def load_data():
   df = pd.read_csv("House_price.csv")
   return df

df = load_data()
# --- HEADER ---
st.title("🏠 Real Estate Price Analytics")
st.markdown("Enter the property details below to estimate the market value.")
st.divider()

# --- MAIN LAYOUT ---
col1, col2 = st.columns([1, 2], gap="large")

with col1:
    st.subheader("📍 Property Details")
    
    income = st.number_input("Avg. Area Income ($)", value=float(df["Avg. Area Income"].mean()), step=500.0)
    age = st.slider("House Age (Years)", 0, 15, 5)
    
    # Use columns for smaller inputs
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        rooms = st.number_input("Total Rooms", value=int(df["Number of Rooms"].mean()))
    with sub_col2:
        bedrooms = st.number_input("Bedrooms", value=int(df["Number of Bedrooms"].mean()))
        
    population = st.number_input("Area Population", value=float(df["Area Population"].mean()))
    
    predict_btn = st.button("Calculate Estimated Value")

with col2:
    st.subheader("📊 Market Results")
    
    # Simple model training
    X = df[['Avg. Area Income', 'House Age', 'Number of Rooms', 'Number of Bedrooms', 'Area Population']]
    y = df['Price']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    if predict_btn:
        prediction = model.predict([[income, age, rooms, bedrooms, population]])[0]
        
        # Professional Metric Display
        m1, m2 = st.columns(2)
        m1.metric(label="Estimated Price", value=f"${prediction:,.2f}")
        m2.metric(label="Price per Room", value=f"${(prediction/rooms):,.2f}")
        
        st.info(f"Analysis based on a dataset of {len(df)} historical sales in this region.")
        
        # Show a small chart of similar houses
        st.write("### Area Price Distribution")
        st.area_chart(df['Price'].sample(50))
    else:
        st.write("👈 Adjust parameters and click 'Calculate' to see the prediction.")
        st.image("https://images.unsplash.com/photo-1560518883-ce09059eeffa?ixlib=rb-1.2.1&auto=format&fit=crop&w=1000&q=80", use_container_width=True)

# --- FOOTER ---
st.divider()
st.caption("Powered by Scikit-Learn and Streamlit | House Price Prediction Project 2026")

 