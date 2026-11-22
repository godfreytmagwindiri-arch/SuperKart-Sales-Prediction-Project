import streamlit as st
import requests
import pandas as pd
import json

st.set_page_config(page_title="SuperKart Sales Predictor", layout="wide")
st.title("🛒 SuperKart Store Sales Prediction Portal")

backend_url = "http://backend-api:7860"

tab1, tab2 = st.tabs(["Online Single Prediction", "Batch Forecast Prediction"])

with tab1:
    st.header("Single Product Sales Prediction")
    col1, col2 = st.columns(2)
    with col1:
        product_weight = st.number_input("Product Weight", min_value=1.0, max_value=50.0, value=12.66)
        sugar_content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        allocated_area = st.number_input("Allocated Area Ratio", min_value=0.0, max_value=1.0, value=0.027, format="%.3f")
        mrp = st.number_input("Maximum Retail Price (MRP)", min_value=1.0, max_value=500.0, value=117.08)
        store_age = st.number_input("Store Age (Years)", min_value=1, max_value=100, value=16)
    with col2:
        store_size = st.selectbox("Store Size", ["Medium", "High", "Small"])
        city_type = st.selectbox("City Location Type", ["Tier 2", "Tier 1", "Tier 3"])
        store_type = st.selectbox("Store Type", ["Supermarket Type2", "Departmental Store", "Supermarket Type1", "Food Mart"])
        product_id_char = st.selectbox("Product ID Category Prefix", ["FD", "NC", "DR"])
        product_type_cat = st.selectbox("Product Type Category", ["Non Perishables", "Perishables"])

    if st.button("Predict Store Sales Total"):
        payload = {
            "Product_Weight": product_weight,
            "Product_Sugar_Content": sugar_content,
            "Product_Allocated_Area": allocated_area,
            "Product_MRP": mrp,
            "Store_Size": store_size,
            "Store_Location_City_Type": city_type,
            "Store_Type": store_type,
            "Product_Id_char": product_id_char,
            "Store_Age_Years": store_age,
            "Product_Type_Category": product_type_cat
        }
        try:
            r = requests.post(f"{backend_url}/v1/predict", json=payload)
            if r.status_code == 200:
                pred = r.json()["prediction"]
                st.success(f"Predicted Total Revenue: ${pred:,.2f}")
            else:
                st.error(f"Error from backend: {r.text}")
        except Exception as e:
            st.error(f"Could not reach prediction backend: {e}")

with tab2:
    st.header("Upload Batch CSV file for Predictions")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        try:
            files = {'file': uploaded_file.getvalue()}
            r = requests.post(f"{backend_url}/v1/predictbatch", files=files)
            if r.status_code == 200:
                predictions = r.json()
                st.success("Batch Processing Completed!")
                res_df = pd.read_csv(uploaded_file)
                res_df['Predicted_Sales_Total'] = [predictions[str(i)] for i in range(len(res_df))]
                st.dataframe(res_df)
            else:
                st.error(f"Error processing batch: {r.text}")
        except Exception as e:
            st.error(f"Could not complete batch forecast: {e}")