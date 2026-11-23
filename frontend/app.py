import os
from io import BytesIO
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="SuperKart Sales Predictor", layout="wide")
st.title("🛒 SuperKart Sales Prediction")
API = os.getenv("BACKEND_URL", "http://backend-api:7860").rstrip("/")

@st.cache_data(ttl=300)
def get_schema(url):
    r = requests.get(f"{url}/v1/schema", timeout=15)
    r.raise_for_status()
    return r.json()

try:
    schema = get_schema(API)
except (requests.RequestException, ValueError) as e:
    st.error(f"Backend connection failed: {e}")
    st.stop()

st.caption("Sales predictions are estimates based on the trained model.")
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.subheader("Product and Store Information")
    with st.form("prediction_form"):
        values = {}
        col1, col2 = st.columns(2)

        for i, name in enumerate(schema["features"]):
            with (col1 if i % 2 == 0 else col2):
                if name in schema["numeric_features"]:
                    default = float(schema.get("numeric_defaults", {}).get(name, 0))
                    values[name] = st.number_input(name, value=default)
                else:
                    options = schema.get("categorical_values", {}).get(name, [])
                    values[name] = (
                        st.selectbox(name, [str(x) for x in options])
                        if options else st.text_input(name)
                    )

        submit = st.form_submit_button("Predict Sales")

    if submit:
        try:
            r = requests.post(f"{API}/v1/sales", json=values, timeout=30)
            r.raise_for_status()
            prediction = float(r.json()["Predicted_Sales"])
            st.metric("Predicted Store Sales Total", f"{prediction:,.2f}")
        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            st.error(f"Prediction failed: {e}")
            if "r" in locals() and not r.ok:
                st.code(r.text)

with tab2:
    st.subheader("Batch Sales Prediction")
    st.write("Required columns:", ", ".join(schema["features"]))

    uploaded = st.file_uploader("Upload CSV", type="csv")

    if uploaded is not None and st.button("Run Batch Prediction"):
        try:
            content = uploaded.getvalue()
            df = pd.read_csv(BytesIO(content))

            missing = [c for c in schema["features"] if c not in df.columns]
            if missing:
                raise ValueError(f"Missing predictors: {missing}")

            r = requests.post(
                f"{API}/v1/salesbatch",
                files={"file": (uploaded.name, content, "text/csv")},
                timeout=120)
            r.raise_for_status()

            result = r.json()
            predictions = [result[str(i)] for i in range(len(df))]
            df["Predicted_Sales_Total"] = predictions

            st.success("Batch prediction completed successfully.")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "Download Predictions",
                df.to_csv(index=False),
                "SuperKart_Batch_Predictions.csv",
                "text/csv")

        except (requests.RequestException, ValueError, KeyError, TypeError) as e:
            st.error(f"Batch prediction failed: {e}")
            if "r" in locals() and not r.ok:
                st.code(r.text)
