import streamlit as st
import requests
from PIL import Image
import os

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="EcoWatch AI", layout="centered")

st.title("🛰️ EcoWatch: Remote Sensing Monitor")

# ------------------ FEATURE SWITCH ------------------
mode = st.sidebar.selectbox(
    "Choose Feature",
    ["Land Classification", "Land Change Analysis"]
)

# ------------------ BACKEND CONFIG ------------------
BACKEND_URL = "https://lerontroy-satmap-detection-backend.hf.space/predict"

# =========================================================
# 🔹 FEATURE 1: LAND CLASSIFICATION (EXISTING)
# =========================================================
if mode == "Land Classification":

    st.markdown("""
    **Story:** Our AI monitors satellite imagery to distinguish between natural forests 
    and industrial encroachment, aiding in global reforestation verification.
    """)

    uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png"])

    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

        if st.button("Analyze Land Use"):
            with st.spinner("Analyzing on Hugging Face... (May take 30s to wake up)"):
                files = {"file": uploaded_file.getvalue()}
                try:
                    response = requests.post(BACKEND_URL, files=files, timeout=45)

                    if response.status_code == 503:
                        st.warning("🔄 Backend is still waking up. Please wait 15 seconds and try again.")
                    else:
                        res = response.json()

                        st.divider()
                        st.subheader(f"Prediction: {res['prediction']}")
                        st.progress(res['confidence'])
                        st.write(f"**Confidence Score:** {res['confidence']*100:.2f}%")

                        # Dynamic advice
                        if res['prediction'] == "Forest":
                            st.success("✅ Conservation Status: Protected. High carbon sequestration.")
                        elif res['prediction'] == "Industrial":
                            st.error("🚨 Warning: Industrial encroachment detected. Verify permits.")
                        elif res['prediction'] == "Herbaceous Vegetation":
                            st.info("🌱 Reforestation Potential: Suitable for new plantation efforts.")

                except requests.exceptions.Timeout:
                    st.error("⏱️ Connection timed out. The backend might be starting up.")
                except Exception as e:
                    st.error(f"❌ Connection Error: Ensure backend is 'Running' on Hugging Face.")


# =========================================================
# 🔹 FEATURE 2: LAND CHANGE ANALYSIS (UPDATED YEARS)
# =========================================================
elif mode == "Land Change Analysis":

    st.markdown("### 📍 Land Change Analysis Over Time")
    st.write("Explore how land usage has evolved over the years using satellite imagery.")

    # Locations
    location = st.selectbox(
        "Select Location",
        ["Bachupally", "Durgam Cheruvu", "Anurag University", "Secunderabad"]
    )

    # Fixed years
    years = [2010, 2014, 2017, 2020, 2025]

    year = st.selectbox("Select Year", years)

    # Folder mapping
    folder_map = {
        "Bachupally": "data/bachupally",
        "Durgam Cheruvu": "data/durgam",
        "Anurag University": "data/anurag",
        "Secunderabad": "data/secunderabad"
    }

    img_path = f"{folder_map[location]}/{year}.png"

    # Display image
    if os.path.exists(img_path):
        st.image(img_path, caption=f"{location} - {year}", use_container_width=True)
    else:
        st.warning("⚠️ Image not found. Please ensure images are placed correctly.")

    # ------------------ INSIGHTS ------------------
    st.divider()
    st.subheader("📊 Observations")

    if location == "Durgam Cheruvu":
        st.error("⚠️ Significant reduction in water body area observed over time.")
    elif location == "Banjara Hills":
        st.warning("🏙️ Rapid urban expansion replacing green cover.")
    elif location == "Anurag University":
        st.info("🏗️ Infrastructure development increased over time.")
    elif location == "Gachibowli":
        st.warning("🏢 IT and commercial expansion visible in recent years.")

    # ------------------ COMPARISON ------------------
    st.divider()
    st.subheader("🔍 Compare Two Years")

    year2 = st.selectbox("Compare with Year", years, index=4)

    col1, col2 = st.columns(2)

    with col1:
        img1 = f"{folder_map[location]}/{year}.png"
        if os.path.exists(img1):
            st.image(img1, caption=str(year), use_container_width=True)

    with col2:
        img2 = f"{folder_map[location]}/{year2}.png"
        if os.path.exists(img2):
            st.image(img2, caption=str(year2), use_container_width=True)
