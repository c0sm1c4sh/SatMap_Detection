import streamlit as st
import requests
from PIL import Image
import time

st.set_page_config(page_title="EcoWatch AI", layout="centered")
st.title("🛰️ EcoWatch: Remote Sensing Monitor")

# CONFIGURATION: Update this to your confirmed Hugging Face URL
BACKEND_URL = "https://lerontroy-satmap-detection-backend.hf.space/predict"

st.markdown("""
**Story:** Our AI monitors satellite imagery to distinguish between natural forests 
and industrial encroachment, aiding in global reforestation verification.
""")

uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png"])

if uploaded_file:
    # use_container_width is the modern parameter for streamlit
    st.image(uploaded_file, use_container_width=True)
    
    if st.button("Analyze Land Use"):
        # Wrapping the API call in a spinner
        with st.spinner("Analyzing on Hugging Face... (May take 30s to wake up)"):
            files = {"file": uploaded_file.getvalue()}
            try:
                # 1. API Request
                response = requests.post(BACKEND_URL, files=files, timeout=45)
                
                # Check if backend is still building or sleeping
                if response.status_code == 503:
                    st.warning("🔄 Backend is still waking up. Please wait 15 seconds and try again.")
                else:
                    res = response.json()
                    
                    # 2. Main Results
                    st.divider()
                    st.subheader(f"Prediction: {res['prediction']}")
                    st.progress(res['confidence'])
                    st.write(f"**Confidence Score:** {res['confidence']*100:.2f}%")

                    # 3. Dynamic Advice
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
