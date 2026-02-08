import streamlit as st
import requests
from PIL import Image

st.set_page_config(page_title="EcoWatch AI")
st.title("🛰️ EcoWatch: Remote Sensing Monitor")

st.markdown("""
**Story:** Our AI monitors satellite imagery to distinguish between natural forests 
and industrial encroachment, aiding in global reforestation verification.
""")

uploaded_file = st.file_uploader("Upload Satellite Image", type=["jpg", "png"])

if uploaded_file:
    st.image(uploaded_file, use_container_width=True)
    
    # CRITICAL: Use 'backend' as the hostname because they are in the same Docker network
    if st.button("Analyze Land Use"):
        with st.spinner("Processing on RTX 3060..."):
            files = {"file": uploaded_file.getvalue()}
            try:
                # Connect to Backend
                response = requests.post("http://127.0.0.1:8000/predict", files=files)
                res = response.json()
                
                # 1. Main Result
                st.subheader(f"Prediction: {res['prediction']}")
                st.progress(res['confidence'])
                st.write(f"Confidence: {res['confidence']*100:.2f}%")

                # 2. Add the Story Context (Dynamic Advice)
                if res['prediction'] == "Forest":
                    st.success("✅ Conservation Status: Protected. High carbon sequestration.")
                elif res['prediction'] == "Industrial":
                    st.error("🚨 Warning: Industrial encroachment detected. Verify permits.")
                elif res['prediction'] == "Herbaceous Vegetation":
                    st.info("🌱 Reforestation Potential: Suitable for new plantation efforts.")

            except Exception as e:
                st.error(f"Connection Error: {e}")
