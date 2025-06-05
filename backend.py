import streamlit as st
from PIL import Image
import base64
import io
from google.cloud import aiplatform
import os

# Set page config FIRST
st.set_page_config(page_title="Chest X-Ray Classifier", layout="centered")

# Set Google Cloud credentials
import json
from google.oauth2 import service_account

if st.query_params.get("health") == ["true"]:
    st.write("ok")
    st.stop()
# Check if the environment variable is set

creds_json = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
credentials = service_account.Credentials.from_service_account_info(json.loads(creds_json))

if creds_json is None:
    st.error("Environment variable for credentials not found.")
    st.stop()

# Vertex AI project settings
PROJECT_ID = "x-ray-classification-458602"
REGION = "us-central1"
ENDPOINT_ID = "8720672022100705280"
CHEST_VALIDATOR_ENDPOINT_ID = "8864998316409094144"

# Initialize Vertex AI once before calling any endpoints
aiplatform.init(project=PROJECT_ID, location=REGION, credentials=credentials)

# Initialize both endpoints AFTER aiplatform.init
endpoint = aiplatform.Endpoint(endpoint_name=ENDPOINT_ID)
chest_validator_endpoint = aiplatform.Endpoint(endpoint_name=CHEST_VALIDATOR_ENDPOINT_ID)


# Streamlit page settings
tab1, tab2 = st.tabs(["🔍 Chest X-ray Classifier", "📘 About & Disclaimer"])

with tab1:
    st.header(" AI-Powered Chest X-ray Analysis Tool")
    st.markdown("Upload a chest X-ray image to receive an AI-generated prediction using Google AutoML Vision.")
    st.info("🔍 This is an experimental AI tool and **not intended for medical diagnosis or clinical use**.")


    # File uploader
    uploaded_file = st.file_uploader("📤 Upload an X-ray image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read file as bytes once
        original_bytes = uploaded_file.read()

        if len(original_bytes) > 1_572_864:
            st.warning("Image is too large — compressing before upload...")

            # Open and compress the image
            image = Image.open(io.BytesIO(original_bytes)).convert("RGB")
            image.thumbnail((512, 512))  # Resize while keeping aspect ratio

            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=85)
            image_bytes = buffer.getvalue()
        else:
            image_bytes = original_bytes
            image = Image.open(io.BytesIO(image_bytes))


        # Display the image using PIL
        image = Image.open(io.BytesIO(image_bytes))
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        if st.button("🔍 Analyze Image"):
        # Encode image to base64
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

        # Step 1: Validate if this is likely a chest X-ray
            with st.spinner("Checking if this is a chest X-ray..."):
                validation_response = chest_validator_endpoint.predict(instances=[{"content": image_b64}])
                validation_predictions = validation_response.predictions[0]

            # Extract confidence score for "Chest X-ray"
                confidence_dict = dict(zip(
                    validation_predictions["displayNames"],
                    validation_predictions["confidences"],

            ))
                chest_confidence = confidence_dict.get("Chest_Xray", 0.0)

                st.write(f"🧠 **Chest X-ray Confidence**: {chest_confidence:.2%}")

                if chest_confidence < 0.8:
                        st.markdown("""
<div style='
    background-color:#fff3cd;
    color:#856404;
    border-left: 8px solid #ffa500;
    padding: 16px;
    font-size: 20px;
    font-weight: bold;
    border-radius: 6px;
    margin-bottom: 20px;
'>
⚠️ <span style='font-size: 26px;'>Warning</span><br><br>
This image has a <u>low probability</u> of being a chest X-ray.<br>
<strong>Proceeding may yield inaccurate results.</strong>
</div>
""", unsafe_allow_html=True)

        
        # Step 2: Run classification model
            with st.spinner("Sending image to the classification model..."):
                response = endpoint.predict(instances=[{"content": image_b64}])
                predictions = response.predictions[0]

            st.write("✅ Prediction complete!")

        # Display predictions above 15%
            for label, score in zip(predictions["displayNames"], predictions["confidences"]):
                if score >= 0.15:
                    st.success(f"**{label}**: {score:.1%}")

        st.markdown("""
---
#### ⚠️ Disclaimer

This tool is for **educational and experimental purposes only**. It uses AI to analyze uploaded images based on patterns learned from publicly available datasets.

**It is not a diagnostic tool.**  
No information provided by this application should be interpreted as medical advice, diagnosis, or treatment. Always consult a licensed medical professional for any health-related concerns.

By using this site, you acknowledge that results may be inaccurate or misleading, and you agree not to use the information provided for medical decision-making.
---
""")


            # Optional warning for low-confidence results

with tab2:
    st.header("📘 Chest X-ray Classifier vs Clinical Diagnosis Performance (Version 1)")

    st.markdown("""
This summary compares the performance of an AI model trained on chest X-rays with published clinical benchmarks.
It highlights areas where the AI aligns with or falls short of typical radiologist-level sensitivity.
""")

    # Manual table using Markdown
    st.markdown("""
| Condition           | Model Performance | Clinical Benchmark | Remarks                                     |
|---------------------|-------------------|---------------------|----------------------------------------------|
| NORMAL              | 0.991             | 95–99%              | Excellent (matches expert-level review)      |
| BACTERIAL PNEUMONIA | 0.866             | 80–90%              | Strong, close to clinical diagnosis          |
| COVID 19            | 0.855             | 70–90%              | Comparable to early COVID imaging studies    |
| PNEUMOTHORAX        | 0.718             | 85–95%              | Needs improvement (misses some cases)        |
| VIRAL PNEUMONIA     | 0.676             | 70–85%              | Slightly below range                         |
| LUNG NODULES        | 0.669             | 65–85%              | Acceptable, on the lower end                 |
| MASS                | 0.633             | 60–80%              | Within clinical variation                    |
| EMPHYSEMA           | 0.506             | 50–75%              | Borderline (often underdiagnosed in X-rays)  |
| PLEURAL THICKENING  | 0.372             | 40–70%              | Significantly underperforming                |
""")

    st.markdown("""
---

### 🔄 Future Improvements
We are actively working to enhance this model's accuracy and reliability. Upcoming updates will include additional training data, better handling of non-chest X-ray inputs, and expanded support for edge-case diagnoses. Our goal is to build a more robust and clinically aligned system through iterative validation and model refinement.

---

### 📚 Sources & References
1. Bairwa, H., & Jangid, R. (2024). *Pneumonia Detection from Chest X-Rays Using the Chexnet Deep Learning Algorithm.* https://doi.org/10.20944/preprints202407.0104.v1  
2. Wang, X., et al. (2017). *ChestX-ray8: Hospital-scale chest X-ray database and benchmarks.* IEEE CVPR. https://doi.org/10.1109/cvpr.2017.369  
3. Johnson, A.E.W., et al. (2019). *MIMIC-CXR: A publicly available database of chest radiographs with reports.* Sci Data, 6, 317. https://doi.org/10.1038/s41597-019-0322-0
""")
