import streamlit as st

st.set_page_config(page_title="Under Maintenance", layout="centered")
tab1, tab2 = st.tabs(["Maintenance Message", "Model Performance"])

with tab1:
    st.markdown("""
<div style='text-align: center; padding-top: 100px;'>
    <h1 style='font-size: 3em;'>🛠️ We're Updating Things</h1>
    <p style='font-size: 1.5em;'>Our Chest X-ray AI tool is currently undergoing maintenance.</p>
    <p style='font-size: 1.2em;'>Please check back later while we make improvements to serve you better.</p> 
    <p style='font-size: 1.2em;'>For any questions in the meantime, please email khalidsm2004@gmail.com. </p>
</div>
""", unsafe_allow_html=True)
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
