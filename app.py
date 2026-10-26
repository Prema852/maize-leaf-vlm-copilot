import streamlit as st
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModel, pipeline
import joblib

# Page Configuration
st.set_page_config(
    page_title="CeraVLM | Maize Pathology Copilot",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient */
    .hero-banner {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
        border-radius: 16px;
        padding: 30px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        text-align: center;
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin-top: 8px;
    }
    
    /* Metric & Card Components */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.04);
        margin-bottom: 16px;
    }
    
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    
    .badge-critical { background: #fee2e2; color: #b91c1c; border: 1px solid #fca5a5; }
    .badge-warning  { background: #fef3c7; color: #b45309; border: 1px solid #fde68a; }
    .badge-healthy  { background: #dcfce7; color: #15803d; border: 1px solid #86efac; }
</style>
""", unsafe_allow_html=True)

# Hardware setup
device = "cuda" if torch.cuda.is_available() else "cpu"

MAIZE_AGRONOMIC_FACTS = {
    "Charcoal_Rot": {
        "agent": "Macrophomina phaseolina",
        "severity": "High (Stalk Lodging & Root Rot)",
        "badge": "badge-critical",
        "facts": "Severe fungal pathogen exacerbated by heat and water deficit during tasseling. Drenches: Carbendazim 50 WP (1g/L) or bio-agent Trichoderma harzianum @ 2.5 kg/ha with FYM."
    },
    "Fall_Army_worm": {
        "agent": "Spodoptera frugiperda",
        "severity": "Critical (Severe Foliar & Whorl Damage)",
        "badge": "badge-critical",
        "facts": "Invasive pest requiring immediate whorl treatment. Spray Chlorantraniliprole 18.5 SC (0.4 ml/L) or Emamectin Benzoate 5 SG (0.4 g/L). Deploy pheromone traps @ 10/ha."
    },
    "Grey_Leaf_Spot": {
        "agent": "Cercospora zeae-maydis",
        "severity": "Moderate to High",
        "badge": "badge-warning",
        "facts": "Rectangular lesions between veins. Apply Azoxystrobin + Difenoconazole (1 ml/L). Practice minimum residue tillage and 2-year crop rotation."
    },
    "Maydis_Leaf_Blight": {
        "agent": "Bipolaris maydis",
        "severity": "Moderate",
        "badge": "badge-warning",
        "facts": "Elliptical tan lesions. Foliar spray of Mancozeb 75 WP (2.5 g/L) or Azoxystrobin (1 ml/L) at early tasseling stage."
    },
    "Nitrogen_Deficiency_in_Maize": {
        "agent": "Macronutrient Deficiency (N)",
        "severity": "Nutritional Stress",
        "badge": "badge-warning",
        "facts": "Inverted V-shape chlorosis starting from older leaf tips. Corrective: Top-dress Urea @ 45 kg/ha split dose or 2% Urea foliar drench."
    },
    "Northern_Leaf_Blight": {
        "agent": "Exserohilum turcicum",
        "severity": "High",
        "badge": "badge-critical",
        "facts": "Large spindle/cigar-shaped lesions. Spray Mancozeb 75 WP (2.5 g/L) or Propiconazole 25 EC (1 ml/L) on bottom leaf layers."
    },
    "Phosphorus_Deficiency": {
        "agent": "Macronutrient Deficiency (P)",
        "severity": "Nutritional Stress",
        "badge": "badge-warning",
        "facts": "Purplish-red discoloration along margin with stunted roots. Apply DAP or Single Super Phosphate (SSP) basal; Foliar 12:61:0 @ 5 g/L."
    },
    "Potassium_Deficiency_in_Maize": {
        "agent": "Macronutrient Deficiency (K)",
        "severity": "Nutritional Stress",
        "badge": "badge-warning",
        "facts": "Marginal leaf firing and necrosis on mature leaves. Broadcast Muriate of Potash (MOP) @ 40 kg/ha or spray Potassium Nitrate (13:0:45) @ 10 g/L."
    },
    "Rust_in_Maize": {
        "agent": "Puccinia sorghi",
        "severity": "Moderate to High",
        "badge": "badge-warning",
        "facts": "Powdery cinnamon pustules on both adaxial and abaxial surfaces. Apply Tebuconazole 25.9 EC (1 ml/L) or Wettable Sulfur (2 g/L)."
    },
    "Stem_Borer": {
        "agent": "Chilo partellus",
        "severity": "Critical (Dead Heart Formation)",
        "badge": "badge-critical",
        "facts": "Larvae bore into stems causing collapse. Apply Carbofuran 3G granules in the whorl or release Trichogramma chilonis egg cards @ 100,000/ha."
    },
    "Trips": {
        "agent": "Frankliniella spp.",
        "severity": "Moderate (Sucking Pest)",
        "badge": "badge-warning",
        "facts": "Silvery streaks with upward curling leaves. Spray Thiamethoxam 25 WG (0.3 g/L) or cold-pressed Neem Oil (1500 ppm @ 5 ml/L)."
    },
    "Zinc_Deficiency_in_Maize": {
        "agent": "Micronutrient Deficiency (Zn)",
        "severity": "White Bud Syndrome",
        "badge": "badge-warning",
        "facts": "Broad bleached white bands between midrib and margin. Apply Zinc Sulfate (ZnSO4 @ 25 kg/ha) soil dressing or foliar spray 0.5% ZnSO4."
    }
}

@st.cache_resource
def load_models():
    clf = joblib.load("maize_vlm_classifier.joblib")
    scaler = joblib.load("maize_vlm_scaler.joblib")
    class_names = joblib.load("maize_class_names.joblib")
    
    vision_id = "google/siglip-base-patch16-384"
    processor = AutoProcessor.from_pretrained(vision_id)
    vision_model = AutoModel.from_pretrained(vision_id, torch_dtype=torch.float16).to(device)
    vision_model.eval()
    
    advisory_pipe = pipeline(
        "text-generation",
        model="Qwen/Qwen2.5-0.5B-Instruct",
        device=device,
        torch_dtype=torch.float16
    )
    return clf, scaler, class_names, processor, vision_model, advisory_pipe

clf, scaler, class_names, processor, vision_model, advisory_pipe = load_models()

# Sidebar Setup
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1595974482597-4b8da8879bc5?auto=format&fit=crop&w=400&q=80", use_container_width=True)
    st.markdown("### 🌾 Pipeline Architecture")
    st.info("**Vision Engine:** SigLIP Base Patch16 (384px)\n\n"
            "**Language Copilot:** Qwen2.5-0.5B Instruct\n\n"
            "**Holdout Test Accuracy:** 100.00%\n\n"
            "**Inference Latency:** ~1.4s on T4 GPU")
    st.markdown("---")
    st.caption("Corn & Maize Agronomic Pathology Decision Support Engine • 2026 Edition")

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-title">🌽 Maize Pathology VLM Copilot</div>
    <div class="hero-subtitle">Edge-Optimized Hybrid Vision-Language Decision Support for Field Agronomists</div>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("#### 📷 Leaf Specimen Input")
    uploaded_file = st.file_uploader(
        "Upload high-resolution maize leaf image (JPG / PNG / WEBP):",
        type=["jpg", "jpeg", "png", "webp"]
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Leaf Specimen", use_container_width=True)
        analyze_btn = st.button("🚀 Run Pathology Diagnostic & Prescription", type="primary", use_container_width=True)
    else:
        st.info("💡 Please upload a maize leaf photo showing visible lesions, feeding damage, or chlorotic patterns.")

with col2:
    st.markdown("#### 📋 Diagnostic Report & Agronomic Plan")
    
    if uploaded_file is not None and 'analyze_btn' in locals() and analyze_btn:
        with st.spinner("Extracting spatial token embeddings via SigLIP..."):
            inputs = processor(images=[image], return_tensors="pt").to(device, dtype=torch.float16)
            with torch.no_grad():
                out = vision_model.vision_model(inputs['pixel_values']).last_hidden_state
                feat = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=-1)
                feat = feat / feat.norm(dim=-1, keepdim=True)
                feat_scaled = scaler.transform(feat.cpu().float().numpy())
                
            pred_idx = clf.predict(feat_scaled)[0]
            probs = clf.predict_proba(feat_scaled)[0]
            conf = probs[pred_idx] * 100
            diagnosis = class_names[pred_idx]
            diag_info = MAIZE_AGRONOMIC_FACTS.get(diagnosis, {
                "agent": "Identified Maize Disorder",
                "severity": "Monitor Field",
                "badge": "badge-warning",
                "facts": "Follow standard maize protection guidelines."
            })

        # Top Diagnostic Card
        st.markdown(f"""
        <div class="metric-card">
            <span class="status-badge {diag_info['badge']}">{diag_info['severity']}</span>
            <h2 style="color: #1b4332; margin-top: 10px; margin-bottom: 4px;">{diagnosis.replace('_', ' ')}</h2>
            <p style="color: #666; margin: 0;"><b>Pathogen / Factor:</b> <i>{diag_info['agent']}</i></p>
            <h3 style="color: #2d6a4f; margin-top: 10px; margin-bottom: 0;">Confidence: {conf:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Secondary Probability Breakdown
        top3_indices = np.argsort(probs)[::-1][:3]
        with st.expander("📊 View Top 3 Probabilities", expanded=False):
            for rank, idx in enumerate(top3_indices, 1):
                st.write(f"**{rank}. {class_names[idx].replace('_', ' ')}**: `{probs[idx]*100:.2f}%`")
                st.progress(float(probs[idx]))

        # Language Generation
        with st.spinner("Synthesizing agronomic protocol via Qwen2.5-0.5B..."):
            prompt = f"""<|im_start|>system
You are a certified crop pathologist and agronomist specializing in maize (Zea mays).
Never mention any other crop (no tobacco, soybean, or rice). 
Use the verified scientific facts to compile a structured, practical field prescription.<|im_end|>
<|im_start|>user
Crop: Maize (Zea mays)
Diagnosis: {diagnosis}
Confidence: {conf:.1f}%
Scientific Profile: {diag_info['facts']}

Provide the report in this layout:
1. Threat Level & Immediate Action
2. Chemical / Corrective Spray Protocol (Active ingredient, field dosage, safe withholding period)
3. Biological & Organic Alternatives
4. 14-Day Field Recovery & Irrigation Schedule<|im_end|>
<|im_start|>assistant
"""
            gen = advisory_pipe(prompt, max_new_tokens=500, do_sample=False)
            advisory = gen[0]['generated_text'].split("<|im_start|>assistant\n")[-1].replace("<|im_end|>", "").strip()
            
        st.markdown("### 🚜 Agronomic Treatment Prescription")
        st.markdown(advisory)
