"""
=============================================================================
Maize Pathology VLM - Core Vision Inference & Feature Extraction Pipeline
Architecture: Google SigLIP (Patch16-384) + Calibrated Scikit-Learn Head
Production Environment: Streamlit Community Cloud (Compatible with CPU & CUDA)
=============================================================================
"""

import gc
from pathlib import Path
import joblib
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel

# -----------------------------------------------------------------------------
# 1. Hardware & Environment Setup
# -----------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------------------------------
# 2. Grounded Agronomic Protocol Knowledge Base (12 Calibrated Conditions)
# -----------------------------------------------------------------------------
MAIZE_AGRONOMIC_PROTOCOLS = {
    "Charcoal_Rot": {
        "agent": "Macrophomina phaseolina",
        "severity": "High (Stalk Lodging & Root Rot)",
        "chemical": "Carbendazim 50 WP (1 g/L) or Thiophanate-Methyl 70 WP (1 g/L)."
    },
    "Fall_Army_worm": {
        "agent": "Spodoptera frugiperda",
        "severity": "Critical (Severe Foliar & Whorl Damage)",
        "chemical": "Chlorantraniliprole 18.5 SC @ 0.4 ml/L OR Emamectin Benzoate 5 SG @ 0.4 g/L."
    },
    "Grey_Leaf_Spot": {
        "agent": "Cercospora zeae-maydis",
        "severity": "Moderate to High",
        "chemical": "Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L water."
    },
    "Maydis_Leaf_Blight": {
        "agent": "Bipolaris maydis",
        "severity": "Moderate",
        "chemical": "Mancozeb 75 WP @ 2.5 g/L OR Azoxystrobin 23 SC @ 1 ml/L."
    },
    "Nitrogen_Deficiency_in_Maize": {
        "agent": "Macronutrient Deficiency (N)",
        "severity": "Nutritional Stress",
        "chemical": "Top-dress Urea @ 40-50 kg/ha split dose OR 2% foliar Urea spray."
    },
    "Northern_Leaf_Blight": {
        "agent": "Exserohilum turcicum",
        "severity": "High (Foliar Necrosis)",
        "chemical": "Propiconazole 25 EC @ 1 ml/L OR Mancozeb 75 WP @ 2.5 g/L."
    },
    "Phosphorus_Deficiency": {
        "agent": "Macronutrient Deficiency (P)",
        "severity": "Nutritional Stress",
        "chemical": "Foliar 12:61:0 @ 5 g/L OR Single Super Phosphate @ 100 kg/ha basal."
    },
    "Potassium_Deficiency_in_Maize": {
        "agent": "Macronutrient Deficiency (K)",
        "severity": "Nutritional Stress",
        "chemical": "MOP @ 40-50 kg/ha OR 13:0:45 foliar spray @ 10 g/L."
    },
    "Rust_in_Maize": {
        "agent": "Puccinia sorghi",
        "severity": "Moderate to High",
        "chemical": "Tebuconazole 25.9 EC @ 1 ml/L OR Mancozeb 75 WP @ 2.5 g/L."
    },
    "Stem_Borer": {
        "agent": "Chilo partellus",
        "severity": "Critical (Dead Heart Formation)",
        "chemical": "Carbofuran 3G @ 5 kg/ha in whorls OR Chlorantraniliprole 18.5 SC @ 0.4 ml/L."
    },
    "Trips": {
        "agent": "Frankliniella spp.",
        "severity": "Moderate (Sucking Pest)",
        "chemical": "Thiamethoxam 25 WG @ 0.3 g/L OR Imidacloprid 17.8 SL @ 0.3 ml/L."
    },
    "Zinc_Deficiency_in_Maize": {
        "agent": "Micronutrient Deficiency (Zn)",
        "severity": "White Bud Syndrome",
        "chemical": "Soil ZnSO4 @ 25 kg/ha OR foliar spray of 0.5% ZnSO4 (5 g/L) + 0.25% lime."
    }
}

# -----------------------------------------------------------------------------
# 3. Model Pipeline Loader (SigLIP Backbone & Pre-Trained Classifiers)
# -----------------------------------------------------------------------------
def load_vlm_pipeline(
    classifier_path="maize_vlm_classifier.joblib",
    scaler_path="maize_vlm_scaler.joblib",
    class_names_path="maize_class_names.joblib",
    vision_id="google/siglip-base-patch16-384"
):
    """
    Loads visual token processor, SigLIP backbone, feature scaler, and calibrated classifier head.
    """
    print(f"[*] Loading Stage 1 Vision Backbone: {vision_id} on {device.upper()}...")
    clf = joblib.load(classifier_path)
    scaler = joblib.load(scaler_path)
    class_names = joblib.load(class_names_path)

    processor = AutoImageProcessor.from_pretrained(vision_id)
    vision_model = AutoModel.from_pretrained(
        vision_id,
        torch_dtype=torch.float32,   # float32 ensures seamless deployment across both Cloud CPU and GPU
        low_cpu_mem_usage=True
    ).to(device)
    vision_model.eval()
    gc.collect()

    print("[✔] Visual Backbone & Diagnostic Artifacts Loaded Successfully.")
    return clf, scaler, class_names, processor, vision_model

# -----------------------------------------------------------------------------
# 4. Feature Extraction & Pathological Diagnosis Function
# -----------------------------------------------------------------------------
def run_vlm_inference(image_input, clf, scaler, class_names, processor, vision_model):
    """
    Executes dense spatial token pooling and classification on a maize foliar specimen.
    Accepts a filepath string or a PIL Image instance.
    """
    if isinstance(image_input, (str, Path)):
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input.convert("RGB")

    # Extract spatial token representations via Google SigLIP
    inputs = processor(images=[img], return_tensors="pt").to(device, dtype=torch.float32)
    with torch.no_grad():
        out = vision_model.vision_model(inputs["pixel_values"]).last_hidden_state
        # Dual spatial pooling: Concatenate Global Average Pooling + Global Max Pooling
        mean_pooled = out.mean(dim=1)
        max_pooled = out.max(dim=1).values
        feat = torch.cat([mean_pooled, max_pooled], dim=-1)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat_scaled = scaler.transform(feat.cpu().float().numpy())

    # Multi-class calibrated classification
    probs = clf.predict_proba(feat_scaled)[0]
    top_idx = int(np.argmax(probs))
    max_conf = float(probs[top_idx]) * 100.0

    # Probabilistic thresholding for asymptomatic/healthy crops
    if max_conf < 60.0:
        return {
            "Diagnosis": "Healthy Maize Specimen",
            "Confidence": f"{max_conf:.2f}%",
            "Raw_Confidence": max_conf,
            "Severity": "Healthy Crop / Asymptomatic",
            "Causal_Agent": "None",
            "Prescription": "Maintain routine fertigation and periodic foliar scouting.",
            "Top_3_Probabilities": [("Healthy Crop", max_conf)]
        }

    raw_class = class_names[top_idx]
    diagnosis_title = raw_class.replace("_", " ")
    diag_meta = MAIZE_AGRONOMIC_PROTOCOLS.get(raw_class, {
        "agent": "Identified Maize Disorder",
        "severity": "Check Field Plot",
        "chemical": "Consult regional agricultural extension."
    })

    # Sort top 3 probabilities for multi-diagnosis audit
    top3_indices = np.argsort(probs)[::-1][:3]
    top3_probs = [(class_names[i].replace("_", " "), float(probs[i]) * 100.0) for i in top3_indices]

    return {
        "Diagnosis": diagnosis_title,
        "Confidence": f"{max_conf:.2f}%",
        "Raw_Confidence": max_conf,
        "Severity": diag_meta["severity"],
        "Causal_Agent": diag_meta["agent"],
        "Prescription": diag_meta["chemical"],
        "Top_3_Probabilities": top3_probs
    }

# -----------------------------------------------------------------------------
# 5. CLI Verification Routine
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    # Provide sample image path as CLI argument or default to test.jpg
    test_image = sys.argv[1] if len(sys.argv) > 1 else "sample_leaf.jpg"

    if Path(test_image).exists():
        clf, scaler, class_names, processor, vision_model = load_vlm_pipeline()
        result = run_vlm_inference(test_image, clf, scaler, class_names, processor, vision_model)

        print("\n" + "=" * 65)
        print(f"📊 SPECIMEN DIAGNOSTIC RESULT: {Path(test_image).name}")
        print("=" * 65)
        print(f"Diagnosis:        {result['Diagnosis']}")
        print(f"Confidence:       {result['Confidence']}")
        print(f"Severity:         {result['Severity']}")
        print(f"Causal Agent:     {result['Causal_Agent']}")
        print(f"Prescription:     {result['Prescription']}")
        print("\nTop 3 Probabilities:")
        for name, prob in result["Top_3_Probabilities"]:
            print(f"  - {name}: {prob:.2f}%")
        print("=" * 65)
    else:
        print(f"[!] Test specimen '{test_image}' not found.")
        print("Usage: python vlm_pipeline.py <path_to_leaf_image.jpg>")
