import datetime
import gc
import io
import traceback
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import gspread
import joblib
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from transformers import AutoImageProcessor, AutoModel

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="CeraVLM | Maize Pathology Copilot",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- UI Custom Styling ---
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .hero-banner {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 50%, #40916c 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        margin-bottom: 20px;
        text-align: center;
    }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .badge-critical { background: #fee2e2; color: #b91c1c; padding: 4px 10px; border-radius: 15px; font-weight: 600; }
    .badge-warning  { background: #fef3c7; color: #b45309; padding: 4px 10px; border-radius: 15px; font-weight: 600; }
    .badge-healthy  { background: #dcfce7; color: #15803d; padding: 4px 10px; border-radius: 15px; font-weight: 600; }
</style>
""",
    unsafe_allow_html=True,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/spreadsheets",
]


# --- Google Cloud Services Setup ---
def get_google_services():
  try:
    creds_dict = dict(st.secrets["gcp_service_account"])
    raw_key = creds_dict.get("private_key", "")
    if "\\n" in raw_key:
      creds_dict["private_key"] = raw_key.replace("\\n", "\n")
    creds_dict["private_key"] = creds_dict["private_key"].strip()

    credentials = service_account.Credentials.from_service_account_info(
        creds_dict, scopes=SCOPES
    )
    drive_service = build("drive", "v3", credentials=credentials)
    gc_client = gspread.authorize(credentials)
    sheet = gc_client.open_by_key(st.secrets["drive_config"]["sheet_id"]).sheet1
    return drive_service, sheet
  except Exception as e:
    st.error(f"❌ Google Connection Failed ({type(e).__name__}): {repr(e)}")
    return None, None


def upload_to_drive(drive_service, folder_id, image_bytes, filename):
  media = MediaIoBaseUpload(
      io.BytesIO(image_bytes), mimetype="image/jpeg", resumable=False
  )
  file_metadata = {"name": filename, "parents": [folder_id]}
  uploaded = (
      drive_service.files()
      .create(
          body=file_metadata,
          media_body=media,
          fields="id",
          supportsAllDrives=True,  # Bypasses service account quota restrictions
      )
      .execute()
  )
  file_id = uploaded.get("id")
  return f"https://drive.google.com/file/d/{file_id}/view"


# --- Agronomic Protocols & Prescriptions ---
MAIZE_AGRONOMIC_PROTOCOLS = {
    "Charcoal_Rot": {
        "agent": "Macrophomina phaseolina",
        "severity": "High (Stalk Lodging)",
        "badge": "badge-critical",
        "chemical": (
            "Carbendazim 50 WP (1 g/L) or Thiophanate-Methyl 70 WP (1 g/L)."
        ),
    },
    "Fall_Army_worm": {
        "agent": "Spodoptera frugiperda",
        "severity": "Critical (Whorl Defoliation)",
        "badge": "badge-critical",
        "chemical": (
            "Chlorantraniliprole 18.5 SC @ 0.4 ml/L OR Emamectin Benzoate 5 SG @"
            " 0.4 g/L."
        ),
    },
    "Grey_Leaf_Spot": {
        "agent": "Cercospora zeae-maydis",
        "severity": "Moderate to High",
        "badge": "badge-warning",
        "chemical": (
            "Azoxystrobin 18.2% + Difenoconazole 11.4% SC @ 1 ml/L water."
        ),
    },
    "Maydis_Leaf_Blight": {
        "agent": "Bipolaris maydis",
        "severity": "Moderate",
        "badge": "badge-warning",
        "chemical": "Mancozeb 75 WP @ 2.5 g/L OR Azoxystrobin 23 SC @ 1 ml/L.",
    },
    "Nitrogen_Deficiency_in_Maize": {
        "agent": "Macronutrient Deficiency (N)",
        "severity": "Nutritional Stress",
        "badge": "badge-warning",
        "chemical": "Top-dress Urea @ 40 kg/ha OR 2% foliar Urea spray.",
    },
    "Northern_Leaf_Blight": {
        "agent": "Exserohilum turcicum",
        "severity": "High (Foliar Necrosis)",
        "badge": "badge-critical",
        "chemical": "Propiconazole 25 EC @ 1 ml/L OR Mancozeb 75 WP @ 2.5 g/L.",
    },
    "Phosphorus_Deficiency": {
        "agent": "Macronutrient Deficiency (P)",
        "severity": "Nutritional Stress",
        "badge": "badge-warning",
        "chemical": (
            "Foliar 12:61:0 @ 5 g/L OR Single Super Phosphate @ 100 kg/ha."
        ),
    },
    "Potassium_Deficiency_in_Maize": {
        "agent": "Macronutrient Deficiency (K)",
        "severity": "Nutritional Stress",
        "badge": "badge-warning",
        "chemical": "MOP @ 40 kg/ha OR 13:0:45 foliar spray @ 10 g/L.",
    },
    "Rust_in_Maize": {
        "agent": "Puccinia sorghi",
        "severity": "Moderate to High",
        "badge": "badge-warning",
        "chemical": "Tebuconazole 25.9 EC @ 1 ml/L OR Mancozeb 75 WP @ 2.5 g/L.",
    },
    "Stem_Borer": {
        "agent": "Chilo partellus",
        "severity": "Critical (Dead Heart)",
        "badge": "badge-critical",
        "chemical": (
            "Carbofuran 3G @ 5 kg/ha in whorls OR Chlorantraniliprole 18.5 SC @"
            " 0.4 ml/L."
        ),
    },
    "Trips": {
        "agent": "Frankliniella spp.",
        "severity": "Moderate",
        "badge": "badge-warning",
        "chemical": (
            "Thiamethoxam 25 WG @ 0.3 g/L OR Imidacloprid 17.8 SL @ 0.3 ml/L."
        ),
    },
    "Zinc_Deficiency_in_Maize": {
        "agent": "Micronutrient Deficiency (Zn)",
        "severity": "White Bud",
        "badge": "badge-warning",
        "chemical": "0.5% ZnSO4 (5 g/L) + 0.25% lime spray.",
    },
}


# --- Model Loading ---
@st.cache_resource
def load_vision_pipeline():
  clf = joblib.load("maize_vlm_classifier.joblib")
  scaler = joblib.load("maize_vlm_scaler.joblib")
  class_names = joblib.load("maize_class_names.joblib")
  vision_id = "google/siglip-base-patch16-384"
  processor = AutoImageProcessor.from_pretrained(vision_id)
  vision_model = AutoModel.from_pretrained(
      vision_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
  ).to(device)
  vision_model.eval()
  gc.collect()
  return clf, scaler, class_names, processor, vision_model


clf, scaler, class_names, processor, vision_model = load_vision_pipeline()

# --- Main Interface ---
st.markdown(
    """
<div class="hero-banner">
    <h2>🌽 Maize Pathology VLM Copilot (Batch Diagnostic Edition)</h2>
    <p>Upload field leaf specimens for parallel diagnosis, cloud tracking, and export.</p>
</div>
""",
    unsafe_allow_html=True,
)

uploaded_files = st.file_uploader(
    "Upload Maize Leaf Images (Multiple selections supported):",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if uploaded_files:
  st.write(f"📁 **Selected {len(uploaded_files)} specimen(s)**")
  run_batch = st.button(
      "🚀 Run Batch Diagnostics & Cloud Logging",
      type="primary",
      use_container_width=True,
  )

  if run_batch:
    drive_service, sheet = get_google_services()
    folder_id = st.secrets["drive_config"]["folder_id"] if drive_service else ""

    results_data = []
    rows_for_sheet = []
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, file in enumerate(uploaded_files):
      status_text.text(
          f"Processing image {idx+1}/{len(uploaded_files)}: {file.name}..."
      )
      image = Image.open(file).convert("RGB")

      # SigLIP Feature Extraction & Inference
      inputs = processor(images=[image], return_tensors="pt").to(
          device, dtype=torch.float32
      )
      with torch.no_grad():
        out = vision_model.vision_model(
            inputs["pixel_values"]
        ).last_hidden_state
        feat = torch.cat([out.mean(dim=1), out.max(dim=1).values], dim=-1)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        feat_scaled = scaler.transform(feat.cpu().float().numpy())

      probs = clf.predict_proba(feat_scaled)[0]
      top_idx = int(np.argmax(probs))
      max_conf = float(probs[top_idx]) * 100

      if max_conf < 60.0:
        diagnosis_title = "Healthy Maize Specimen"
        severity = "Healthy Crop / Asymptomatic"
        pathogen = "None"
        prescription = "Maintain routine irrigation and scouting."
        badge = "badge-healthy"
      else:
        raw_class = class_names[top_idx]
        diagnosis_title = raw_class.replace("_", " ")
        diag_meta = MAIZE_AGRONOMIC_PROTOCOLS.get(
            raw_class,
            {
                "agent": "Unknown",
                "severity": "Check Field",
                "badge": "badge-warning",
                "chemical": "Consult extension agent.",
            },
        )
        severity = diag_meta["severity"]
        pathogen = diag_meta["agent"]
        prescription = diag_meta["chemical"]
        badge = diag_meta["badge"]

      # Cloud Logging with Fallback
      now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
      cloud_filename = (
          f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.name}"
      )
      drive_url = "Direct Upload Only (Quota Bypass)"

      if drive_service and folder_id:
        try:
          drive_url = upload_to_drive(
              drive_service, folder_id, file.getvalue(), cloud_filename
          )
        except Exception as drive_err:
          # Catches Google Drive quota block without failing the Sheets logging
          st.warning(
              f"⚠️ Drive image storage skipped for `{file.name}` (personal Drive"
              " quota limit). Logging diagnosis directly to Sheet."
          )
          drive_url = "Direct Upload Only"

      # Append row data for Google Sheets
      rows_for_sheet.append([
          now_str,
          cloud_filename,
          drive_url,
          diagnosis_title,
          f"{max_conf:.2f}%",
          severity,
      ])

      results_data.append({
          "Image Name": file.name,
          "Diagnosis": diagnosis_title,
          "Confidence": f"{max_conf:.2f}%",
          "Severity": severity,
          "Causal Factor": pathogen,
          "Recommended Prescription": prescription,
          "Drive Storage": drive_url,
          "Badge": badge,
      })

      progress_bar.progress((idx + 1) / len(uploaded_files))

    # Append to Google Sheet
    if sheet and rows_for_sheet:
      try:
        sheet.append_rows(rows_for_sheet)
        st.toast(
            f"✅ Successfully logged {len(rows_for_sheet)} record(s) to Google"
            " Sheets!"
        )
      except Exception as e:
        st.error(f"❌ Google Sheets append error: {e}")

    status_text.text("✅ All specimens analyzed successfully!")

    # Display Diagnostic Cards
    st.markdown("### 📊 Diagnostic Results")
    for res in results_data:
      st.markdown(
          f"""
            <div class="metric-card">
                <span class="{res['Badge']}">{res['Severity']}</span>
                <h3 style="margin: 8px 0 4px 0; color: #1b4332;">{res['Diagnosis']} (Confidence: {res['Confidence']})</h3>
                <p style="margin: 2px 0;"><b>Specimen:</b> {res['Image Name']} | <b>Agent:</b> <i>{res['Causal Factor']}</i></p>
                <p style="margin: 2px 0;"><b>Rx Protocol:</b> {res['Recommended Prescription']}</p>
                <p style="margin: 2px 0; font-size: 0.85rem;"><a href="{res['Drive Storage']}" target="_blank">🔗 Drive Record</a></p>
            </div>
            """,
          unsafe_allow_html=True,
      )

    # Export Diagnostic Summary
    st.markdown("---")
    st.markdown("### 📥 Download Diagnostic Report")
    df_report = pd.DataFrame(results_data).drop(columns=["Badge"])
    csv_bytes = df_report.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📄 Download Diagnostic Summary as CSV",
        data=csv_bytes,
        file_name=f"maize_pathology_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )
