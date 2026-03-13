import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
from deep_translator import GoogleTranslator
import io
import datetime


from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT


st.set_page_config(
    page_title="AGRICARE 🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)


_defaults = {
    "img_bytes":     None,
    "temp_path":     "temp_agricare.jpg",
    "last_file_id":  None,
    "rice_ok":       None,
    "rice_conf":     None,
    "disease":       None,
    "disease_conf":  None,
    "severity":      None,
    "severity_pct":  None,
    "gradcam_orig":  None,
    "gradcam_heat":  None,
    "gradcam_cam":   None,
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet">

<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

.stApp {
    background: #080d08;
    background-image:
        radial-gradient(ellipse 70% 55% at 50% -5%, rgba(22,163,74,0.20) 0%, transparent 65%),
        radial-gradient(ellipse 50% 40% at 85% 95%,  rgba(15,50,15,0.30)  0%, transparent 60%),
        url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%2316a34a' fill-opacity='0.025'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E");
    min-height: 100vh;
    font-family: 'DM Sans', sans-serif;
}
#MainMenu, footer, header, [data-testid="stToolbar"] { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stDecoration"] { display: none; }
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #080d08; }
::-webkit-scrollbar-thumb { background: #16a34a55; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #16a34a; }
.block-container {
    max-width: 1080px !important;
    padding-top: 0.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
.agricare-hero {
    text-align: center;
    padding: 2.4rem 1rem 1.4rem;
    position: relative;
}
.agricare-hero::before {
    content: '';
    position: absolute;
    top: -30px; left: 50%; transform: translateX(-50%);
    width: 680px; height: 300px;
    background: radial-gradient(ellipse, rgba(34,197,94,0.12) 0%, transparent 70%);
    pointer-events: none; z-index: 0;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(22,163,74,0.10);
    border: 1px solid rgba(22,163,74,0.32);
    color: #4ade80;
    font-family: 'Space Mono', monospace;
    font-size: 0.67rem; letter-spacing: 0.15em; text-transform: uppercase;
    padding: 5px 15px; border-radius: 100px; margin-bottom: 1.3rem;
    position: relative; z-index: 1;
    animation: fadeSlideDown 0.5s ease both;
}
.hero-badge .dot {
    width: 6px; height: 6px; background: #4ade80; border-radius: 50%; flex-shrink: 0;
    animation: pulse-dot 2.2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%,100% { opacity:1; transform:scale(1); }
    50%      { opacity:0.55; transform:scale(0.7); }
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(3.4rem, 8vw, 6.2rem);
    font-weight: 900; color: #f0fdf4;
    line-height: 1.0; letter-spacing: -0.025em;
    margin-bottom: 0.75rem;
    position: relative; z-index: 1;
    animation: fadeSlideDown 0.6s ease 0.08s both;
}
.hero-title .green {
    background: linear-gradient(135deg, #86efac 0%, #22c55e 45%, #166534 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.hero-sub {
    color: #6b7280; font-size: 1.0rem; font-weight: 300; font-style: italic;
    max-width: 440px; margin: 0 auto 1.9rem; line-height: 1.8;
    position: relative; z-index: 1;
    animation: fadeSlideDown 0.6s ease 0.16s both;
}
.stat-row {
    display: flex; justify-content: center; gap: 0.8rem; flex-wrap: wrap;
    margin-bottom: 2rem; position: relative; z-index: 1;
    animation: fadeSlideDown 0.6s ease 0.24s both;
}
.stat-pill {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.065);
    color: #9ca3af; font-size: 0.75rem; padding: 6px 15px; border-radius: 100px;
    font-family: 'Space Mono', monospace; letter-spacing: 0.04em;
}
.stat-pill strong { color: #4ade80; margin-right: 5px; }
.fancy-divider {
    display: flex; align-items: center; gap: 1rem;
    margin: 0 auto 2rem; max-width: 380px;
    position: relative; z-index: 1;
    animation: fadeSlideDown 0.6s ease 0.30s both;
}
.fancy-divider::before { content:''; flex:1; height:1px; background:linear-gradient(to right, transparent, rgba(74,222,128,0.22)); }
.fancy-divider::after  { content:''; flex:1; height:1px; background:linear-gradient(to left,  transparent, rgba(74,222,128,0.22)); }
.fancy-divider span { color: #4ade80; font-size: 0.95rem; opacity: 0.55; }
.step-label {
    font-family: 'Space Mono', monospace; font-size: 0.7rem; letter-spacing: 0.14em;
    text-transform: uppercase; color: #4ade80; opacity: 0.6; margin-bottom: 0.5rem; display: block;
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.016) !important;
    border: 2px dashed rgba(74,222,128,0.20) !important;
    border-radius: 18px !important; padding: 1.5rem 1.8rem !important;
    transition: border-color .3s, background .3s, box-shadow .3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(74,222,128,0.48) !important;
    background: rgba(74,222,128,0.03) !important;
    box-shadow: 0 5px 26px rgba(22,163,74,0.10) !important;
}
[data-testid="stFileUploader"] label {
    color: #a3c9a8 !important; font-family: 'DM Sans', sans-serif !important; font-size: 0.93rem !important;
}
[data-testid="stFileDropzone"] { background: transparent !important; border: none !important; }
.stButton > button {
    background: linear-gradient(135deg, #16a34a 0%, #15803d 100%) !important;
    color: #f0fdf4 !important; border: none !important; border-radius: 11px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important;
    letter-spacing: 0.03em !important; padding: 0.55rem 1.4rem !important;
    transition: all .22s ease !important; box-shadow: 0 4px 14px rgba(22,163,74,0.26) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 22px rgba(22,163,74,0.40) !important;
}

/* ── DOWNLOAD BUTTON ── */
[data-testid="stDownloadButton"] > button {
    width: 100% !important;
    background: linear-gradient(135deg, #052e16 0%, #14532d 50%, #166534 100%) !important;
    color: #86efac !important;
    border: 1px solid rgba(74,222,128,0.30) !important;
    border-radius: 14px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.80rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    padding: 0.85rem 1.8rem !important;
    transition: all .25s ease !important;
    box-shadow: 0 4px 20px rgba(22,163,74,0.18), inset 0 1px 0 rgba(74,222,128,0.12) !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: linear-gradient(135deg, #14532d 0%, #166534 50%, #16a34a 100%) !important;
    color: #f0fdf4 !important;
    box-shadow: 0 8px 28px rgba(22,163,74,0.38) !important;
    transform: translateY(-2px) !important;
    border-color: rgba(74,222,128,0.55) !important;
}

.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(74,222,128,0.20) !important;
    border-radius: 10px !important; color: #d1fae5 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.9rem !important;
}
.stSelectbox label {
    color: #6b7280 !important; font-size: 0.72rem !important;
    font-family: 'Space Mono', monospace !important;
    letter-spacing: 0.07em !important; text-transform: uppercase !important;
}
.result-card {
    background: rgba(255,255,255,0.022); border: 1px solid rgba(255,255,255,0.065);
    border-radius: 20px; padding: 1.5rem 1.7rem; margin: 0.7rem 0;
    position: relative; overflow: hidden;
    transition: border-color .3s ease, box-shadow .3s ease;
}
.result-card::before {
    content:''; position:absolute; inset:0;
    background: linear-gradient(140deg, rgba(74,222,128,0.038) 0%, transparent 55%);
    pointer-events: none;
}
.result-card:hover { border-color: rgba(74,222,128,0.20); box-shadow: 0 4px 22px rgba(22,163,74,0.07); }
.result-card-label {
    font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.17em;
    text-transform:uppercase; color:#4ade80; margin-bottom:0.6rem; opacity:0.72;
}
.result-card-value { font-family:'Playfair Display',serif; font-size:1.6rem; font-weight:700; color:#f0fdf4; line-height:1.25; }
.result-card-sub { color:#6b7280; font-size:0.83rem; margin-top:0.3rem; font-style:italic; }
.conf-bar-wrap { margin-top: 1rem; }
.conf-bar-label { display:flex; justify-content:space-between; color:#9ca3af; font-size:0.75rem; margin-bottom:6px; font-family:'Space Mono',monospace; }
.conf-bar-track { height:5px; background:rgba(255,255,255,0.065); border-radius:100px; overflow:hidden; }
.conf-bar-fill  { height:100%; border-radius:100px; background:linear-gradient(90deg, #15803d, #4ade80); }
.severity-badge {
    display:inline-flex; align-items:center; gap:6px; padding:6px 15px;
    border-radius:100px; font-family:'Space Mono',monospace;
    font-size:0.75rem; font-weight:700; letter-spacing:0.09em; text-transform:uppercase; margin-top:0.45rem;
}
.severity-low    { background:rgba(74,222,128,0.11); color:#4ade80; border:1px solid rgba(74,222,128,0.26); }
.severity-medium { background:rgba(251,191,36,0.11);  color:#fbbf24; border:1px solid rgba(251,191,36,0.26); }
.severity-high   { background:rgba(239,68,68,0.11);   color:#f87171; border:1px solid rgba(239,68,68,0.26);  }
.section-header {
    font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:700; color:#f0fdf4;
    margin:2.6rem 0 1rem; display:flex; align-items:center; gap:0.7rem;
}
.section-header::after { content:''; flex:1; height:1px; background:linear-gradient(to right, rgba(74,222,128,0.22), transparent); }
.image-title {
    font-family:'Space Mono',monospace; font-size:0.67rem; letter-spacing:0.13em;
    text-transform:uppercase; color:#4ade80; opacity:0.6; margin-bottom:0.5rem; display:block;
}
.gradcam-grid-label {
    text-align:center; font-family:'Space Mono',monospace; font-size:0.65rem;
    letter-spacing:0.1em; text-transform:uppercase; color:#6b7280; margin-top:0.45rem;
    padding:4px 9px; background:rgba(255,255,255,0.022); border-radius:7px; border:1px solid rgba(255,255,255,0.05);
}
.treatment-box {
    background:rgba(22,163,74,0.045); border:1px solid rgba(22,163,74,0.16);
    border-left:3px solid #16a34a; border-radius:0 16px 16px 0;
    padding:1.5rem 1.7rem; margin:0.7rem 0; white-space:pre-wrap;
    font-family:'DM Sans',sans-serif; font-size:0.91rem; color:#bbf7d0; line-height:1.9;
}

/* ── REPORT DOWNLOAD CARD ── */
.report-download-wrap {
    background: rgba(5,46,22,0.45);
    border: 1px solid rgba(74,222,128,0.18);
    border-radius: 18px;
    padding: 1.4rem 1.7rem 1.5rem;
    margin: 2.4rem 0 0.5rem;
}
.report-download-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.15rem; font-weight: 700; color: #f0fdf4; margin-bottom: 0.25rem;
}
.report-download-sub {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.82rem; color: #6b7280; font-style: italic; margin-bottom: 1rem;
}
.report-includes { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-bottom: 1rem; }
.report-tag {
    background: rgba(74,222,128,0.08); border: 1px solid rgba(74,222,128,0.18);
    color: #86efac; font-family: 'Space Mono', monospace;
    font-size: 0.62rem; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 3px 10px; border-radius: 100px;
}

[data-testid="stNotification"] { border-radius: 12px !important; }
.stSuccess > div { background:rgba(22,163,74,0.09) !important; border:1px solid rgba(22,163,74,0.28) !important; border-radius:12px !important; color:#86efac !important; }
.stError > div   { background:rgba(239,68,68,0.08) !important;  border:1px solid rgba(239,68,68,0.26)  !important; border-radius:12px !important; }
.stWarning > div { background:rgba(251,191,36,0.07) !important; border:1px solid rgba(251,191,36,0.22) !important; border-radius:12px !important; }
[data-testid="stImage"] img { border-radius: 14px !important; display: block; }
.stSpinner > div { border-color: #16a34a transparent transparent transparent !important; }
h1, h2, h3 { font-family:'Playfair Display',serif !important; color:#f0fdf4 !important; }
p, li, .stText, .stMarkdown p { color:#9ca3af !important; font-family:'DM Sans',sans-serif !important; line-height:1.7 !important; }
.agricare-footer {
    text-align:center; padding:2.4rem 1rem 1.6rem; color:#2d3748;
    font-family:'Space Mono',monospace; font-size:0.66rem; letter-spacing:0.12em;
    text-transform:uppercase; border-top:1px solid rgba(255,255,255,0.042); margin-top:4rem;
}
.agricare-footer strong { color: #4ade80; }
@keyframes fadeSlideDown { from{opacity:0;transform:translateY(-15px);} to{opacity:1;transform:translateY(0);} }
@keyframes fadeSlideUp   { from{opacity:0;transform:translateY(15px);}  to{opacity:1;transform:translateY(0);} }
.fade-up { animation: fadeSlideUp 0.5s ease both; }
[data-testid="stHorizontalBlock"] { gap: 1.6rem !important; align-items: flex-start; }
@media (max-width:768px) {
    .hero-title { font-size:2.5rem !important; }
    .hero-sub   { font-size:0.93rem !important; }
    .stat-row   { gap:0.45rem; }
    .stat-pill  { font-size:0.68rem; padding:5px 10px; }
    .block-container { padding-left:0.9rem !important; padding-right:0.9rem !important; }
}
</style>
""", unsafe_allow_html=True)


_spacer, _lang_col = st.columns([6.5, 1.5])
with _lang_col:
    language = st.selectbox(
        "🌐 Language",
        ["English", "Telugu", "Hindi", "Tamil"],
        index=0,
        label_visibility="collapsed",
        key="lang_select"
    )

lang_codes = {"English": "en", "Telugu": "te", "Hindi": "hi", "Tamil": "ta"}

def translate_text(text):
    if language == "English":
        return text
    try:
        return GoogleTranslator(source="auto", target=lang_codes[language]).translate(text)
    except Exception:
        return text



st.markdown(f"""
<div class="agricare-hero">
    <div style="display:flex;justify-content:center;margin-bottom:0;">
        <div class="hero-badge">
            <span class="dot"></span>
            {translate_text("Smart Rice leaf Disease Diagnosis and Cure Recommendation")}
        </div>
    </div>
    <h1 class="hero-title">AGRI<span class="green">CARE</span></h1>
    <center><p class="hero-sub">{translate_text("Upload a paddy leaf image to detect disease, severity and treatment.")}</p>
    <div class="stat-row">
    </div>
    <div class="fancy-divider"><span>🌿</span></div>
</div>
""", unsafe_allow_html=True)



@st.cache_resource
def load_disease_model():
    return tf.keras.models.load_model("paddy_final_model")

@st.cache_resource
def load_rice_model():
    return tf.keras.models.load_model("rice_vs_nonrice_model1.h5")

model      = load_disease_model()
rice_model = load_rice_model()

class_names = [
    'bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight',
    'blast', 'brown_spot', 'dead_heart', 'downy_mildew', 'hispa', 'normal', 'tungro'
]



def _leaf_cv_check(img_path):
    bgr = cv2.imread(img_path)
    if bgr is None:
        return False, {}
    hsv   = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w  = hsv.shape[:2]
    total = h * w
    veg_mask = cv2.inRange(hsv, np.array([20, 30, 30]), np.array([85, 255, 255]))
    total_green = np.sum(veg_mask > 0) / total
    if total_green == 0:
        return False, {"total_green": 0, "largest_blob": 0, "concentration": 0}
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(veg_mask)
    if num_labels <= 1:
        return False, {"total_green": total_green, "largest_blob": 0, "concentration": 0}
    areas         = stats[1:, cv2.CC_STAT_AREA]
    largest       = int(np.max(areas))
    largest_ratio = largest / total
    concentration = largest / np.sum(veg_mask > 0)
    passed = (total_green >= 0.08) and (largest_ratio >= 0.05) and (concentration >= 0.40)
    return passed, {
        "total_green":   round(total_green,   3),
        "largest_blob":  round(largest_ratio, 3),
        "concentration": round(concentration, 3)
    }

def is_rice_leaf(img_path):
    passed_cv, _ = _leaf_cv_check(img_path)
    if not passed_cv:
        return False, 0.0
    img       = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    prob = float(rice_model.predict(img_array, verbose=0)[0][0])
    return prob >= 0.5, prob



def predict_disease(img_path):
    img       = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    preds     = model.predict(img_array, verbose=0)
    idx       = int(np.argmax(preds, axis=1)[0])
    return class_names[idx], float(np.max(preds))



def apply_green_mask(img_path):
    img      = cv2.imread(img_path)
    img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv      = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    green_mask = cv2.inRange(hsv, np.array([25, 40, 40]),  np.array([85, 255, 255]))
    blue_mask  = cv2.inRange(hsv, np.array([90, 50, 50]),  np.array([130, 255, 255]))
    dark_mask  = cv2.inRange(hsv, np.array([0,  0,  0]),   np.array([180, 50,  50]))
    exclude      = cv2.bitwise_or(blue_mask, dark_mask)
    refined_mask = cv2.bitwise_and(green_mask, cv2.bitwise_not(exclude))
    leaf_area    = cv2.bitwise_and(img_rgb, img_rgb, mask=refined_mask)
    return leaf_area, refined_mask, img_rgb

def estimate_severity(img_path):
    leaf_img, mask, original = apply_green_mask(img_path)
    if cv2.countNonZero(mask) == 0:
        return "Low", 0.0, original, leaf_img
    hsv_leaf     = cv2.cvtColor(leaf_img, cv2.COLOR_RGB2HSV)
    disease_mask = cv2.inRange(hsv_leaf, np.array([10, 50, 50]), np.array([30, 255, 255]))
    disease_mask = cv2.bitwise_and(disease_mask, disease_mask, mask=mask)
    disease_area    = cv2.countNonZero(disease_mask)
    total_leaf_area = cv2.countNonZero(mask)
    if total_leaf_area == 0:
        return "Low", 0.0, original, leaf_img
    pct      = round((disease_area / total_leaf_area) * 100, 2)
    severity = "Low" if pct < 30 else ("Medium" if pct < 60 else "High")
    return severity, pct, original, leaf_img


def get_last_conv_layer(mdl):
    for layer in reversed(mdl.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

LAST_CONV_LAYER = get_last_conv_layer(model)

def make_gradcam_heatmap(img_array, mdl, conv_layer_name):
    grad_model = tf.keras.models.Model(
        inputs  = mdl.inputs,
        outputs = [mdl.get_layer(conv_layer_name).output, mdl.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array, training=False)
        if isinstance(preds, list):
            preds = preds[0]
        pred_idx      = tf.argmax(preds[0])
        class_channel = preds[0][pred_idx]
    grads        = tape.gradient(class_channel, conv_out)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out     = conv_out[0]
    heatmap      = tf.reduce_sum(conv_out * pooled_grads, axis=-1)
    heatmap      = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def display_gradcam(img_path):
    img_pil   = image.load_img(img_path, target_size=(224, 224))
    img_arr   = image.img_to_array(img_pil)
    img_arr   = np.expand_dims(img_arr, axis=0)
    img_arr   = preprocess_input(img_arr)
    heatmap   = make_gradcam_heatmap(img_arr, model, LAST_CONV_LAYER)
    orig_bgr  = cv2.imread(img_path)
    orig_rgb  = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
    heatmap_resized = cv2.resize(heatmap, (orig_rgb.shape[1], orig_rgb.shape[0]))
    heatmap_u8  = np.uint8(255 * heatmap_resized)
    heatmap_col = cv2.applyColorMap(heatmap_u8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_col, cv2.COLOR_BGR2RGB)
    superimposed = cv2.addWeighted(orig_rgb, 0.6, heatmap_rgb, 0.4, 0)
    return orig_rgb, heatmap_rgb, superimposed



treatment_db = {
    "bacterial_leaf_blight":    {"medicine": "Streptocycline + Copper Oxychloride",  "low": "0.1 g + 1 g per liter",   "medium": "0.2 g + 1.5 g per liter", "high": "0.3 g + 2 g per liter"},
    "bacterial_leaf_streak":    {"medicine": "Kasugamycin + Copper Oxychloride",     "low": "0.5 ml + 1 g per liter",  "medium": "1 ml + 1.5 g per liter",  "high": "1.5 ml + 2 g per liter"},
    "bacterial_panicle_blight": {"medicine": "Streptocycline + Carbendazim",         "low": "0.1 g + 0.5 g per liter", "medium": "0.2 g + 1 g per liter",   "high": "0.3 g + 1.5 g per liter"},
    "blast":                    {"medicine": "Tricyclazole 75% WP",                  "low": "0.4 g per liter",         "medium": "0.6 g per liter",         "high": "0.8 g per liter"},
    "brown_spot":               {"medicine": "Mancozeb 75% WP",                     "low": "1 g per liter",           "medium": "1.5 g per liter",         "high": "2 g per liter"},
    "dead_heart":               {"medicine": "Chlorantraniliprole 18.5% SC",         "low": "0.3 ml per liter",        "medium": "0.5 ml per liter",        "high": "0.75 ml per liter"},
    "downy_mildew":             {"medicine": "Metalaxyl + Mancozeb",                 "low": "1 g per liter",           "medium": "1.5 g per liter",         "high": "2 g per liter"},
    "hispa":                    {"medicine": "Chlorpyrifos 20% EC",                  "low": "1 ml per liter",          "medium": "2 ml per liter",          "high": "2.5 ml per liter"},
    "tungro":                   {"medicine": "Imidacloprid (Vector Control)",        "low": "0.25 ml per liter",       "medium": "0.5 ml per liter",        "high": "0.75 ml per liter"},
    "normal":                   {"medicine": "No treatment required",                "low": "N/A",                     "medium": "N/A",                     "high": "N/A"},
}

def recommend_treatment(disease, severity):
    """Returns the treatment text string for the UI display box (unchanged)."""
    if disease == "normal":
        return """Healthy Crop Condition

The uploaded leaf does not show any visible symptoms of disease.
Maintain proper crop management practices to ensure continued healthy growth.

Recommended Practices:
\u2022 Maintain proper irrigation schedule
\u2022 Apply balanced fertilizers based on soil test recommendations
\u2022 Monitor the crop regularly for early disease detection
\u2022 Ensure proper field drainage and spacing
\u2022 Follow integrated pest management (IPM) practices"""
    d        = treatment_db[disease]
    medicine = d["medicine"]
    dosage   = d[severity.lower()]
    return f"""Pesticide / Fungicide:
{medicine}

Recommended Dosage:
{dosage}

Application Guidelines:
\u2022 Prepare the spray solution according to the recommended dosage.
\u2022 Ensure uniform coverage on both upper and lower leaf surfaces.
\u2022 Apply the spray preferably during early morning or late evening.
\u2022 Avoid spraying during strong wind or rainfall conditions.
\u2022 Repeat the application at an interval of 7\u201310 days if symptoms persist.

Crop Protection Advisory:
\u2022 Remove severely infected leaves or plant parts if possible.
\u2022 Maintain proper field sanitation to reduce pathogen spread.
\u2022 Avoid excessive nitrogen fertilization as it may increase disease susceptibility.
\u2022 Monitor crop conditions regularly after treatment.

Note: Always follow local agricultural department guidelines and safety
precautions while handling pesticides."""



def _arr_to_rl_img(arr_rgb, max_w_mm, max_h_mm):
    """Convert a numpy RGB uint8 array → ReportLab Image, scaled to fit."""
    pil   = Image.fromarray(arr_rgb.astype(np.uint8))
    buf   = io.BytesIO()
    pil.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    ih, iw = arr_rgb.shape[:2]
    scale  = min((max_w_mm * mm) / iw, (max_h_mm * mm) / ih)
    return RLImage(buf, width=iw * scale, height=ih * scale)


def _pil_to_rl_img(pil_img, max_w_mm, max_h_mm):
    return _arr_to_rl_img(np.array(pil_img.convert("RGB")), max_w_mm, max_h_mm)


def generate_pdf_report(img_bytes, disease, disease_conf,
                        severity, severity_pct,
                        gradcam_orig, gradcam_heat, gradcam_cam):
    """Build a styled dark-themed A4 PDF report. Returns raw bytes."""

    # ── colours ──
    C_BG      = colors.HexColor("#0b150b")
    C_CARD    = colors.HexColor("#101e10")
    C_CARD2   = colors.HexColor("#0d180d")
    C_GREEN   = colors.HexColor("#16a34a")
    C_GREEN_L = colors.HexColor("#4ade80")
    C_TEXT    = colors.HexColor("#d1fae5")
    C_MUTED   = colors.HexColor("#6b7280")
    C_WHITE   = colors.HexColor("#f0fdf4")
    C_BORDER  = colors.HexColor("#1a3320")
    C_AMBER   = colors.HexColor("#fbbf24")
    C_RED     = colors.HexColor("#f87171")

    sev_color = {"Low": C_GREEN_L, "Medium": C_AMBER, "High": C_RED}.get(severity, C_GREEN_L)

    buf    = io.BytesIO()
    PAGE_W, PAGE_H = A4
    M      = 16 * mm
    BODY   = PAGE_W - 2 * M

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=M, rightMargin=M,
        topMargin=M, bottomMargin=M,
    )

    def draw_bg(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(C_BG)
        canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        canvas.setFillColor(colors.HexColor("#0a2a0a"))
        canvas.rect(0, PAGE_H - 72 * mm, PAGE_W, 72 * mm, fill=1, stroke=0)
        canvas.restoreState()

    def ps(name, size=9, bold=False, italic=False, color=None,
           align=TA_LEFT, sb=0, sa=0, leading=None, indent=0):
        fn = "Helvetica"
        if bold and italic: fn = "Helvetica-BoldOblique"
        elif bold:          fn = "Helvetica-Bold"
        elif italic:        fn = "Helvetica-Oblique"
        return ParagraphStyle(
            name, fontName=fn, fontSize=size,
            textColor=color or C_TEXT,
            alignment=align, spaceBefore=sb, spaceAfter=sa,
            leading=leading or (size * 1.45),
            leftIndent=indent,
        )

    story = []
    ts    = datetime.datetime.now().strftime("%d %B %Y  ·  %H:%M")

    hdr = Table([[
        Paragraph("AGRICARE", ps("hb", size=22, bold=True, color=C_WHITE)),
        Paragraph(f"Generated: {ts}", ps("hts", size=7, color=C_MUTED, align=TA_LEFT)),
    ]], colWidths=[BODY * 0.60, BODY * 0.40])
    hdr.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ("BACKGROUND",    (0,0), (-1,-1), C_CARD),
        ("BOX",           (0,0), (-1,-1), 1.5, C_GREEN),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("TOPPADDING",    (0,0), (-1,-1), 10),
        ("BOTTOMPADDING", (0,0), (-1,-1), 10),
    ]))
    story.append(hdr)
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=3*mm))

    story.append(Paragraph("Uploaded Leaf Image",
                            ps("sec0", size=11, bold=True, color=C_GREEN_L, sb=2*mm, sa=2*mm)))
    pil_upload = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    upload_buf = io.BytesIO()
    pil_upload.save(upload_buf, format="JPEG", quality=90)
    upload_buf.seek(0)
    ih_u, iw_u = pil_upload.height, pil_upload.width
    scale_u = min((BODY * 0.55) / iw_u, (72 * mm) / ih_u)
    img_rl = RLImage(upload_buf, width=iw_u * scale_u, height=ih_u * scale_u)
    img_tbl = Table([[img_rl]], colWidths=[BODY])
    img_tbl.setStyle(TableStyle([
        ("ALIGN",         (0,0), (-1,-1), "CENTER"),
        ("BACKGROUND",    (0,0), (-1,-1), C_CARD2),
        ("BOX",           (0,0), (-1,-1), 1, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,-1), 8),
        ("LEFTPADDING",   (0,0), (-1,-1), 8),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 4*mm))

    story.append(Paragraph("Diagnosis Summary",
                            ps("sec1", size=11, bold=True, color=C_GREEN_L, sb=2*mm, sa=2*mm)))

    fmt_disease = disease.replace("_", " ").title() if disease != "normal" else "Healthy"
    
    sev_str     = f"{severity}  ({severity_pct}% affected)" if disease != "normal" else "N / A"

    cw = BODY / 3
    diag = Table([
        [Paragraph("DETECTED DISEASE",  ps("dl1", size=7, bold=True, color=C_MUTED)),
         
         Paragraph("SEVERITY LEVEL",    ps("dl3", size=7, bold=True, color=C_MUTED))],
        [Paragraph(fmt_disease,  ps("dv1", size=14, bold=True, color=C_WHITE)),
         
         Paragraph(sev_str,      ps("dv3", size=12, bold=True, color=sev_color))],
    ], colWidths=[cw, cw, cw])
    diag.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_CARD2),
        ("ROWBACKGROUND", (0,0), (-1, 0), colors.HexColor("#0a160a")),
        ("BOX",           (0,0), (-1,-1), 1, C_BORDER),
        ("INNERGRID",     (0,0), (-1,-1), 0.4, C_BORDER),
        ("TOPPADDING",    (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,-1), 7),
        ("LEFTPADDING",   (0,0), (-1,-1), 9),
        ("RIGHTPADDING",  (0,0), (-1,-1), 9),
        ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(diag)
    story.append(Spacer(1, 4*mm))

    
    story.append(Paragraph("Treatment Recommendation",
                            ps("sec3", size=11, bold=True, color=C_GREEN_L, sb=2*mm, sa=2*mm)))

    if disease == "normal":
        advisory_lines = [
            "Maintain proper irrigation schedule.",
            "Apply balanced fertilizers based on soil test recommendations.",
            "Monitor the crop regularly for early disease detection.",
            "Ensure proper field drainage and adequate plant spacing.",
            "Follow integrated pest management (IPM) practices.",
        ]
        cell = [
            Paragraph("Healthy Crop — No Treatment Required",
                      ps("ht", size=11, bold=True, color=C_GREEN_L, sa=3)),
            Paragraph(
                "The uploaded leaf shows no visible symptoms of disease. "
                "Continue good agronomic practices to maintain crop health.",
                ps("hb", size=9, color=C_TEXT, sa=4)
            ),
            Paragraph("Recommended Practices:", ps("rh", size=9, bold=True, color=C_MUTED, sa=2)),
        ]
        for ln in advisory_lines:
            cell.append(Paragraph(f"  •   {ln}", ps("rb", size=9, color=C_TEXT, sa=1, indent=6)))
        htbl = Table([[ cell ]], colWidths=[BODY])
        htbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_CARD2),
            ("BOX",           (0,0), (-1,-1), 1, C_GREEN),
            ("LEFTPADDING",   (0,0), (-1,-1), 11),
            ("RIGHTPADDING",  (0,0), (-1,-1), 11),
            ("TOPPADDING",    (0,0), (-1,-1), 10),
            ("BOTTOMPADDING", (0,0), (-1,-1), 10),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ]))
        story.append(htbl)

    else:
        d        = treatment_db[disease]
        medicine = d["medicine"]
        dosage   = d[severity.lower()]

        med = Table([
            [Paragraph("PESTICIDE / FUNGICIDE", ps("ml1", size=7, bold=True, color=C_MUTED)),
             Paragraph("RECOMMENDED DOSAGE",    ps("ml2", size=7, bold=True, color=C_MUTED))],
            [Paragraph(medicine, ps("mv1", size=12, bold=True, color=C_GREEN_L)),
             Paragraph(dosage,   ps("mv2", size=12, bold=True, color=C_WHITE))],
        ], colWidths=[BODY * 0.58, BODY * 0.42])
        med.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_CARD2),
            ("BOX",           (0,0), (-1,-1), 1.2, C_GREEN),
            ("INNERGRID",     (0,0), (-1,-1), 0.4, C_BORDER),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
            ("RIGHTPADDING",  (0,0), (-1,-1), 10),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
        ]))
        story.append(med)
        story.append(Spacer(1, 4*mm))

        guide_items = [
            "Prepare the spray solution according to the recommended dosage.",
            "Ensure uniform coverage on both upper and lower leaf surfaces.",
            "Apply the spray preferably during early morning or late evening.",
            "Avoid spraying during strong wind or rainfall conditions.",
            "Repeat application at an interval of 7-10 days if symptoms persist.",
        ]
        advisory_items = [
            "Remove severely infected leaves or plant parts if possible.",
            "Maintain proper field sanitation to reduce pathogen spread.",
            "Avoid excessive nitrogen fertilization to limit susceptibility.",
            "Monitor crop conditions regularly after treatment.",
            "Consult your local agricultural officer if symptoms worsen.",
        ]

        def col_block(title, items, tc):
            out = [Paragraph(title, ps(f"ct{title[:2]}", size=8, bold=True, color=tc, sa=3))]
            for it in items:
                out.append(Paragraph(f"  •  {it}",
                                     ps(f"ci{title[:2]}", size=8, color=C_TEXT, sa=1.5, leading=11, indent=4)))
            return out

        two = Table([
            [col_block("Application Guidelines",  guide_items,    C_GREEN_L),
             col_block("Crop Protection Advisory", advisory_items, C_AMBER)]
        ], colWidths=[BODY/2 - 2*mm, BODY/2 - 2*mm])
        two.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_CARD2),
            ("BOX",           (0,0), (-1,-1), 0.5, C_BORDER),
            ("INNERGRID",     (0,0), (-1,-1), 0.5, C_BORDER),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 9),
            ("RIGHTPADDING",  (0,0), (-1,-1), 9),
        ]))
        story.append(two)
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            "Note: Always follow local agricultural department guidelines and "
            "safety precautions while handling pesticides.",
            ps("note", size=7.5, italic=True, color=C_MUTED, sa=2)
        ))

    doc.build(story, onFirstPage=draw_bg, onLaterPages=draw_bg)
    return buf.getvalue()


ss = st.session_state

st.markdown(f'<span class="step-label">{translate_text("Step 1 — Upload Paddy Leaf Image")}</span>',
            unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    translate_text("Drop your paddy leaf image here · JPG  PNG  JPEG"),
    type=["jpg", "png", "jpeg"],
    label_visibility="visible"
)

if uploaded_file is not None and uploaded_file.file_id != ss.last_file_id:

    raw_bytes = uploaded_file.read()
    ss.rice_ok = ss.rice_conf = None
    ss.disease = ss.disease_conf = None
    ss.severity = ss.severity_pct = None
    ss.gradcam_orig = ss.gradcam_heat = ss.gradcam_cam = None
    ss.img_bytes    = raw_bytes
    ss.last_file_id = uploaded_file.file_id

    pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    pil_img.save(ss.temp_path)

    with st.spinner("🔍 " + translate_text("Checking if this is a rice leaf...")):
        rice_ok, rice_conf = is_rice_leaf(ss.temp_path)
    ss.rice_ok, ss.rice_conf = rice_ok, rice_conf

    if rice_ok:
        with st.spinner("🔬 " + translate_text("AgriCare is analysing the leaf...")):
            disease, disease_conf = predict_disease(ss.temp_path)
        ss.disease, ss.disease_conf = disease, disease_conf

        if disease != "normal":
            severity, pct, _, _ = estimate_severity(ss.temp_path)
            ss.severity, ss.severity_pct = severity, pct
            try:
                orig, heat, cam = display_gradcam(ss.temp_path)
                ss.gradcam_orig, ss.gradcam_heat, ss.gradcam_cam = orig, heat, cam
            except Exception as e:
                st.warning(f"Grad-CAM could not be generated: {e}")

if ss.img_bytes is not None:

    display_img = Image.open(io.BytesIO(ss.img_bytes)).convert("RGB")
    st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown(f'<span class="image-title">{translate_text("Uploaded Leaf Image")}</span>',
                    unsafe_allow_html=True)
        st.image(display_img, use_container_width=True)

    if ss.rice_ok is False:
        st.error("⚠️  " + translate_text(
            "This is NOT a rice leaf. Please upload a clear close-up paddy leaf image."))

    elif ss.rice_ok is True:
        st.success("🌾  " + translate_text(
            f"Rice leaf confirmed ({ss.rice_conf*100:.1f}% confidence). Running disease analysis..."))

        if ss.disease == "normal":
            with col2:
                st.markdown(f"""
                <div class="result-card fade-up">
                    <div class="result-card-label">{translate_text("Detection Result")}</div>
                    <div class="result-card-value">✅ {translate_text("Healthy Leaf")}</div>
                    <div class="result-card-sub">{translate_text("No disease detected")}</div>
                    <div class="conf-bar-wrap">
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="section-header">🌱 {translate_text("Crop Advisory")}</div>',
                        unsafe_allow_html=True)
            st.markdown(
                f'<div class="treatment-box">'
                f'{translate_text(recommend_treatment("normal", "low"))}'
                f'</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="report-download-wrap">
                <div class="report-download-title">📄 {translate_text("Download Diagnostic Report")}</div>
                <div class="report-download-sub">{translate_text("Export a complete PDF with your leaf image and crop advisory.")}</div>
                <div class="report-includes">
                    <span class="report-tag">{translate_text("Leaf Image")}</span>
                    <span class="report-tag">{translate_text("Diagnosis Result")}</span>
                    <span class="report-tag">{translate_text("Treatment Recommendation")}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            _pdf   = generate_pdf_report(
                ss.img_bytes, ss.disease, ss.disease_conf,
                "N/A", 0.0, None, None, None
            )
            _fname = f"agricare_healthy_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label="⬇   Download PDF Report",
                data=_pdf, file_name=_fname,
                mime="application/pdf", key="dl_healthy"
            )

        elif ss.disease is not None:
            sev_class   = f"severity-{ss.severity.lower()}"
            fmt_disease = ss.disease.replace("_", " ").title()

            with col2:
                st.markdown(f"""
                <div class="result-card fade-up">
                    <div class="result-card-label">{translate_text("Detected Disease")}</div>
                    <div class="result-card-value">{translate_text(fmt_disease)}</div>
                    <div class="result-card-sub">{translate_text("Paddy leaf disease identified")}</div>
                    <div style="margin-top:1rem">
                        <div class="result-card-label">{translate_text("Severity Level")}</div>
                        <span class="severity-badge {sev_class}">
                            {translate_text(ss.severity)} &mdash; {ss.severity_pct}% {translate_text("affected")}
                        </span>
                    </div>
                    
                </div>""", unsafe_allow_html=True)

            st.markdown(
                f'<div class="section-header">🔬 {translate_text("Disease Infected Area (Grad-CAM)")}</div>',
                unsafe_allow_html=True)
            if ss.gradcam_orig is not None:
                c1, c2, c3 = st.columns(3, gap="medium")
                with c1:
                    st.image(ss.gradcam_orig, use_container_width=True)
                    st.markdown(f'<div class="gradcam-grid-label">{translate_text("Original")}</div>',
                                unsafe_allow_html=True)
                with c2:
                    st.image(ss.gradcam_heat, use_container_width=True)
                    st.markdown(f'<div class="gradcam-grid-label">{translate_text("Heatmap")}</div>',
                                unsafe_allow_html=True)
                with c3:
                    st.image(ss.gradcam_cam, use_container_width=True)
                    st.markdown(f'<div class="gradcam-grid-label">{translate_text("Model Attention")}</div>',
                                unsafe_allow_html=True)

            st.markdown(
                f'<div class="section-header">💊 {translate_text("Treatment Recommendation")}</div>',
                unsafe_allow_html=True)
            st.markdown(
                f'<div class="treatment-box">'
                f'{translate_text(recommend_treatment(ss.disease, ss.severity))}'
                f'</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div class="report-download-wrap">
                <div class="report-download-title">📄 {translate_text("Download Diagnostic Report")}</div>
                <div class="report-download-sub">{translate_text("Full PDF with images, Grad-CAM, disease diagnosis, severity and complete treatment guidelines.")}</div>
                <div class="report-includes">
                    <span class="report-tag">{translate_text("Leaf Image")}</span>
                    <span class="report-tag">{translate_text("Disease Name")}</span>
                    <span class="report-tag">{translate_text("Severity Level")}</span>
                    <span class="report-tag">{translate_text("Pesticide + Dosage")}</span>
                    <span class="report-tag">{translate_text("Application Guidelines")}</span>
                    <span class="report-tag">{translate_text("Crop Advisory")}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            _pdf   = generate_pdf_report(
                ss.img_bytes, ss.disease, ss.disease_conf,
                ss.severity, ss.severity_pct,
                ss.gradcam_orig, ss.gradcam_heat, ss.gradcam_cam
            )
            _fname = f"agricare_{ss.disease}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            st.download_button(
                label="⬇   Download PDF Report",
                data=_pdf, file_name=_fname,
                mime="application/pdf", key="dl_disease"
            )


st.markdown(f"""
<div class="agricare-footer">
    <strong>AGRICARE</strong> &nbsp;·&nbsp;
</div>
""", unsafe_allow_html=True)