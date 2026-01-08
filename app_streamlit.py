import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os
import uuid
import datetime
import pandas as pd
import cloudinary
import cloudinary.uploader
import gspread
import requests
import urllib.request
import warnings
from io import BytesIO
from PIL import Image
from google.oauth2.service_account import Credentials
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER

warnings.filterwarnings("ignore")

# =================================================================
# 1. CẤU HÌNH HỆ THỐNG & ĐÁM MÂY
# =================================================================

# ID file từ Google Drive bạn cung cấp
FILES_DRIVE = {
    "unet_best.pth": "1JrB9BpL2kacwau3MPqCo6Rq1cjFlkLLJ",
    "deeplabv3plus_best.pth": "1UaRDnAMsPNGiB4_OeC2OfHlXIfPUfOOB",
    "efficientnet_attention_best.pth": "1Q7JHPqnzPvb5fV-VzlhMC1jYlKJRuwKF",
    "hybrid_best.pth": "1SKaJBRYUmDJV9qAyGcUY78zCawCncaHJ"
}

# Cấu hình Cloudinary
cloudinary.config( 
    cloud_name = "dq7whcy51", 
    api_key = "677482925994952", 
    api_secret = "1WYJ_fYnUu_nNhgDqLfRCVSAr1Q" 
)

def get_gsheets_client():
    """Kết nối Google Sheets bằng credentials trong st.secrets"""
    try:
        creds_dict = st.secrets["gsheets"]["service_account"]
        scopes = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_url(st.secrets["gsheets"]["spreadsheet"])
        return spreadsheet.worksheet("Sheet1")
    except Exception as e:
        st.error(f"Lỗi Google Sheets: {e}")
        st.stop()

# =================================================================
# 2. HÀM TẢI FILE (FONT & MODELS)
# =================================================================

@st.cache_resource
def setup_assets():
    """Tải font và các file trọng số mô hình"""
    # 1. Setup Font Tiếng Việt
    os.makedirs("fonts", exist_ok=True)
    font_reg = os.path.join("fonts", "NotoSans-Regular.ttf")
    font_bold = os.path.join("fonts", "NotoSans-Bold.ttf")
    
    if not os.path.exists(font_reg):
        urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf", font_reg)
    if not os.path.exists(font_bold):
        urllib.request.urlretrieve("https://github.com/google/fonts/raw/main/ofl/notosans/NotoSans%5Bwdth%2Cwght%5D.ttf", font_bold)

    # 2. Tải Models từ Drive
    for filename, file_id in FILES_DRIVE.items():
        if not os.path.exists(filename):
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            with st.spinner(f"Đang tải mô hình {filename}..."):
                urllib.request.urlretrieve(url, filename)
    
    return font_reg, font_bold

FONT_PATH, FONT_BOLD_PATH = setup_assets()

# =================================================================
# 3. ĐỊNH NGHĨA KIẾN TRÚC MÔ HÌNH
# =================================================================

class HybridSegmentation(nn.Module):
    def __init__(self, unet, deeplab):
        super().__init__()
        self.unet, self.deeplab = unet, deeplab
    def forward(self, x):
        with torch.no_grad():
            return torch.max(torch.sigmoid(self.unet(x)), torch.sigmoid(self.deeplab(x)))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x * self.ca(x)
        return x * self.sa(torch.cat([torch.mean(x,1,True), torch.max(x,1,True)[0]], 1))

class EfficientNetWithAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.attention = CBAM(self.feature_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        x = self.backbone.forward_features(x)
        return self.classifier(self.attention(x))

# =================================================================
# 4. TẢI MÔ HÌNH VÀ XỬ LÝ AI
# =================================================================

@st.cache_resource
def load_all_models():
    import segmentation_models_pytorch as smp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Segmentation
    u_net = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1).to(device)
    u_net.load_state_dict(torch.load("unet_best.pth", map_location=device, weights_only=False)["model_state_dict"])
    
    d_lab = smp.DeepLabV3Plus(encoder_name="resnet50", in_channels=3, classes=1).to(device)
    d_lab.load_state_dict(torch.load("deeplabv3plus_best.pth", map_location=device, weights_only=False)["model_state_dict"])
    
    hybrid = HybridSegmentation(u_net, d_lab).eval()

    # Load Classification
    with open("06_classification_complete.json", "r") as f: cls_ckpt = json.load(f)
    num_classes = cls_ckpt["config"]["num_classes"]
    cls_model = EfficientNetWithAttention(num_classes).to(device)
    state = torch.load("efficientnet_attention_best.pth", map_location=device, weights_only=False)
    cls_model.load_state_dict(state['model_state_dict'])
    cls_model.eval()
    
    idx_to_class = {v: k for k, v in (state.get("class_to_idx") or cls_ckpt.get("class_to_idx")).items()}
    return hybrid, cls_model, idx_to_class, device

hybrid, cls_model, idx_to_class, device = load_all_models()

def run_inference(image, patient_name, age, gender, note):
    record_id = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. AI Process
    img_input = cv2.resize(image, (256, 256)).astype(np.float32)/255.0
    img_input = (img_input - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(img_input).permute(2, 0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        mask = hybrid(tensor).squeeze().cpu().numpy()
    
    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_vis = (cv2.cvtColor((mask_resized > 0.5).astype(np.uint8)*255, cv2.COLOR_GRAY2BGR))
    overlay = cv2.addWeighted(image, 0.7, cv2.cvtColor(cv2.applyColorMap(np.uint8(255*mask_resized), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB), 0.3, 0)

    # 2. Classification
    ys, xs = np.where(mask_resized > 0.5)
    roi = cv2.resize(image[ys.min():ys.max(), xs.min():xs.max()], (224,224)) if len(xs)>0 else cv2.resize(image, (224,224))
    roi_t = torch.from_numpy((roi.astype(np.float32)/255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).permute(2,0,1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        probs = torch.softmax(cls_model(roi_t), 1).cpu().numpy()[0]
    label, conf = idx_to_class[np.argmax(probs)], probs[np.argmax(probs)]

    # 3. Cloud Sync
    def up_cv2(img, tag):
        _, buf = cv2.imencode('.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        return cloudinary.uploader.upload(buf.tobytes(), folder="skin_app", public_id=f"{record_id}_{tag}")['secure_url']

    urls = [up_cv2(image, "orig"), up_cv2(overlay, "ov"), up_cv2(mask_vis, "mask")]

    # 4. Save Sheets
    data = {"record_id": record_id, "timestamp": timestamp, "name": patient_name, "age": int(age), 
            "gender": gender, "note": note, "diagnosis": label, "confidence": float(conf),
            "url_orig": urls[0], "url_ov": urls[1], "url_mask": urls[2]}
    get_gsheets_client().append_row(list(data.values()))

    info = f"**ID:** {record_id}  \n**Bệnh nhân:** {patient_name}  \n**Chẩn đoán:** {label} ({conf*100:.2f}%)"
    return overlay, mask_vis, info, data

# =================================================================
# 5. XUẤT BÁO CÁO PDF
# =================================================================

def generate_pdf_report(data, overlay_img, mask_img):
    pdfmetrics.registerFont(TTFont('VietFont', FONT_PATH))
    pdfmetrics.registerFont(TTFont('VietFont-Bold', FONT_BOLD_PATH))
    
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4)
    styles = getSampleStyleSheet()
    
    elements = [
        Paragraph("BÁO CÁO CHẨN ĐOÁN DA LIỄU", ParagraphStyle('T', fontName='VietFont-Bold', fontSize=18, alignment=TA_CENTER)),
        Spacer(1, 12),
        Table([["ID Bệnh án:", data['record_id']], ["Bệnh nhân:", data['name']], ["Chẩn đoán:", data['diagnosis']], ["Độ tin cậy:", f"{data['confidence']*100:.2f}%"]], 
              colWidths=[4*cm, 10*cm], style=TableStyle([('FONTNAME', (0,0), (-1,-1), 'VietFont'), ('GRID', (0,0), (-1,-1), 0.5, colors.grey)])),
        Spacer(1, 12)
    ]
    
    def add_img(img_np, cap):
        elements.append(Paragraph(cap, styles['Normal']))
        img_b = BytesIO()
        Image.fromarray(img_np).save(img_b, format='PNG')
        elements.append(RLImage(img_b, width=10*cm, height=10*cm))
        
    add_img(overlay_img, "Ảnh Phân Vùng:")
    add_img(mask_img, "Ảnh Mask:")
    doc.build(elements)
    buf.seek(0)
    return buf

# =================================================================
# 6. GIAO DIỆN STREAMLIT
# =================================================================

st.set_page_config(page_title="AI Dermatology", layout="wide")
st.title("🩺 Hệ thống Chẩn đoán bệnh da liễu AI")

tabs = st.tabs(["Chẩn đoán mới", "Tra cứu bệnh án"])

with tabs[0]:
    up = st.file_uploader("Tải ảnh", type=["jpg", "png", "jpeg"])
    name = st.text_input("Tên bệnh nhân")
    age = st.number_input("Tuổi", 0, 120, 25)
    gen = st.radio("Giới tính", ["Nam", "Nữ"], horizontal=True)
    note = st.text_area("Ghi chú")
    
    if st.button("Chẩn đoán"):
        if up and name:
            img = np.array(Image.open(up).convert("RGB"))
            with st.spinner("AI đang xử lý..."):
                ov, mk, info, res_data = run_inference(img, name, age, gen, note)
                st.image(ov, width=None, use_container_width=True)
                st.info(info)
                st.download_button("📥 Tải báo cáo PDF", generate_pdf_report(res_data, ov, mk), f"BA_{res_data['record_id']}.pdf")
        else: st.warning("Vui lòng điền đủ thông tin!")

with tabs[1]:
    search = st.text_input("Tìm tên bệnh nhân")
    if st.button("Tìm kiếm"):
        rows = get_gsheets_client().get_all_records()
        df = pd.DataFrame(rows)
        if search: df = df[df['name'].str.contains(search, case=False, na=False)]
        st.dataframe(df[["record_id", "timestamp", "name", "diagnosis"]].tail(10))
        
        sid = st.selectbox("Xem chi tiết ID", df['record_id'].tolist())
        if sid:
            r = df[df['record_id'] == sid].iloc[0]
            st.image(r['url_ov'], use_container_width=True)
            st.write(f"**Ghi chú:** {r['note']}")