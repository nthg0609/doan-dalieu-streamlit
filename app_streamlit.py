import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os
import uuid
import datetime
from PIL import Image
import pandas as pd
from io import BytesIO
import requests
import warnings
warnings.filterwarnings("ignore")

import urllib.request

def download_if_missing(url, filename):
    if not os.path.exists(filename):
        st.info(f"Đang tải {filename}...")
        urllib.request. urlretrieve(url, filename)
        st.success(f"✅ Đã tải {filename}")

# Tải file DeepLab nếu chưa có
download_if_missing(
    "https://huggingface.co/spaces/nthg0609/DoAn_DaLieu/resolve/main/deeplabv3plus_best.pth",
    "deeplabv3plus_best.pth"  # ← THÊM DÒNG NÀY
)
# ==== Google Sheets Setup ====
import gspread
from google.oauth2.service_account import Credentials

def get_gsheets_client():
    """Kết nối Google Sheets bằng gspread"""
    try:
        # Lấy credentials từ st.secrets
        credentials_dict = st.secrets["gsheets"]["service_account"]
        
        # Tạo credentials object
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive"
        ]
        
        credentials = Credentials.from_service_account_info(
            credentials_dict,
            scopes=scopes
        )
        
        # Kết nối gspread
        client = gspread.authorize(credentials)
        
        # Mở spreadsheet
        spreadsheet_url = st.secrets["gsheets"]["spreadsheet"]
        spreadsheet = client.open_by_url(spreadsheet_url)
        worksheet = spreadsheet.worksheet("Sheet1")
        
        return worksheet
    
    except Exception as e:
        st.error(f"❌ Lỗi kết nối Google Sheets: {e}")
        st.stop()

# ==== SAFE FILENAME ====
def safe_str(s):
    return "".join(c for c in s if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')

import cloudinary
import cloudinary.uploader

# Cấu hình Cloudinary
cloudinary.config( 
    cloud_name = "dq7whcy51", 
    api_key = "677482925994952", 
    api_secret = "1WYJ_fYnUu_nNhgDqLfRCVSAr1Q" 
)

def upload_to_cloud(image_path):
    """Upload ảnh lên Cloudinary"""
    response = cloudinary.uploader.upload(image_path)
    return response['secure_url']

def save_to_gsheets(data_dict):
    """Lưu dữ liệu vào Google Sheets"""
    try: 
        worksheet = get_gsheets_client()
        
        # Lấy dữ liệu hiện tại
        existing_data = worksheet.get_all_records()
        
        # Nếu sheet trống, thêm header
        if not existing_data: 
            headers = list(data_dict.keys())
            worksheet.append_row(headers)
        
        # Thêm dòng mới
        new_row = list(data_dict.values())
        worksheet.append_row(new_row)
        
        st.success("✅ Đã lưu vào Google Sheets")
    
    except Exception as e: 
        st.error(f"Lỗi khi lưu vào Google Sheets: {e}")

# =================================================================
# 1. ĐỊNH NGHĨA CÁC LỚP MÔ HÌNH
# =================================================================
class HybridSegmentation(nn.Module):
    def __init__(self, unet, deeplab):
        super().__init__()
        self.unet = unet
        self. deeplab = deeplab
    
    def forward(self, x):
        with torch.no_grad():
            pred_unet = torch.sigmoid(self.unet(x))
            pred_dl = torch.sigmoid(self.deeplab(x))
            return torch.max(pred_unet, pred_dl)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn. AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn. Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn. Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()
    
    def forward(self, x):
        x = x * self.channel_att(x)
        return x * self.spatial_att(x)

class EfficientNetWithAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.attention = CBAM(self.feature_dim, reduction=16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn. Dropout(0.3), 
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True), 
            nn.Dropout(0.3), 
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone. forward_features(x)
        features = self.attention(features)
        return self.classifier(self.global_pool(features).flatten(1))

# =================================================================
# 2. HÀM TẢI MÔ HÌNH
# =================================================================
@st.cache_resource
def load_all_models():
    try:
        import segmentation_models_pytorch as smp
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Đọc cấu hình JSON
        with open("02_unet_complete.json", "r") as f:  
            unet_ckpt = json.load(f)
        with open("03_deeplabv3plus_complete.json", "r") as f:  
            deeplab_ckpt = json.load(f)
        with open("06_classification_complete.json", "r") as f:  
            cls_ckpt = json.load(f)

        # 2. Tên file weights
        unet_weight = "unet_best.pth"
        deeplab_weight = "deeplabv3plus_best.pth"
        cls_weight = "efficientnet_attention_best.pth"

        # Kiểm tra tồn tại
        for f in [unet_weight, deeplab_weight, cls_weight]:  
            if not os.path.exists(f):
                raise FileNotFoundError(f"Không tìm thấy file: {f}")

        # 3. Load UNet - THÊM weights_only=False
        unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        unet.load_state_dict(
            torch.load(unet_weight, map_location=device, weights_only=False)["model_state_dict"]
        )
        
        # 4. Load DeepLabV3+ - THÊM weights_only=False
        deeplab = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1)
        deeplab.load_state_dict(
            torch. load(deeplab_weight, map_location=device, weights_only=False)["model_state_dict"]
        )
        
        hybrid_model = HybridSegmentation(unet, deeplab).eval().to(device)

        # 5. Load Classification - THÊM weights_only=False
        num_classes = cls_ckpt["config"]["num_classes"]
        cls_model = EfficientNetWithAttention(num_classes=num_classes)
        state = torch.load(cls_weight, map_location=device, weights_only=False)
        cls_model.load_state_dict(state['model_state_dict'])
        cls_model = cls_model.eval().to(device)
        
        # Mapping nhãn
        class_to_idx = state. get("class_to_idx") or cls_ckpt. get("class_to_idx")
        idx_to_class = {v:  k for k, v in class_to_idx.items()}

        return hybrid_model, cls_model, idx_to_class, device

    except Exception as e:  
        st.error(f"LỖI KHI TẢI MÔ HÌNH: {str(e)}")
        st.write("Các file hiện có trên server:", os.listdir("."))
        st.stop()

# Gọi load models
hybrid, cls_model, idx_to_class, device = load_all_models()

# =================================================================
# 3. HÀM TIỀN XỬ LÝ
# =================================================================
def preprocess_for_segmentation(image):
    img = cv2.resize(image, (256, 256)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()

def preprocess_for_classification(roi):
    import torchvision.transforms as transforms
    img = Image.fromarray(roi)
    transform = transforms.Compose([
        transforms. Resize((224, 224)),
        transforms. ToTensor(),
        transforms. Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def extract_roi(image, mask):
    h, w = image.shape[:2]
    mask = (mask > 0.5).astype(np.uint8)
    
    if mask.sum() == 0:
        cx, cy = w//2, h//2
        crop = image[max(0, cy-112):cy+112, max(0, cx-112):cx+112]
    else:
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = xs. min(), ys.min(), xs.max(), ys.max()
        pad = 30
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            crop = image
    
    crop = cv2.resize(crop, (224, 224))
    return crop

# =================================================================
# 4. HÀM INFERENCE CHÍNH
# =================================================================
def run_inference(image, patient_name, age, gender, note):
    """Chạy inference và lưu kết quả"""
    record_id = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 1. Segmentation
    img_tensor = preprocess_for_segmentation(image).to(device)
    
    with torch.no_grad():
        mask_pred = hybrid(img_tensor).squeeze().cpu().numpy()
    
    mask_resized = cv2.resize(mask_pred, (image.shape[1], image.shape[0]))
    mask_binary = (mask_resized > 0.5).astype(np.uint8)
    
    # 2. ROI extraction
    roi = extract_roi(image, mask_binary)
    
    # 3. Classification
    roi_tensor = preprocess_for_classification(roi).to(device)
    
    with torch.no_grad():
        logits = cls_model(roi_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
    pred_idx = np.argmax(probs)
    label = idx_to_class[pred_idx]
    conf = probs[pred_idx]
    
    # 4. Tạo overlay
    overlay = image.copy()
    mask_colored = cv2.applyColorMap(np.uint8(255 * mask_resized), cv2.COLORMAP_JET)
    mask_colored = cv2.cvtColor(mask_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(overlay, 0.6, mask_colored, 0.4, 0)
    
    # 5. Tạo mask visualization
    mask_vis = cv2.cvtColor(np.uint8(mask_binary * 255), cv2.COLOR_GRAY2BGR)
    
    # 6. Upload ảnh lên Cloudinary
    def upload_cv2(img_np, filename):
        _, buffer = cv2.imencode('.jpg', img_np)
        res = cloudinary.uploader.upload(buffer. tobytes(), folder="skin_app", public_id=f"{record_id}_{filename}")
        return res['secure_url']
    
    url_orig = upload_cv2(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), "original")
    url_ov = upload_cv2(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR), "overlay")
    url_mask = upload_cv2(mask_vis, "mask")
    
    # 7. Lưu vào Google Sheets
    data_dict = {
        "record_id": record_id,
        "timestamp": timestamp,
        "name": patient_name,
        "age": int(age),
        "gender": gender,
        "note": note,
        "diagnosis": label,
        "confidence": float(conf),
        "url_orig": url_orig,
        "url_ov": url_ov,
        "url_mask": url_mask
    }
    
    save_to_gsheets(data_dict)
    
    # 8. Thông tin hiển thị
    info = f"""
    **Bệnh án ID:** {record_id}

    **Bệnh nhân:** {patient_name}, {age} tuổi, {gender}

    **Chẩn đoán:** {label}

    **Độ tin cậy:** {conf*100:. 2f}%

    **Ghi chú:** {note if note else 'Không có'}

    **Thời gian:** {timestamp}

    *(Đã lưu lên Google Sheets & Cloudinary)*
    """
        
    return overlay, mask_vis, info, record_id, label, conf, timestamp  

from reportlab.lib.pagesizes import A4
from reportlab.lib. styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER, TA_LEFT

def generate_pdf_report(record_id, patient_name, age, gender, note, label, conf, timestamp, 
                        overlay_img, mask_img):
    """Tạo báo cáo PDF"""
    
    # Tạo buffer
    buffer = BytesIO()
    
    # Tạo document
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=2*cm, leftMargin=2*cm, 
                           topMargin=2*cm, bottomMargin=2*cm)
    
    # Container cho elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=20,
        alignment=TA_CENTER
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2ca02c'),
        spaceAfter=10
    )
    
    # Tiêu đề
    title = Paragraph("BÁO CÁO CHẨN ĐOÁN DA LIỄU", title_style)
    elements.append(title)
    elements.append(Spacer(1, 0.5*cm))
    
    # Thông tin bệnh nhân
    patient_info = [
        ["<b>Bệnh án ID: </b>", record_id],
        ["<b>Bệnh nhân:</b>", f"{patient_name}"],
        ["<b>Tuổi:</b>", f"{age} tuổi"],
        ["<b>Giới tính:</b>", gender],
        ["<b>Thời gian:</b>", timestamp],
        ["<b>Chẩn đoán:</b>", f"<font color='red'><b>{label}</b></font>"],
        ["<b>Độ tin cậy:</b>", f"{conf*100:. 2f}%"],
        ["<b>Ghi chú:</b>", note if note else "Không có"]
    ]
    
    table = Table(patient_info, colWidths=[5*cm, 12*cm])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.grey)
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 1*cm))
    
    # Thêm ảnh overlay
    heading = Paragraph("Ảnh Overlay (Phân vùng tổn thương)", heading_style)
    elements.append(heading)
    
    # Convert numpy array to PIL Image, then save to buffer
    overlay_pil = Image. fromarray(overlay_img)
    img_buffer = BytesIO()
    overlay_pil.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    img = RLImage(img_buffer, width=12*cm, height=12*cm)
    elements.append(img)
    elements.append(Spacer(1, 0.5*cm))
    
    # Thêm ảnh mask
    heading2 = Paragraph("Mask Phân vùng", heading_style)
    elements.append(heading2)
    
    mask_pil = Image. fromarray(mask_img)
    mask_buffer = BytesIO()
    mask_pil.save(mask_buffer, format='PNG')
    mask_buffer.seek(0)
    
    img2 = RLImage(mask_buffer, width=12*cm, height=12*cm)
    elements.append(img2)
    
    # Footer
    elements.append(Spacer(1, 1*cm))
    footer_text = "<i>Báo cáo được tạo tự động bởi Hệ thống Chẩn đoán Da liễu AI</i>"
    footer = Paragraph(footer_text, styles['Normal'])
    elements.append(footer)
    
    # Build PDF
    doc.build(elements)
    
    buffer.seek(0)
    return buffer
# =================================================================
# 5. HÀM TRA CỨU BỆNH ÁN
# =================================================================
def search_patient_records(patient_name=""):
    """Tìm kiếm bệnh án theo tên"""
    try: 
        worksheet = get_gsheets_client()
        records = worksheet.get_all_records()
        
        if not records:
            return "Chưa có bệnh án nào được lưu."
        
        df = pd.DataFrame(records)
        
        if patient_name:
            df = df[df['name'].str.contains(patient_name, case=False, na=False)]
        
        if df.empty:
            return "Không tìm thấy bệnh án."
        
        output = f"Tìm thấy {len(df)} bệnh án gần nhất:\n\n"
        
        for _, r in df.tail(5).iterrows():
            output += (
                f"ID: {r['record_id']}\n"
                f"Thời gian: {r['timestamp']}\n"
                f"Bệnh nhân: {r['name']}, {r['age']} tuổi\n"
                f"Chẩn đoán: {r['diagnosis']} ({float(r['confidence'])*100:.2f}%)\n"
                f"---\n"
            )
        
        return output
    
    except Exception as e: 
        return f"Lỗi khi tìm kiếm: {e}"

def load_patient_images(record_id):
    """Load ảnh từ Cloudinary dựa trên record_id"""
    try: 
        worksheet = get_gsheets_client()
        records = worksheet.get_all_records()
        
        df = pd.DataFrame(records)
        row = df[df['record_id'] == record_id]
        
        if row.empty:
            return None, None, None, "Không tìm thấy ID."
        
        r = row.iloc[0]
        
        def get_img(url):
            resp = requests.get(url)
            return cv2.cvtColor(np.array(Image.open(BytesIO(resp.content))), cv2.COLOR_RGB2BGR)
        
        original_img = get_img(r['url_orig'])
        overlay_img = get_img(r['url_ov'])
        mask_img = get_img(r['url_mask'])
        
        info = (
            f"Bệnh án ID: {r['record_id']}\n"
            f"Bệnh nhân: {r['name']}, {r['age']} tuổi\n"
            f"Chẩn đoán: {r['diagnosis']}\n"
            f"Ghi chú: {r['note']}"
        )
        
        return (
            cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), 
            cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB), 
            cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB), 
            info
        )
    
    except Exception as e:  
        return None, None, None, f"Lỗi khi tải ảnh: {e}"

# =================================================================
# 6.  GIAO DIỆN STREAMLIT
# =================================================================
st.set_page_config(page_title="Chẩn đoán Da Liễu", layout="wide")
st.title("🩺 Hệ thống Chẩn đoán bệnh da liễu AI")

tabs = st.tabs(["Chẩn đoán mới", "Tra cứu bệnh án"])

# TAB 1: CHẨN ĐOÁN MỚI
with tabs[0]: 
    st.header("Chẩn đoán mới (AI Diagnosis)")
    
    uploaded = st.file_uploader("Tải ảnh tổn thương", type=["jpg", "png", "jpeg"])
    patient_name = st.text_input("Tên bệnh nhân")
    age = st.number_input("Tuổi", min_value=0, max_value=120, step=1)
    gender = st.radio("Giới tính", options=["Nam", "Nữ"])
    note = st.text_area("Ghi chú (Tiền sử, mô tả triệu chứng ... )")
    
    if st.button("Chẩn đoán"):
        if uploaded and patient_name and age:   
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            with st.spinner("Đang phân tích..."):
                overlay, mask_vis, info, record_id, label, conf, timestamp = run_inference(img_rgb, patient_name, age, gender, note)
            
            # Hiển thị overlay
            st. image(overlay, caption="Ảnh Overlay (Phân vùng + Gốc)", use_container_width=True)
            
            # Hiển thị thông tin (có xuống dòng)
            st.info(info)
            
            st.write(f"**ID bệnh án:** `{record_id}` (Lưu lại để tra cứu)")
            
            # ===== NÚT DOWNLOAD PDF =====
            with st.spinner("Đang tạo báo cáo PDF..."):
                # mask_vis đã có rồi, KHÔNG CẦN tạo lại
                pdf_buffer = generate_pdf_report(
                    record_id, patient_name, age, gender, note, 
                    label, conf, timestamp,
                    overlay, mask_vis  # ← Dùng mask_vis đã có từ run_inference
                )

            st.download_button(
                label="📥 Tải báo cáo PDF",
                data=pdf_buffer,
                file_name=f"benh_an_{record_id}. pdf",
                mime="application/pdf"
            )
            
        else:
            st.warning("Vui lòng nhập đầy đủ thông tin và tải ảnh lên")

# TAB 2: TRA CỨU BỆNH ÁN
with tabs[1]:  
    st.header("Tra cứu bệnh án")
    
    search_name = st.text_input("Tìm kiếm theo tên bệnh nhân (để trống = tất cả)", key="search_name")
    
    if st.button("Tìm kiếm", key="btn_search"):
        with st.spinner("Đang tìm kiếm..."):
            search_result = search_patient_records(search_name)
        st.text_area("Kết quả:", value=search_result, height=250)

    st.divider()
    
    record_id = st.text_input("Nhập ID bệnh án", key="record_id_load")
    
    if st.button("Xem bệnh án", key="btn_load"):
        with st.spinner("Đang tải ảnh..."):
            orig, overlay, mask, info = load_patient_images(record_id)
        
        if orig is not None:  
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(orig, caption="Ảnh gốc", use_container_width=True)
            
            with col2:
                st.image(overlay, caption="Ảnh overlay", use_container_width=True)
            
            with col3:
                st.image(mask, caption="Mask phân vùng", use_container_width=True)
            
            st.info(info)
        else:
            st.warning(info)