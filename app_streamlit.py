import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
import json
import os
import uuid
import datetime
from fpdf import FPDF
import zipfile
from PIL import Image

# ==== SAFE FILENAME ====
def safe_str(s):
    return "".join(c for c in s if c.isalnum() or c in (' ', '_')).rstrip().replace(' ', '_')


# =================================================================
# 1. ĐỊNH NGHĨA CÁC LỚP MÔ HÌNH (Phải đặt trước khi sử dụng)
# =================================================================

class HybridSegmentation(nn.Module):
    def __init__(self, unet, deeplab):
        super().__init__()
        self.unet = unet
        self.deeplab = deeplab
    def forward(self, x):
        with torch.no_grad():
            pred_unet = torch.sigmoid(self.unet(x))
            pred_dl = torch.sigmoid(self.deeplab(x))
            return torch.max(pred_unet, pred_dl)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
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
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(in_channels, reduction)
        self.spatial_att = SpatialAttention()
    def forward(self, x):
        x_att = x * self.channel_att(x)
        return x_att * self.spatial_att(x_att)

class EfficientNetWithAttention(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_b0', pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.attention = CBAM(self.feature_dim, reduction=16)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        features = self.backbone.forward_features(x)
        features_att = self.attention(features)
        features_pooled = self.global_pool(features_att).flatten(1)
        return self.classifier(features_pooled)

# =================================================================
# 2. HÀM TẢI MÔ HÌNH TỐI ƯU (Sử dụng Cache để tiết kiệm RAM)
# =================================================================

@st.cache_resource
def load_all_models():
    import segmentation_models_pytorch as smp
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hàm xử lý lỗi đường dẫn tuyệt đối từ Windows sang Linux
    def get_valid_path(json_data):
        original_path = json_data["paths"]["best_model"]
        filename = os.path.basename(original_path)  # Chỉ lấy tên file
        return filename if os.path.exists(filename) else original_path

    # Tải cấu hình từ các file JSON
    with open("02_unet_complete.json", "r") as f: unet_ckpt = json.load(f)
    with open("03_deeplabv3plus_complete.json", "r") as f: deeplab_ckpt = json.load(f)
    with open("06_classification_complete.json", "r") as f: cls_ckpt = json.load(f)

    # Khởi tạo và tải trọng số cho Segmentation
    unet = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1, activation=None)
    unet.load_state_dict(torch.load(get_valid_path(unet_ckpt), map_location=device)["model_state_dict"])
    
    deeplab = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=1, activation=None)
    deeplab.load_state_dict(torch.load(get_valid_path(deeplab_ckpt), map_location=device)["model_state_dict"])
    
    hybrid_model = HybridSegmentation(unet, deeplab).eval().to(device)

    # Khởi tạo và tải trọng số cho Classification
    num_classes = cls_ckpt["config"]["num_classes"]
    cls_path = get_valid_path(cls_ckpt)
    state = torch.load(cls_path, map_location=device)
    
    # Lấy ánh xạ lớp
    class_to_idx = state.get("class_to_idx") or cls_ckpt.get("class_to_idx")
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    
    cls_model = EfficientNetWithAttention(num_classes=num_classes)
    cls_model.load_state_dict(state['model_state_dict'])
    cls_model = cls_model.eval().to(device)
    
    return hybrid_model, cls_model, idx_to_class, device

# Thực thi tải mô hình duy nhất một lần
hybrid, cls_model, idx_to_class, device = load_all_models()

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
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return transform(img).unsqueeze(0)

def extract_roi(image, mask):
    h, w = image.shape[:2]
    mask = (mask > 0.5).astype(np.uint8)
    if mask.sum() == 0:
        cx, cy = w//2, h//2
        crop = image[max(0,cy-112):cy+112, max(0,cx-112):cx+112]
    else:
        ys, xs = np.where(mask)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        pad = 30
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0: crop = image
        crop = cv2.resize(crop, (224,224))
    return crop

def run_inference(image, patient_name, age, gender, note):
    import warnings
    warnings.filterwarnings("ignore")
    record_id = str(uuid.uuid4())[:8]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = safe_str(patient_name)
    records_dir = "patient_records"
    patient_dir = os.path.join(records_dir, f"{safe_name}_{record_id}")
    os.makedirs(patient_dir, exist_ok=True)

    orig = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    tensor = preprocess_for_segmentation(img_rgb).to(device)
    with torch.no_grad():
        mask = hybrid(tensor).squeeze().cpu().numpy()
    mask_bin = (mask > 0.5).astype(np.uint8)
    mask_vis = cv2.resize(mask_bin*255, (img_rgb.shape[1], img_rgb.shape[0]))
    overlay = img_rgb.copy()
    colored_mask = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)
    roi = extract_roi(img_rgb, mask)
    roi_tensor = preprocess_for_classification(roi).to(device)
    with torch.no_grad():
        output = cls_model(roi_tensor)
        probs = torch.softmax(output,1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        label = idx_to_class[pred_idx]
        conf = probs[pred_idx]
    original_path = os.path.join(patient_dir, f"original_{timestamp}.jpg")
    overlay_path = os.path.join(patient_dir, f"overlay_{timestamp}.jpg")
    mask_path = os.path.join(patient_dir, f"mask_{timestamp}.png")
    cv2.imwrite(original_path, orig)
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(mask_path, mask_vis)
    record_data = {
        "record_id": record_id,
        "timestamp": timestamp,
        "patient_info": {
            "name": patient_name,
            "age": age,
            "gender": gender,
            "note": note
        },
        "diagnosis": {
            "label": label,
            "confidence": float(conf),
            "confidence_percent": float(conf * 100)
        },
        "images": {
            "original": original_path,
            "overlay": overlay_path,
            "mask": mask_path
        }
    }
    record_file = os.path.join(patient_dir, "record.json")
    with open(record_file, "w", encoding="utf-8") as f:
        json.dump(record_data, f, ensure_ascii=False, indent=2)
    csv_file = "records.csv"
    if not os.path.exists(csv_file):
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write("record_id,patient_name,age,gender,note,diagnosis,confidence,patient_dir\n")
    rel_patient_dir = os.path.relpath(patient_dir, os.getcwd())
    with open(csv_file,"a",encoding="utf-8") as f:
        f.write(f'{record_id},"{patient_name}",{age},{gender},"{note}",{label},{conf:.4f},"{rel_patient_dir}"\n')
    info = f"ID bệnh án: {record_id}\nChẩn đoán: {label}\nĐộ tin cậy: {conf*100:.2f}%\nBệnh nhân: {patient_name} tuổi {age} ({gender})\nĐã lưu tại: {patient_dir}"
    return overlay, info, record_id

def search_patient_records(patient_name=""):
    records_dir = "patient_records"
    if not os.path.exists(records_dir):
        return "Chưa có bệnh án nào được lưu."
    results = []
    safe_query = safe_str(patient_name.lower())
    for patient_folder in os.listdir(records_dir):
        folder_name_only = patient_folder.lower()
        if safe_query in folder_name_only or patient_name == "":
            patient_path = os.path.join(records_dir, patient_folder)
            record_file = os.path.join(patient_path, "record.json")
            if os.path.exists(record_file):
                with open(record_file, "r", encoding="utf-8") as f:
                    record = json.load(f)
                results.append(record)
    if not results:
        return f"Không tìm thấy bệnh án cho bệnh nhân: {patient_name}"
    output = f"Tìm thấy {len(results)} bệnh án gần nhất:\n\n"
    for record in results[-5:]:
        output += f"ID: {record['record_id']}\n"
        output += f"Thời gian: {record['timestamp']}\n"
        output += f"Bệnh nhân: {record['patient_info']['name']}, {record['patient_info']['age']} tuổi ({record['patient_info']['gender']})\n"
        output += f"Chẩn đoán: {record['diagnosis']['label']} ({record['diagnosis']['confidence_percent']:.2f}%)\n"
        folder_name = f"{safe_str(record['patient_info']['name'])}_{record['record_id']}"
        output += f"Thư mục: {os.path.join(records_dir, folder_name)}\n\n"
    return output

def load_patient_images(record_id):
    records_dir = "patient_records"
    if not os.path.exists(records_dir):
        return None, None, None, "Thư mục patient_records không tồn tại."
    target_folder = None
    for patient_folder in os.listdir(records_dir):
        if patient_folder.endswith(f"_{record_id}"):
            target_folder = patient_folder
            break
    if target_folder is None:
        return None, None, None, f"Không tìm thấy bệnh án với ID: {record_id}"
    patient_path = os.path.join(records_dir, target_folder)
    record_file = os.path.join(patient_path, "record.json")
    with open(record_file, "r", encoding="utf-8") as f:
        record = json.load(f)
    def load_img(path):
        if not os.path.exists(path): return None
        img = cv2.imread(path)
        if img is not None:
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        return img
    original_img = load_img(record['images']['original'])
    overlay_img = load_img(record['images']['overlay'])
    mask_img = load_img(record['images']['mask'])
    info = f"Bệnh án ID: {record['record_id']}\n"
    info += f"Thời gian: {record['timestamp']}\n"
    info += f"Bệnh nhân: {record['patient_info']['name']}, {record['patient_info']['age']} tuổi ({record['patient_info']['gender']})\n"
    info += f"Ghi chú: {record['patient_info']['note']}\n"
    info += f"Chẩn đoán: {record['diagnosis']['label']} ({record['diagnosis']['confidence_percent']:.2f}%)"
    return original_img, overlay_img, mask_img, info

def download_patient_zip(record_id):
    records_dir = "patient_records"
    target_folder = None
    for patient_folder in os.listdir(records_dir):
        if patient_folder.endswith(f"_{record_id}"):
            target_folder = os.path.join(records_dir, patient_folder)
            break
    if not target_folder or not os.path.exists(target_folder):
        return None
    zip_path = os.path.join(target_folder, f"{record_id}_record.zip")
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for filename in os.listdir(target_folder):
            file_path = os.path.join(target_folder, filename)
            zipf.write(file_path, arcname=filename)
    return zip_path if os.path.exists(zip_path) else None

import os
import json
import textwrap
import re
from PIL import Image, ImageDraw, ImageFont

# Cấu hình (thay đổi theo ý bạn)
SCALE = 1.4
BASE_TITLE = 32
BASE_HEADER = 22
BASE_FIELD = 22
BASE_MAIN = 22
BASE_SMALL = 14
IMG_TARGET_MM = 55
MIN_IMG_SCALE = 0.80

def _remove_emoji(s):
    try:
        return re.sub(r'[\U00010000-\U0010ffff]', '', s)
    except re.error:
        return ''.join(ch for ch in s if ord(ch) <= 0xFFFF)


def _find_font_pair():
    # Lấy thư mục hiện tại của file script này
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Định nghĩa các ứng viên bằng đường dẫn tương đối
    # os.path.join giúp tự động điều chỉnh dấu / hoặc \ tùy theo hệ điều hành (Windows/Linux)
    candidates = [
        (os.path.join(base_dir, "fonts", "static", "Roboto-Regular.ttf"),
         os.path.join(base_dir, "fonts", "static", "Roboto-Bold.ttf")),
        
        (os.path.join(base_dir, "fonts", "DejaVuSans.ttf"),
         os.path.join(base_dir, "fonts", "DejaVuSans-Bold.ttf")),
        
        # Fallback cho Linux server nếu các font trên bị thiếu
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
         "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    
    for reg, bold in candidates:
        if os.path.exists(reg):
            return reg, (bold if os.path.exists(bold) else None)
            
    return None, None

def _text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0,0), text, font=font)
        return (bbox[2]-bbox[0], bbox[3]-bbox[1])
    except Exception:
        try:
            return font.getsize(text)
        except Exception:
            return (len(text)*6, 12)

def export_patient_pdf(record_id):
    records_dir = "patient_records"
    if not os.path.exists(records_dir):
        return None

    # tìm folder bệnh án
    target_folder = None
    for patient_folder in os.listdir(records_dir):
        if patient_folder.endswith(f"_{record_id}"):
            target_folder = os.path.join(records_dir, patient_folder)
            break
    if not target_folder:
        return None

    record_file = os.path.join(target_folder, "record.json")
    if not os.path.exists(record_file):
        return None

    with open(record_file, "r", encoding="utf-8") as f:
        record = json.load(f)

    original = record['images'].get("original")
    overlay = record['images'].get("overlay")
    mask = record['images'].get("mask")
    pdf_path = os.path.join(target_folder, f"{record_id}_report.pdf")

    # A4 @ DPI
    DPI = 150
    A4_MM = (210, 297)
    W, H = (int(A4_MM[0] / 25.4 * DPI), int(A4_MM[1] / 25.4 * DPI))

    # compute font sizes
    title_size = int(BASE_TITLE * SCALE * DPI/150)
    header_size = int(BASE_HEADER * SCALE * DPI/150)
    field_size = int(BASE_FIELD * SCALE * DPI/150)
    main_size = int(BASE_MAIN * SCALE * DPI/150)
    small_size = int(BASE_SMALL * SCALE * DPI/150)

    # line spacing settings: tăng hai giá trị này để tăng khoảng cách giữa các dòng
    LINE_GAP = int(14 * DPI/150)       # chính — khoảng cách giữa các dòng lớn (mặc định 6 trước đây)
    LINE_GAP_SMALL = int(10 * DPI/150)  # nhỏ — dùng cho các dòng phụ/nhỏ

    # extra vertical gap between boxes (mm)
    EXTRA_GAP_MM = 8
    EXTRA_GAP_PX = int(EXTRA_GAP_MM / 25.4 * DPI)

    # load fonts (fallback)
    reg_fp, bold_fp = _find_font_pair()
    try:
        if reg_fp:
            font_title = ImageFont.truetype(bold_fp or reg_fp, size=title_size)
            font_header = ImageFont.truetype(bold_fp or reg_fp, size=header_size)
            font_field = ImageFont.truetype(reg_fp, size=field_size)
            font_main = ImageFont.truetype(reg_fp, size=main_size)
            font_small = ImageFont.truetype(reg_fp, size=small_size)
        else:
            raise Exception("No TTF")
    except Exception:
        font_title = ImageFont.load_default()
        font_header = ImageFont.load_default()
        font_field = ImageFont.load_default()
        font_main = ImageFont.load_default()
        font_small = ImageFont.load_default()

    def new_page():
        return Image.new("RGB", (W, H), (255,255,255))

    page = new_page()
    draw = ImageDraw.Draw(page)

    margin = int(16/25.4*DPI)
    cur_y = margin

    # Title
    title = _remove_emoji("HỒ SƠ BỆNH ÁN DA LIỄU")
    tw, th = _text_size(draw, title, font_title)
    draw.text(((W - tw)//2, cur_y), title, fill=(10,40,80), font=font_title)
    cur_y += th + LINE_GAP  # dùng LINE_GAP sau tiêu đề

    # divider + extra gap
    draw.line((margin, cur_y, W-margin, cur_y), fill=(220,220,220), width=max(1, int(1 * DPI/150)))
    cur_y += LINE_GAP + EXTRA_GAP_PX

    # PATIENT INFO: compute exact heights and draw box exactly
    left_lines = [
        f"ID: {record.get('record_id','')}",
        f"Họ tên: {record['patient_info'].get('name','')}",
        f"Giới tính: {record['patient_info'].get('gender','')}",
        f"Tuổi: {record['patient_info'].get('age','')}"
    ]
    left_heights = [_text_size(draw, _remove_emoji(l), font_field)[1] for l in left_lines]
    left_block_h = sum(left_heights) + (len(left_lines)-1)*LINE_GAP_SMALL

    notes = record['patient_info'].get('note','')
    notes_wrapped = textwrap.wrap(notes, width=60) if notes else []
    right_lines = ["Ghi chú:"] + notes_wrapped
    right_heights = [_text_size(draw, _remove_emoji(l), font_main)[1] for l in right_lines]
    right_block_h = sum(right_heights) + (len(right_lines)-1)*LINE_GAP_SMALL

    inner_pad = int(12 * DPI/150)
    box_h = max(left_block_h, right_block_h) + inner_pad*4

    box_x0 = margin
    box_x1 = W - margin
    border_w = max(2, int(3 * DPI/150))
    draw.rectangle([box_x0, cur_y, box_x1, cur_y + box_h], outline=(50,130,200), width=border_w)

    # left column
    lx = box_x0 + inner_pad
    ly = cur_y + inner_pad
    for l in left_lines:
        draw.text((lx, ly), _remove_emoji(l), fill=(0,0,0), font=font_field)
        lh = _text_size(draw, _remove_emoji(l), font_field)[1]
        ly += lh + LINE_GAP_SMALL

    # right column
    rx = lx + int((box_x1 - box_x0)*0.45)
    ry = cur_y + inner_pad
    for l in right_lines:
        draw.text((rx, ry), _remove_emoji(l), fill=(0,0,0), font=font_main)
        rh = _text_size(draw, _remove_emoji(l), font_main)[1]
        ry += rh + LINE_GAP_SMALL

    # timestamp top-right inside box
    ts_text = f"Thời gian: {record.get('timestamp','')}"
    ts_w, ts_h = _text_size(draw, ts_text, font_small)
    draw.text((box_x1 - inner_pad - ts_w, cur_y + inner_pad), ts_text, fill=(0,0,0), font=font_small)

    # extra vertical gap after patient box
    cur_y += box_h + LINE_GAP + EXTRA_GAP_PX

    # DIAGNOSIS box
    diag_label = f"Chẩn đoán: {record['diagnosis'].get('label','')}"
    diag_conf = f"Độ tin cậy: {record['diagnosis'].get('confidence_percent',0):.2f} %"
    diag_label_h = _text_size(draw, _remove_emoji(diag_label), font_field)[1]
    diag_conf_h = _text_size(draw, diag_conf, font_field)[1]
    diag_block_h = diag_label_h + diag_conf_h + inner_pad*4 + LINE_GAP_SMALL

    draw.rectangle([margin, cur_y, W - margin, cur_y + diag_block_h], outline=(170,200,180), width=max(1, int(2 * DPI/150)))
    d_y = cur_y + inner_pad
    draw.text((margin + inner_pad, d_y), "KẾT QUẢ CHẨN ĐOÁN", fill=(0,80,40), font=font_header)
    d_y += _text_size(draw, "KẾT QUẢ CHẨN ĐOÁN", font_header)[1] + LINE_GAP_SMALL
    draw.text((margin + inner_pad, d_y), _remove_emoji(diag_label), fill=(0,0,0), font=font_field)
    tw_conf, _ = _text_size(draw, diag_conf, font_field)
    draw.text((W - margin - inner_pad - tw_conf, d_y), diag_conf, fill=(0,0,0), font=font_field)

    # extra gap after diag box
    cur_y += diag_block_h + LINE_GAP + EXTRA_GAP_PX

    # IMAGES: sizing and page fit
    images_desired_h = int(IMG_TARGET_MM / 25.4 * DPI)
    caption_h = _text_size(draw, "Ảnh", font_small)[1]
    images_total_h = images_desired_h + caption_h + int(8 * DPI/150)

    footer_h = int(18/25.4*DPI)
    remaining = H - cur_y - footer_h - int(12/25.4*DPI)

    pages = []
    if remaining >= images_total_h:
        chosen_img_h = images_desired_h
        fit_on_same = True
    else:
        scale_factor = remaining / images_total_h
        if scale_factor >= MIN_IMG_SCALE:
            chosen_img_h = max(int(images_desired_h * scale_factor), int(images_desired_h * MIN_IMG_SCALE))
            fit_on_same = True
        else:
            fit_on_same = False
            chosen_img_h = images_desired_h

    if fit_on_same:
        available_w = W - 2*margin
        img_w = int((available_w - 20) / 3)
        img_h = chosen_img_h
        x0 = margin
        y0 = cur_y
        for i, path in enumerate([original, overlay, mask]):
            xi = x0 + i*(img_w + 10)
            draw.rectangle([xi-3, y0-3, xi+img_w+3, y0+img_h+3], fill=(250,250,250))
            if path and os.path.exists(path):
                try:
                    im = Image.open(path).convert("RGB")
                    try:
                        resample = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample = Image.ANTIALIAS
                    im.thumbnail((img_w, img_h), resample)
                    paste_x = xi + (img_w - im.size[0])//2
                    paste_y = y0 + (img_h - im.size[1])//2
                    page.paste(im, (paste_x, paste_y))
                    draw.rectangle([xi, y0, xi+img_w, y0+img_h], outline=(200,200,200), width=1)
                except Exception:
                    draw.rectangle([xi, y0, xi+img_w, y0+img_h], outline=(200,200,200), width=1)
            else:
                draw.rectangle([xi, y0, xi+img_w, y0+img_h], outline=(200,200,200), width=1)
            caption = ["Ảnh tổn thương gốc","Overlay phân vùng AI","Mask phân vùng"][i]
            cw, ch = _text_size(draw, caption, font_small)
            draw.text((xi + (img_w - cw)//2, y0 + img_h + LINE_GAP_SMALL), caption, fill=(90,90,90), font=font_small)

        footer = _remove_emoji("Hệ thống Chẩn đoán bệnh da liễu AI")
        fw, fh = _text_size(draw, footer, font_small)
        draw.text(((W - fw)//2, H - int(16/25.4*DPI)), footer, fill=(120,120,120), font=font_small)
        pages.append(page)
    else:
        note = "Ảnh minh họa (xem trang tiếp theo)"
        draw.text((margin, cur_y), note, fill=(80,80,80), font=font_main)
        footer = _remove_emoji("Hệ thống Chẩn đoán bệnh da liễu AI")
        fw, fh = _text_size(draw, footer, font_small)
        draw.text(((W - fw)//2, H - int(16/25.4*DPI)), footer, fill=(120,120,120), font=font_small)
        pages.append(page)

        page2 = new_page()
        d2 = ImageDraw.Draw(page2)
        available_w = W - 2*margin
        img_w = int((available_w - 20) / 3)
        img_h = images_desired_h
        x0 = margin
        y0 = margin + int(6/25.4*DPI)
        for i, path in enumerate([original, overlay, mask]):
            xi = x0 + i*(img_w + 10)
            d2.rectangle([xi-3, y0-3, xi+img_w+3, y0+img_h+3], fill=(250,250,250))
            if path and os.path.exists(path):
                try:
                    im = Image.open(path).convert("RGB")
                    try:
                        resample = Image.Resampling.LANCZOS
                    except AttributeError:
                        resample = Image.ANTIALIAS
                    im.thumbnail((img_w, img_h), resample)
                    paste_x = xi + (img_w - im.size[0])//2
                    paste_y = y0 + (img_h - im.size[1])//2
                    page2.paste(im, (paste_x, paste_y))
                    d2.rectangle([xi, y0, xi+img_w, y0+img_h], outline=(200,200,200), width=1)
                except Exception:
                    d2.rectangle([xi, y0, xi+img_w, y0+img_h], outline=(200,200,200), width=1)
            else:
                d2.rectangle([xi, y0, xi+img_w, y0+img_h], outline=(200,200,200), width=1)
            caption = ["Ảnh tổn thương gốc","Overlay phân vùng AI","Mask phân vùng"][i]
            cw, ch = _text_size(d2, caption, font_small)
            d2.text((xi + (img_w - cw)//2, y0 + img_h + LINE_GAP_SMALL), caption, fill=(90,90,90), font=font_small)

        fw, fh = _text_size(d2, footer, font_small)
        d2.text(((W - fw)//2, H - int(16/25.4*DPI)), footer, fill=(120,120,120), font=font_small)
        pages.append(page2)

    # Save pages
    try:
        if len(pages) == 1:
            pages[0].save(pdf_path, "PDF", resolution=DPI)
        else:
            pages[0].save(pdf_path, "PDF", resolution=DPI, save_all=True, append_images=pages[1:])
        return pdf_path if os.path.exists(pdf_path) else None
    except Exception as e:
        print("Lỗi khi lưu PDF bằng Pillow:", e)
        return None

# ---- UI -----
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
    note = st.text_area("Ghi chú (Tiền sử, mô tả triệu chứng ...)")
    if st.button("Chẩn đoán"):
        if uploaded and patient_name and age:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            overlay, info, record_id = run_inference(img_rgb, patient_name, age, gender, note)
            st.image(overlay, caption="Ảnh Overlay (Phân vùng + Gốc)", use_container_width=True)
            st.success(info)
            st.write(f"ID bệnh án (medical record ID): `{record_id}`\n(Lưu lại để tra cứu)")
        else:
            st.warning("Vui lòng nhập đầy đủ thông tin và tải ảnh lên")

# TAB 2: TRA CỨU BỆNH ÁN
with tabs[1]:
    st.header("Tra cứu bệnh án")
    search_name = st.text_input("Tìm kiếm theo tên bệnh nhân (để trống = tất cả)", key="search_name")
    if st.button("Tìm kiếm", key="btn_search"):
        search = search_patient_records(search_name)
        st.text_area("Kết quả:", value=search, height=250)

    record_id = st.text_input("Nhập ID bệnh án", key="record_id_load")
    if st.button("Xem bệnh án", key="btn_load"):
        orig, overlay, mask, info = load_patient_images(record_id)
        if orig is not None:
            st.image(orig, caption="Ảnh gốc", use_container_width=True)
            st.image(overlay, caption="Ảnh overlay", use_container_width=True)
            st.image(mask, caption="Mask phân vùng", use_container_width=True)
            st.info(info)
        else:
            st.warning(info)

    if st.button("Tải file ảnh (ZIP)", key="btn_zip"):
        zip_path = download_patient_zip(record_id)
        if zip_path and os.path.exists(zip_path):
            with open(zip_path, "rb") as zipf:
                st.download_button("Tải ZIP", zipf, file_name=os.path.basename(zip_path))
        else:
            st.warning("Không tìm thấy hoặc chưa xuất được Zip")

    if st.button("Tải báo cáo PDF", key="btn_pdf"):
        pdf_path = export_patient_pdf(record_id)
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as pdff:
                st.download_button("Tải PDF", pdff, file_name=os.path.basename(pdf_path))
        else:
            st.warning("Không tìm thấy hoặc chưa xuất được PDF")