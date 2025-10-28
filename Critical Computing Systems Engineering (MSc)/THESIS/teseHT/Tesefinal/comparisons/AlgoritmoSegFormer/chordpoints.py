# chordpoints.py

import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
import logging

from aerodynamic_utils import Point, compute_line_properties

# Suprimir avisos desnecessários dos transformers e do matplotlib
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ── Configurações ───────────────────────────────────────────────────────────
VIDEO_PATH  = r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/screenshotdosvideos/videos/Video 1.MP4"
NUM_FRAMES  = 20
TILE_SIZE   = (512, 512)
ARCH        = "mit-b1"
CHECKPOINT  = r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/Tesefinal/trainingmodel/segformer_final_mitb1.pth"
ROOT_DIR    = r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/Tesefinal/trainingmodel/sail_dataset"

# ── Carregar modelo e processador ───────────────────────────────────────────
def load_model(device):
    df       = pd.read_csv(os.path.join(ROOT_DIR, "class_dict_seg.csv"))
    id2label = df["name"].to_dict()
    label2id = {v:k for k,v in id2label.items()}

    fe    = SegformerImageProcessor()
    model = SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/{ARCH}",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        reshape_last_stage=True
    ).to(device)

    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    return model, fe

# ── Função para extrair frames do vídeo ──────────────────────────────────────
def extract_frames(path, n):
    cap   = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Não foi possível abrir o vídeo: {path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total - 1, n, dtype=int)
    frames = []

    for i in tqdm(idxs, desc="Extrair frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

# ── Segmentar e calcular propriedades ────────────────────────────────────────
def segment_and_compute(frames, model, fe, device):
    out = []
    for frame in tqdm(frames, desc="Segmentar & Computar"):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        pred = np.zeros((H, W), dtype=np.uint8)

        for y in range(0, H, TILE_SIZE[0]):
            for x in range(0, W, TILE_SIZE[1]):
                y1,y2 = y, min(y+TILE_SIZE[0], H)
                x1,x2 = x, min(x+TILE_SIZE[1], W)
                tile = rgb[y1:y2, x1:x2]
                inp  = fe(images=tile, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inp).logits
                up = F.interpolate(logits,
                                   size=(y2-y1, x2-x1),
                                   mode="bilinear",
                                   align_corners=False)
                pred[y1:y2, x1:x2] = up.argmax(1).squeeze().cpu().numpy()

        props = compute_line_properties(pred)
        overlay = frame.copy()
        mask_col = np.zeros_like(frame)
        mask_col[pred==1] = (0,255,0)
        overlay = cv2.addWeighted(frame, 0.6, mask_col, 0.4, 0)
        out.append((overlay, props))
    return out

# ── Helpers de visualização com Matplotlib ──────────────────────────────────
PREVIEW_MAX_W, PREVIEW_MAX_H = 1920, 1080

def get_scaled_image(img):
    h,w = img.shape[:2]
    scale = min(PREVIEW_MAX_W/w, PREVIEW_MAX_H/h, 1.0)
    if scale < 1.0:
        return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA), scale
    return img.copy(), 1.0

def show_preview_matplotlib(title, image):
    disp, _ = get_scaled_image(image[..., ::-1])
    fig_w, fig_h = disp.shape[1]/100, disp.shape[0]/100
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=100)
    ax.imshow(disp)
    ax.set_title(title)
    ax.axis('off')
    plt.show()

# ── Desenhar marcadores e imprimir métricas ─────────────────────────────────
def draw_and_show(results):
    section_names = ["Top", "Mid", "Bot"]
    metrics_data = {f"{m}_{s}": [] for m in ("Camber", "Draft", "Twist") for s in section_names}

    for idx, (overlay, props) in enumerate(results, 1):
        disp = overlay.copy()
        for sec in props:
            cv2.drawMarker(disp, (sec["first"].x, sec["first"].y), (0,165,255),
                           cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
            cv2.drawMarker(disp, (sec["last"].x,  sec["last"].y),  (0,165,255),
                           cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)
            cv2.drawMarker(disp, (int(sec["max_pt"].x), int(sec["max_pt"].y)),
                           (0,0,255), cv2.MARKER_TILTED_CROSS, markerSize=12, thickness=2)

        show_preview_matplotlib(f"Frame {idx}: pontos detetados", disp)

        c_vals, d_vals, t_vals = [], [], []
        for i in range(3):
            if i < len(props):
                sec = props[i]
                c_vals.append(round(sec["camber"], 2))
                d_vals.append(round(sec["draft"], 2))
                t_vals.append(round(sec["twist"], 2))
            else:
                c_vals.append(0.0)
                d_vals.append(0.0)
                t_vals.append(0.0)

        for name, c, d, t in zip(section_names, c_vals, d_vals, t_vals):
            metrics_data[f"Camber_{name}"].append(c)
            metrics_data[f"Draft_{name}"].append(d)
            metrics_data[f"Twist_{name}"].append(t)

        df_frame = pd.DataFrame({"Camber": c_vals, "Draft": d_vals, "Twist": t_vals}, index=section_names)
        print(f"\nFrame {idx} métricas:")
        print(df_frame.to_string())

    df_out = pd.DataFrame(metrics_data).T
    df_out.columns = [f"Frame_{i}" for i in range(1, len(results)+1)]
    df_out = df_out.reset_index().rename(columns={"index": "Metric"})

    out_dir = Path("ground-truth_Parameters_mitb1")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "video1_mitb1_chordpoints.csv"
    df_out.to_csv(csv_path, index=False, float_format="%.2f")
    print(f"\nParâmetros guardados em {csv_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, fe = load_model(device)
    frames = extract_frames(VIDEO_PATH, NUM_FRAMES)
    results = segment_and_compute(frames, model, fe, device)
    draw_and_show(results)

if __name__ == "__main__":
    main()
