# segment_and_report.py

import os
import cv2
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from aerodynamic_utils import compute_line_properties
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import logging

# ── Config ─────────────────────────────────────────────────────────────────────
VIDEO_PATH  = r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/screenshotdosvideos/videos/Video 23.MP4"
OUTPUT_PDF  = "segformer_pdfs/SegFormer_video23.pdf"
OUTPUT_CSV  = "parametrosSegformer/SegFormer_video23.csv"
NUM_FRAMES  = 20
TILE_SIZE   = (512, 512)
ARCH        = "mit-b3"
CHECKPOINT  = r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/Tesefinal/trainingmodel/segformer_final.pth"
ROOT_DIR    = r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/Tesefinal/trainingmodel/sail_dataset"

# A4 portrait & DPI
A4_PORTRAIT = (8.27, 11.69)
DPI         = 300

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


def load_model(device):
    df       = pd.read_csv(os.path.join(ROOT_DIR, "class_dict_seg.csv"))
    id2label = df["name"].to_dict()
    label2id = {v: k for k, v in id2label.items()}
    fe       = SegformerImageProcessor()
    model    = SegformerForSemanticSegmentation.from_pretrained(
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


def extract_frames(path, n):
    cap   = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in tqdm(idxs, desc="Extracting frames", unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


def segment_and_analyze(frames, model, fe, device):
    out = []
    for frame in tqdm(frames, desc="Segment & Analyze", unit="frame"):
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        pred = np.zeros((H, W), dtype=np.uint8)

        # tiled inference
        for y in range(0, H, TILE_SIZE[0]):
            for x in range(0, W, TILE_SIZE[1]):
                y1, y2 = y, min(y + TILE_SIZE[0], H)
                x1, x2 = x, min(x + TILE_SIZE[1], W)
                tile = rgb[y1:y2, x1:x2]
                inp  = fe(images=tile, return_tensors="pt").to(device)
                with torch.no_grad():
                    logits = model(**inp).logits
                up = F.interpolate(
                    logits,
                    size=(y2 - y1, x2 - x1),
                    mode="bilinear",
                    align_corners=False
                )
                pred[y1:y2, x1:x2] = up.argmax(1).squeeze().cpu().numpy()

        props = compute_line_properties(pred)

        # create overlay with segmentation
        overlay = frame.copy()
        mask_color = np.zeros_like(frame)
        mask_color[pred == 1] = (0, 255, 0)
        overlay = cv2.addWeighted(frame, 0.6, mask_color, 0.4, 0)

        # binary mask for separate page
        mask = (pred * 255).astype(np.uint8)

        out.append((overlay, mask, props))
    return out


def _add_metrics_table(ax, props):
    """
    Draws a 6×3 table:
      Row 1:  Camber Top   | Draft Top   | Twist Top
      Row 2:  val_cam_top  | val_draft_top | val_twist_top
      Row 3:  Camber Mid   | Draft Mid   | Twist Mid
      Row 4:  val_cam_mid  | val_draft_mid | val_twist_mid
      Row 5:  Camber Bot   | Draft Bot   | Twist Bot
      Row 6:  val_cam_bot  | val_draft_bot | val_twist_bot
    """
    headers_1 = ["Camber Top", "Draft Top", "Twist Top"]
    vals_1 = [props[0]['camber'], props[0]['draft'], props[0]['twist']]
    headers_2 = ["Camber Mid", "Draft Mid", "Twist Mid"]
    vals_2 = [props[1]['camber'], props[1]['draft'], props[1]['twist']]
    headers_3 = ["Camber Bot", "Draft Bot", "Twist Bot"]
    vals_3 = [props[2]['camber'], props[2]['draft'], props[2]['twist']]

    tbl_data = []
    # Top
    tbl_data.append(headers_1)
    tbl_data.append([
        f"{vals_1[0]:.2f}".replace(".", ",") + "%",
        f"{vals_1[1]:.2f}".replace(".", ",") + "%",
        f"{vals_1[2]:.2f}".replace(".", ",") + "º"
    ])
    # Mid
    tbl_data.append(headers_2)
    tbl_data.append([
        f"{vals_2[0]:.2f}".replace(".", ",") + "%",
        f"{vals_2[1]:.2f}".replace(".", ",") + "%",
        f"{vals_2[2]:.2f}".replace(".", ",") + "º"
    ])
    # Bot
    tbl_data.append(headers_3)
    tbl_data.append([
        f"{vals_3[0]:.2f}".replace(".", ",") + "%",
        f"{vals_3[1]:.2f}".replace(".", ",") + "%",
        f"{vals_3[2]:.2f}".replace(".", ",") + "º"
    ])

    tbl = ax.table(
        cellText=tbl_data,
        cellLoc="center",
        loc="center",
        colWidths=[0.32, 0.32, 0.32]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)

    # Style header rows (even indices) black with white bold text; value rows white with black text
    for (r, c), cell in tbl.get_celld().items():
        if r % 2 == 0:  # header rows: 0,2,4
            cell.set_facecolor("black")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        else:           # value rows: 1,3,5
            cell.set_facecolor("white")
            cell.get_text().set_color("black")
        cell.set_edgecolor("black")
        cell.set_linewidth(1.0)

    ax.axis("off")

def create_pdf(path, data, timings):
    with PdfPages(path) as pdf:
        # ── Page 1: summary table ─────────────────────────────────────────────
        fig = plt.figure(figsize=A4_PORTRAIT, dpi=DPI)
        ax  = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")

        headers = ["METRIC"] + [str(i) for i in range(1, NUM_FRAMES + 1)]
        camber = [
            ["CAMBER TOP"] + [f"{p[2][0]['camber']:.2f}" for p in data],
            ["CAMBER MID"] + [f"{p[2][1]['camber']:.2f}" for p in data],
            ["CAMBER BOT"] + [f"{p[2][2]['camber']:.2f}" for p in data],
        ]
        draft = [
            ["DRAFT TOP"] + [f"{p[2][0]['draft']:.2f}" for p in data],
            ["DRAFT MID"] + [f"{p[2][1]['draft']:.2f}" for p in data],
            ["DRAFT BOT"] + [f"{p[2][2]['draft']:.2f}" for p in data],
        ]
        twist = [
            ["TWIST TOP"] + [f"{p[2][0]['twist']:.2f}" for p in data],
            ["TWIST MID"] + [f"{p[2][1]['twist']:.2f}" for p in data],
            ["TWIST BOT"] + [f"{p[2][2]['twist']:.2f}" for p in data],
        ]

        blank = [""] * (NUM_FRAMES + 1)
        all_rows = camber + [blank] + draft + [blank] + twist

        tbl = ax.table(
            cellText=all_rows,
            colLabels=headers,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(6)
        tbl.auto_set_column_width(col=list(range(len(headers))))
        tbl.scale(1, 1.2)

        for (r, c), cell in tbl.get_celld().items():
            if r == 0 or c == 0:
                cell.get_text().set_weight("bold")
                cell.set_edgecolor("black")
                cell.set_linewidth(1.0)
            else:
                cell.set_edgecolor("lightgrey")
                cell.set_linewidth(0.5)

        pdf.savefig(fig)
        plt.close(fig)

        # ── Pages 2+: Overlay + metrics, then Mask + metrics ────────────────
        for idx, (frame, mask, props) in enumerate(tqdm(data, desc="PDF pages", unit="page"), 1):
            # Overlay + metrics
            fig = plt.figure(figsize=A4_PORTRAIT, dpi=DPI)
            gs  = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.1)
            ax_table = fig.add_subplot(gs[0])
            ax_im    = fig.add_subplot(gs[1])

            _add_metrics_table(ax_table, props)
            ax_im.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ax_im.axis("off")
            ax_im.set_title(f"Frame {idx}: Overlay", fontsize=14, pad=8)

            pdf.savefig(fig)
            plt.close(fig)

            # Mask + metrics
            fig = plt.figure(figsize=A4_PORTRAIT, dpi=DPI)
            gs  = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.1)
            ax_table = fig.add_subplot(gs[0])
            ax_im    = fig.add_subplot(gs[1])

            _add_metrics_table(ax_table, props)
            ax_im.imshow(mask, cmap="gray", vmin=0, vmax=255)
            ax_im.axis("off")
            ax_im.set_title(f"Frame {idx}: Binary Mask", fontsize=14, pad=8)

            pdf.savefig(fig)
            plt.close(fig)

def save_metrics_csv(data, csv_path):
    """
    Flatten props for the first‐page table into a DataFrame and save as CSV:
      - Index: Metric names (Camber_Top, …)
      - Columns: Frame_1 … Frame_N
    """
    rows = []
    row_names = []
    # camber
    for idx, lvl in enumerate(["Top", "Mid", "Bot"]):
        row_names.append(f"Camber_{lvl}")
        rows.append([p[2][idx]['camber'] for p in data])
    # draft
    for idx, lvl in enumerate(["Top", "Mid", "Bot"]):
        row_names.append(f"Draft_{lvl}")
        rows.append([p[2][idx]['draft'] for p in data])
    # twist
    for idx, lvl in enumerate(["Top", "Mid", "Bot"]):
        row_names.append(f"Twist_{lvl}")
        rows.append([p[2][idx]['twist'] for p in data])

    # build DataFrame: rows=metrics, cols=Frame_1…Frame_N
    df = pd.DataFrame(
        data=np.array(rows),
        index=row_names,
        columns=[f"Frame_{i+1}" for i in range(len(data))]
    )

    # write with Metric as first column name, two decimals
    df.to_csv(
        csv_path,
        index=True,
        index_label="Metric",
        float_format="%.2f"
    )
    print(f"\nWrote metrics CSV to {csv_path}\n")

def main():
    overall_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model load
    t0 = time.time()
    model, fe = load_model(device)
    timings = {"Model load": time.time() - t0}

    # Frame extraction
    t0 = time.time()
    frames = extract_frames(VIDEO_PATH, NUM_FRAMES)
    timings["Extract frames"] = time.time() - t0

    # Segmentation & analysis
    t0 = time.time()
    data = segment_and_analyze(frames, model, fe, device)
    timings["Segmentation & analysis"] = time.time() - t0
    
    # PDF generation
    t0 = time.time()
    create_pdf(OUTPUT_PDF, data, timings)
    timings["PDF generation"] = time.time() - t0
    
    # CSV generation
    t0 = time.time()
    save_metrics_csv(data, OUTPUT_CSV)
    timings["CSV generation"] = time.time() - t0

    # Print out all timings
    total = time.time() - overall_start
    print("\nStage timings:")
    for stage, t in timings.items():
        print(f"  {stage}: {t:.2f}s")
    print(f"Total elapsed time: {total:.2f}s")


if __name__ == "__main__":
    main()
