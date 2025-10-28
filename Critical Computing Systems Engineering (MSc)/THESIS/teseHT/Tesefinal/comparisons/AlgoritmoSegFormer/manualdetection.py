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

# aerodynamic_utils functions
from aerodynamic_utils import (
    Point,
    compute_line_properties,
    calculate_inclination,
    calculate_point_of_interception,
    calculate_camber_max_point_by_tan,
    calculate_interception_point_camber,
    calculate_camber,
    calculate_draft_right,
    calculate_twist,
)

# Suppress transformer and matplotlib warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ── Configuration ───────────────────────────────────────────────────────────
VIDEO_PATH  = r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/screenshotdosvideos/videos/Video 2.MP4"
NUM_FRAMES  = 20
TILE_SIZE   = (512, 512)
ARCH        = "mit-b1"
CHECKPOINT  = r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/Tesefinal/trainingmodel/segformer_final_mitb1.pth"
ROOT_DIR    = r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/Tesefinal/trainingmodel/sail_dataset"
PREVIEW_MAX_W, PREVIEW_MAX_H = 1920, 1080

# ── Model Loader ────────────────────────────────────────────────────────────
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

# ── Frame Extraction ────────────────────────────────────────────────────────
def extract_frames(path, n):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs  = np.linspace(0, total-1, n, dtype=int)
    frames = []
    for i in tqdm(idxs, desc="Extracting frames"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

# ── Segmentation & Properties ──────────────────────────────────────────────
def segment_and_compute(frames, model, fe, device):
    results = []
    for frame in tqdm(frames, desc="Segment & Compute"):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        H,W = rgb.shape[:2]
        pred = np.zeros((H,W), dtype=np.uint8)
        # tile-based segmentation
        for y in range(0, H, TILE_SIZE[0]):
            for x in range(0, W, TILE_SIZE[1]):
                y1,y2 = y, min(y+TILE_SIZE[0], H)
                x1,x2 = x, min(x+TILE_SIZE[1], W)
                crop = rgb[y1:y2, x1:x2]
                inp  = fe(images=crop, return_tensors="pt").to(device)
                with torch.no_grad(): logits = model(**inp).logits
                up = F.interpolate(logits, size=(y2-y1,x2-x1), mode="bilinear", align_corners=False)
                pred[y1:y2, x1:x2] = up.argmax(1).squeeze().cpu().numpy()
        props = compute_line_properties(pred)
        overlay = frame.copy()
        mask_c = np.zeros_like(frame)
        mask_c[pred==1] = (0,255,0)
        overlay = cv2.addWeighted(frame,0.6, mask_c,0.4,0)
        results.append((overlay, props))
    return results

# ── Matplotlib Helpers ─────────────────────────────────────────────────────
def get_scaled_image(img):
    h,w = img.shape[:2]
    scale = min(PREVIEW_MAX_W/w, PREVIEW_MAX_H/h, 1.0)
    if scale<1.0:
        img_s = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    else:
        img_s = img.copy()
    return img_s, scale

def show_preview_matplotlib(title, image):
    disp, _ = get_scaled_image(image[..., ::-1])
    fig_w, fig_h = disp.shape[1]/100, disp.shape[0]/100
    fig,ax = plt.subplots(figsize=(fig_w,fig_h), dpi=100)
    ax.imshow(disp)
    ax.set_title(title)
    ax.axis('off')
    plt.show()

def get_click_points_matplotlib(title, image, num_points):
    img_rgb = image[..., ::-1]
    disp, scale = get_scaled_image(img_rgb)
    h_disp,w_disp = disp.shape[:2]
    fig_w, fig_h = w_disp/100, h_disp/100
    fig,ax = plt.subplots(figsize=(fig_w,fig_h), dpi=100)
    ax.imshow(disp); ax.set_title(title); ax.axis('off')
    ax.set_xlim(0, w_disp); ax.set_ylim(h_disp, 0)
    coords=[]
    def onclick(event):
        if event.inaxes!=ax or event.xdata is None: return
        x_d = min(max(event.xdata,0), w_disp-1)
        y_d = min(max(event.ydata,0), h_disp-1)
        x_o,y_o = int(x_d/scale), int(y_d/scale)
        coords.append((x_o,y_o))
        ax.plot(x_d,y_d,'rx'); fig.canvas.draw()
        if len(coords)>=num_points:
            fig.canvas.mpl_disconnect(cid); plt.close(fig)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(); return coords

# ── Draw & Manual Correction ─────────────────────────────────────────────────
def draw_and_show_manual(results):
    section_names = ['Top','Mid','Bot']
    metrics = {f'{m}_{s}':[] for m in ('Camber','Draft','Twist') for s in section_names}

    for idx,(overlay,props) in enumerate(results,1):
        # 1) Show detected points
        vis = overlay.copy()
        for sec in props:
            cv2.drawMarker(vis, (int(sec['first'].x), int(sec['first'].y)), (0,165,255),
                           cv2.MARKER_TILTED_CROSS, 12, 2)
            cv2.drawMarker(vis, (int(sec['last'].x), int(sec['last'].y)), (0,165,255),
                           cv2.MARKER_TILTED_CROSS, 12, 2)
            cv2.drawMarker(vis, (int(sec['max_pt'].x), int(sec['max_pt'].y)), (255,0,0),
                           cv2.MARKER_TILTED_CROSS, 12, 2)
        show_preview_matplotlib(f"Frame {idx}: detected", vis)

        # 2) Manual correction (no markers)
        print(f"Frame {idx}: click {len(props)*2} points (2 per section), then close the window.")
        clicks = get_click_points_matplotlib(f"Frame {idx} correction", overlay, len(props)*2)

        if len(clicks)==len(props)*2:
            # update each section
            for i,sec in enumerate(props):
                p1 = Point(*clicks[2*i]); p2 = Point(*clicks[2*i+1])
                first,last = sorted((p1,p2), key=lambda p: p.x)
                # recompute geometry
                coeff = sec['coeff']; poly=np.poly1d(coeff); dpoly=poly.deriv()
                m = calculate_inclination(Point(first.x, poly(first.x)), Point(last.x, poly(last.x)))
                b = calculate_point_of_interception(m, Point(first.x, poly(first.x)))
                max_pt = calculate_camber_max_point_by_tan(poly, dpoly,
                                                           Point(first.x, poly(first.x)),
                                                           Point(last.x, poly(last.x)))
                intercept_pt = calculate_interception_point_camber(max_pt, m, b)
                cam = calculate_camber(poly, max_pt, intercept_pt, first, last)
                dr  = calculate_draft_right(first, last, intercept_pt)
                tw  = calculate_twist(m)
                sec.update({
                    'first': first,
                    'last': last,
                    'm': m,
                    'b': b,
                    'max_pt': max_pt,
                    'intercept_pt': intercept_pt,
                    'camber': cam,
                    'draft': dr,
                    'twist': tw
                })
        else:
            print("Skipped corrections; using original endpoints.")

        # 3) Show corrected geometry
        vis2 = overlay.copy()
        for sec in props:
            cv2.drawMarker(vis2, (int(sec['first'].x), int(sec['first'].y)), (0,165,255),
                           cv2.MARKER_TILTED_CROSS, 12, 2)
            cv2.drawMarker(vis2, (int(sec['last'].x), int(sec['last'].y)), (0,165,255),
                           cv2.MARKER_TILTED_CROSS, 12, 2)
            cv2.drawMarker(vis2, (int(sec['max_pt'].x), int(sec['max_pt'].y)), (255,0,0),
                           cv2.MARKER_TILTED_CROSS, 12, 2)
        show_preview_matplotlib(f"Frame {idx}: corrected", vis2)

        # 4) Print metrics
        c_vals = [round(sec['camber'],2) for sec in props]
        d_vals = [round(sec['draft'],2)  for sec in props]
        t_vals = [round(sec['twist'],2)  for sec in props]
        # ensure length 3
        while len(c_vals)<3:
            c_vals.append(0.0); d_vals.append(0.0); t_vals.append(0.0)
        df = pd.DataFrame({'Camber':c_vals,'Draft':d_vals,'Twist':t_vals}, index=section_names)
        print(f"\nFrame {idx} metrics:")
        print(df.to_string())

        # accumulate
        for name,c,d,t in zip(section_names, c_vals, d_vals, t_vals):
            metrics[f"Camber_{name}"].append(c)
            metrics[f"Draft_{name}"].append(d)
            metrics[f"Twist_{name}"].append(t)

    # Save CSV
    df_out = pd.DataFrame(metrics).T
    df_out.columns = [f"Frame_{i}" for i in range(1, len(results)+1)]
    df_out = df_out.reset_index().rename(columns={"index":"Metric"})
    out_dir = Path("manualground-truth_Parameters"); out_dir.mkdir(exist_ok=True)
    path = out_dir/"video2_chordpoints_manual.csv"
    df_out.to_csv(path, index=False, float_format="%.2f")
    print(f"\nSaved parameters to {path}")

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model,fe = load_model(device)
    frames = extract_frames(VIDEO_PATH, NUM_FRAMES)
    results = segment_and_compute(frames, model, fe, device)
    draw_and_show_manual(results)

if __name__ == "__main__":
    main()