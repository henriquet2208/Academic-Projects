# batch_segmentation_timing_processing_only.py
import re
import time
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from tqdm import tqdm
import warnings, logging, csv

# ── Config ──────────────────────────────────────────────────────────────────
VIDEO_DIR   = Path(r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/screenshotdosvideos/videos")
NUM_FRAMES  = 20
TILE_SIZE   = (512, 512)
ARCH        = "mit-b1"  # mudar para "mit-b2"/"mit-b3" se quiseres
CHECKPOINT  = Path(r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/Tesefinal/trainingmodel/segformer_final_mitb1.pth") #mudar para "mit-b2"/"mit-b3"
ROOT_DIR    = Path(r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/Tesefinal/trainingmodel/sail_dataset")
ITERATIONS  = 100

PERF_CSV    = Path("processingtimesegformer_mitb1_processing_only.csv") #mudar para "mit-b2"/"mit-b3"
AVG_CSV     = Path("processingtimesegformer_mitb1_processing_only_avg.csv") #mudar para "mit-b2"/"mit-b3"
ORIG_DIR    = Path("Original_Images_mitb1_final") #mudar para "mit-b2"/"mit-b3"
MASK_DIR    = Path("Masks_mitb1_final") #mudar para "mit-b2"/"mit-b3"

SAVE_OUTPUT = True   # ← guardamos overlays/máscaras (fora do tempo)
VERBOSE     = False  # menos prints durante benchmark

for d in (ORIG_DIR, MASK_DIR):
    d.mkdir(parents=True, exist_ok=True)

logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

# ── Helpers ─────────────────────────────────────────────────────────────────
def load_model(device):
    df = pd.read_csv(ROOT_DIR / "class_dict_seg.csv")
    id2label = df["name"].to_dict()
    label2id = {v: k for k, v in id2label.items()}
    proc = SegformerImageProcessor()
    model = SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/{ARCH}", num_labels=len(id2label), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True, reshape_last_stage=True
    ).to(device)
    if CHECKPOINT.exists():
        model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    model.eval()
    return model, proc

def extract_frames(path, n):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if ret:
            frames.append((int(i), frame))
    cap.release()
    return frames

def natural_sort(videos):
    return sorted(videos, key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))

# ── Core loop (processing-only; I/O fora do tempo) ──────────────────────────
def run_iterations(model, proc, video_frames, fps_map, device):
    times = {name: [] for name in video_frames}

    # cabeçalhos do CSV de iterações
    with open(PERF_CSV, 'w', newline='') as f:
        csv.writer(f).writerow(['iteration', 'video', 'processing_time_s'])

    for it in tqdm(range(1, ITERATIONS + 1), desc="Iterations", unit="iter"):
        if VERBOSE: print(f"\n--- Iteration {it}/{ITERATIONS} ---")

        for name, frames in video_frames.items():
            # buffers para guardar depois (fora do tempo)
            overlays_to_save = []
            masks_to_save = []
            fnames_to_save = []

            # ⏱️ cronometrar APENAS processamento (inferência + overlay/máscara em memória)
            start = time.perf_counter()

            for f_idx, (idx, frame) in enumerate(frames, 1):
                H, W = frame.shape[:2]
                pred = np.zeros((H, W), dtype=np.uint8)
                rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # tiling + inferência
                for y in range(0, H, TILE_SIZE[0]):
                    for x in range(0, W, TILE_SIZE[1]):
                        y1, y2 = y, min(y + TILE_SIZE[0], H)
                        x1, x2 = x, min(x + TILE_SIZE[1], W)
                        tile = rgb[y1:y2, x1:x2]

                        inp = proc(images=tile, return_tensors="pt").to(device)
                        with torch.inference_mode():
                            logits = model(**inp).logits
                            up = F.interpolate(
                                logits, size=(y2 - y1, x2 - x1),
                                mode="bilinear", align_corners=False
                            )
                            cls = up.argmax(1).squeeze().detach().cpu().numpy()

                        pred[y1:y2, x1:x2] = cls.astype(np.uint8)

                # construir overlay/máscara EM MEMÓRIA (não guardar ainda)
                mask_img = (pred * 255).astype(np.uint8)
                overlay  = cv2.addWeighted(
                    frame, 0.6,
                    cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR), 0.4, 0
                )

                if SAVE_OUTPUT and it == 1:
                    # mesmo naming/timestamp de antes
                    fps = fps_map.get(name) or 0.0
                    timestamp = (idx / fps) if fps > 0 else 0.0
                    fname = f"frame{f_idx}({timestamp:.3f}s).png"
                    overlays_to_save.append(overlay)
                    masks_to_save.append(mask_img)
                    fnames_to_save.append(fname)

            # garantir que a GPU acabou antes de parar o relógio
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            times[name].append(elapsed)

            # registo por iteração
            with open(PERF_CSV, 'a', newline='') as f:
                csv.writer(f).writerow([it, name, f"{elapsed:.3f}"])

            # 🔽 guardar overlays/máscaras FORA do tempo (como antes)
            if SAVE_OUTPUT and it == 1 and overlays_to_save:
                vid_key = Path(name).stem.replace(' ', '').lower()
                (ORIG_DIR / vid_key).mkdir(parents=True, exist_ok=True)
                (MASK_DIR / vid_key).mkdir(parents=True, exist_ok=True)
                for ov, mk, fname in zip(overlays_to_save, masks_to_save, fnames_to_save):
                    cv2.imwrite(str(ORIG_DIR / vid_key / fname), ov)
                    cv2.imwrite(str(MASK_DIR / vid_key / fname), mk)

            if VERBOSE:
                print(f"Finished {name}: {elapsed:.3f}s")

    return times

def write_avg_csv(times):
    with open(AVG_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['video', 'avg_time_video_s', 'avg_time_frame_s'])
        for name, vals in times.items():
            if not vals:
                continue
            avg_video = sum(vals) / len(vals)
            avg_frame = avg_video / NUM_FRAMES
            writer.writerow([name, f"{avg_video:.3f}", f"{avg_frame:.3f}"])

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model...")
    model, proc = load_model(device)

    vids = natural_sort(list(VIDEO_DIR.glob("*.mp4")))
    print(f"Found {len(vids)} videos: {[v.name for v in vids]}")

    # descodificação/extração de frames — FORA do tempo
    fps_map = {}
    video_frames = {}
    for vid in vids:
        cap = cv2.VideoCapture(str(vid))
        fps_map[vid.name] = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        video_frames[vid.name] = extract_frames(vid, NUM_FRAMES)

    times = run_iterations(model, proc, video_frames, fps_map, device)
    write_avg_csv(times)

    print(f"\n=== Completed {ITERATIONS} iterations ===")
    print(f"Master timings: {PERF_CSV}")
    print(f"Average timings: {AVG_CSV}")

if __name__ == "__main__":
    main()
