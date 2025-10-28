#!/usr/bin/env python3
"""
view_segformer_video.py: Semantic segmentation and line property visualization for videos.

Usage:
    python view_segformer_video.py [input_video] [output_video]
    (defaults to Video 1.MP4 under the script's Tese/videos directory)
"""
import os
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import time
import warnings
import logging
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from Tesefinal.comparisons.AlgoritmoSegFormer.aerodynamic_utils import compute_line_properties


def load_model(root_dir, arch, checkpoint_path, device):
    df = pd.read_csv(os.path.join(root_dir, "class_dict_seg.csv"))
    classes = df["name"]
    id2label = classes.to_dict()
    label2id = {v: k for k, v in id2label.items()}

    fe = SegformerImageProcessor()
    model = SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/{arch}",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        reshape_last_stage=True
    ).to(device)
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
    model.eval()
    return model, fe


def predict_frame(img, model, fe, device, tile_size):
    H, W, _ = img.shape
    ch, cw = tile_size
    pred = np.zeros((H, W), dtype=np.uint8)
    for y in range(0, H, ch):
        for x in range(0, W, cw):
            y1, y2 = y, min(y + ch, H)
            x1, x2 = x, min(x + cw, W)
            tile = img[y1:y2, x1:x2]
            inp = fe(images=tile, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**inp).logits
                up = F.interpolate(
                    logits,
                    size=(y2 - y1, x2 - x1),
                    mode="bilinear",
                    align_corners=False
                )
                pred[y1:y2, x1:x2] = up.argmax(1).squeeze().cpu().numpy()
    return pred


def overlay_results(frame, mask_pred, line_props):
    color_mask = np.zeros_like(frame)
    color_mask[mask_pred == 1] = (0, 255, 0)
    overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
    for i, p in enumerate(line_props):
        f0 = p["first"]
        l0 = p["last"]
        cv2.line(overlay, (int(f0.x), int(f0.y)), (int(l0.x), int(l0.y)), (0, 0, 255), 2)
        text = f"L{i+1}: C={p['camber']:.1f}% D={p['draft']:.1f}% T={p['twist']:.1f}°"
        cv2.putText(overlay, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    return overlay


def process_video(input_path, output_path, model, fe, device, tile_size, num_lines):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {input_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {total_frames} frames at {fps:.1f} FPS ({width}x{height})...")
    start_time = time.time()
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mask_pred = predict_frame(rgb, model, fe, device, tile_size)
        props = compute_line_properties(mask_pred, num_lines=num_lines, poly_deg=2)
        annotated = overlay_results(frame, mask_pred, props)
        out.write(annotated)
        if frame_idx % 200 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames")

    cap.release()
    out.release()
    elapsed = time.time() - start_time
    print(f"Output saved to {output_path}")
    print(f"Total processing time: {elapsed:.2f} seconds")
    return elapsed


def main():
    # determine default paths relative to script location
    script_dir = os.path.abspath(os.path.dirname(__file__))
    default_input = os.path.join(script_dir, "Tese", "videos", "Video 1.MP4")
    default_output = os.path.join(script_dir, "Video1_annotated.mp4")

    parser = argparse.ArgumentParser(description="Video semantic segmentation viewer")
    parser.add_argument("input_video", nargs='?', default=default_input,
                        help=f"Input video path (default: {default_input})")
    parser.add_argument("output_video", nargs='?', default=default_output,
                        help=f"Output video path (default: {default_output})")
    parser.add_argument("--root_dir", default="sail_dataset", help="Dataset root directory")
    parser.add_argument("--arch", default="mit-b3", help="Segformer architecture")
    parser.add_argument("--checkpoint", default="segformer_final.pth", help="Model checkpoint path")
    parser.add_argument("--tile_height", type=int, default=512, help="Tile height for inference")
    parser.add_argument("--tile_width", type=int, default=512, help="Tile width for inference")
    parser.add_argument("--num_lines", type=int, default=3, help="Number of lines to analyze per frame")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    model, fe = load_model(args.root_dir, args.arch, args.checkpoint, device)
    process_video(
        args.input_video,
        args.output_video,
        model,
        fe,
        device,
        tile_size=(args.tile_height, args.tile_width),
        num_lines=args.num_lines
    )

if __name__ == "__main__":
    main()
