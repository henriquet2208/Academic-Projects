import cv2
import numpy as np
import time
import csv
import re
from pathlib import Path
from typing import List, Tuple

"""
Purpose:
    Benchmark a traditional (non-AI) image-processing pipeline over videos by:
      - Sampling a fixed number of frames (NUM_SHOTS) uniformly across each video.
      - Running a color-space + blur + threshold + morphology + contour pipeline.
      - Measuring total processing time per video for the sampled frames only.
      - Repeating the benchmark for a fixed number of iterations to reduce noise.
      - Writing detailed per-iteration times and per-video averages to CSV.

What it does:
    1) For each .mp4 in the configured folder, extract NUM_SHOTS frames spaced
       uniformly in time (without saving the frames to disk).
    2) For each sampled frame, run the traditional processing steps:
         - BGR->RGB->HSV, hue shift, Gaussian blur, threshold, inRange mask,
           dilation, contour detection, polygonal approximation, and drawing.
       Timing includes only the processing (not frame extraction).
    3) Repeat for N iterations, write:
         - "overlay_time_all_processing_only.csv": (iteration, video, time_s)
         - "overlay_time_avg_processing_only.csv": per-video averages
           (avg time per video and per frame).
    4) Designed to compare relative performance of the traditional pipeline
       across runs/videos; not intended to change video files.

Notes:
    - If a video has fewer frames than NUM_SHOTS, it is skipped (warning shown).
    - Processing time is the sum across sampled frames for that video.
    - Ensure OpenCV is installed (cv2), and videos are readable.
"""

SAVE_OUTPUT = False
NUM_SHOTS   = 20   # nº de frames amostrados por vídeo

def extract_frames_for_samples(video_path: Path, num_shots: int) -> Tuple[List[Tuple[int, float]], List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path.name}")
        return [], [], 0.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    if total_frames < num_shots:
        print(f"Warning: {video_path.name} has fewer than {num_shots} frames, skipping.")
        cap.release()
        return [], [], fps

    duration_sec = total_frames / fps if fps > 0 else 0.0
    interval = duration_sec / num_shots

    sample_info, frames = [], []
    for i in range(num_shots):
        timestamp   = i * interval
        frame_index = min(int(round(timestamp * fps)), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        sample_info.append((frame_index, timestamp))
        frames.append(frame)

    cap.release()
    return sample_info, frames, fps


def process_frames_only(frames: List[np.ndarray]) -> float:
    total_processing_time = 0.0
    ConPolyValue = 50
    blur_threshold = 0

    for frame in frames:
        t0 = time.perf_counter()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        h = h.astype(np.int16) + 30
        h[h >= 180] = 0
        h[h <   0] = 180
        hsv = cv2.merge((h.astype(np.uint8), s, v))

        if blur_threshold % 2 == 0:
            blur_threshold += 1
        blur = cv2.GaussianBlur(hsv, (blur_threshold, blur_threshold), 3)

        _, thresh = cv2.threshold(blur, ConPolyValue, 255, cv2.THRESH_BINARY)
        rgb_thresh = cv2.cvtColor(thresh, cv2.COLOR_HSV2RGB)
        mask = cv2.inRange(rgb_thresh, (255, 0, 0), (255, 255, 0))

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(mask, kernel)

        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[2], reverse=True)
        top_contours = contours[:3]

        mask_lines = np.zeros_like(mask)
        for c in top_contours:
            eps = 0.0001 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            cv2.drawContours(mask_lines, [approx], -1, 255, thickness=2)

        overlay = frame.copy()
        for c in top_contours:
            eps = 0.0001 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 3)

        t1 = time.perf_counter()
        total_processing_time += (t1 - t0)

    return total_processing_time


def process_video_processing_only(video_path: Path) -> float:
    sample_info, frames, fps = extract_frames_for_samples(video_path, NUM_SHOTS)
    if not frames:
        return 0.0
    return process_frames_only(frames)


def main():
    video_dir  = Path(r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/screenshotdosvideos/videos")

    iterations = 100
    master_csv = Path("overlay_time_all_processing_only.csv")
    with open(master_csv, 'w', newline='') as mf:
        csv.writer(mf).writerow(['iteration', 'video', 'processing_time_s'])

    times_acc = {}

    video_list = sorted(
        video_dir.glob("*.mp4"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1))
    )

    for it in range(1, iterations + 1):
        for video_path in video_list:
            vid_name = video_path.name
            t = process_video_processing_only(video_path)

            with open(master_csv, 'a', newline='') as mf:
                csv.writer(mf).writerow([it, vid_name, f"{t:.3f}"])

            times_acc.setdefault(vid_name, []).append(t)

    avg_csv = Path("overlay_time_avg_processing_only.csv")
    with open(avg_csv, 'w', newline='') as af:
        writer = csv.writer(af)
        writer.writerow(['video', 'avg_time_video_s', 'avg_time_frame_s'])
        for vid, lst in times_acc.items():
            avg_video = sum(lst) / len(lst)
            avg_frame = avg_video / NUM_SHOTS
            writer.writerow([vid, f"{avg_video:.3f}", f"{avg_frame:.3f}"])

if __name__ == "__main__":
    main()
