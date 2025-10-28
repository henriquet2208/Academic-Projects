import cv2
import numpy as np
import re
from pathlib import Path

def process_video(video_path: Path,
                  original_folder: Path,
                  masks_folder: Path) -> None:
    """
    Processes a single video, detects orange sail stripes on 20 sample frames,
    saves binary masks with only the detected lines to masks_folder and
    overlays those lines on the original frames in original_folder.
    """
    print(f"\n=== Starting analysis for: {video_path.name} ===")

    original_folder.mkdir(parents=True, exist_ok=True)
    masks_folder.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path.name}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    duration_sec = total_frames / fps if fps > 0 else 0
    num_shots = 20
    if total_frames < num_shots:
        print(f"Warning: {video_path.name} has fewer than {num_shots} frames, skipping.")
        cap.release()
        return

    interval = duration_sec / num_shots
    ConPolyValue = 50
    blur_threshold = 0

    for i in range(num_shots):
        timestamp   = i * interval
        frame_index = min(int(round(timestamp * fps)), total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  [Frame {i+1}] empty at index {frame_index}, skipping")
            continue

        # — HSV-based stripe detection up to dilated mask —
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

        # — find contours on the dilated mask —
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[2], reverse=True)
        top_contours = contours[:3]

        # — build a clean mask showing only those 3 contour lines —
        mask_lines = np.zeros_like(mask)
        for c in top_contours:
            eps = 0.0001 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            cv2.drawContours(mask_lines, [approx], -1, 255, thickness=2)

        mask_fname = f"mask_{i+1:02d}_{timestamp:.2f}s.png"
        cv2.imwrite(str(masks_folder / mask_fname), mask_lines)

        # — overlay the same 3 lines on the original image —
        overlay = frame.copy()
        for c in top_contours:
            eps = 0.0001 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps, True)
            cv2.drawContours(overlay, [approx], -1, (0, 255, 0), 3)

        overlay_fname = f"overlay_{i+1:02d}_{timestamp:.2f}s.png"
        cv2.imwrite(str(original_folder / overlay_fname), overlay)

        print(f"  [Frame {i+1}/{num_shots}] idx={frame_index} saved mask & overlay")

    cap.release()
    print(f"=== Finished analysis for: {video_path.name} ===\n")


def main():
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        pass

    video_dir  = Path(r"C:/Users/asus/OneDrive - Instituto Superior de Engenharia do Porto/Desktop/teseHT/screenshotdosvideos/videos")
    orig_base  = Path("original_images")
    masks_base = Path("Masks")
    orig_base.mkdir(parents=True, exist_ok=True)
    masks_base.mkdir(parents=True, exist_ok=True)

    video_list = sorted(
        video_dir.glob("*.mp4"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)) if re.search(r"(\d+)", p.stem) else 0
    )

    if not video_list:
        print(f"No .mp4 files found in: {video_dir}")
        return

    print(f"\n=== Processing {len(video_list)} videos ===")
    for video_path in video_list:
        process_video(video_path, orig_base / video_path.stem, masks_base / video_path.stem)

if __name__ == "__main__":
    main()
