import os
import time
import warnings
import logging
import random
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
import albumentations as A
from albumentations.core.transforms_interface import DualTransform

# ---------------------------
# User‐set Hyperparameters
# ---------------------------
ARCH = "mit-b1"  # options: mit-b2, mit-b3, mit-b4, mit-b5
INITIAL_LR = 5e-5
BATCH_SIZE = 4
NUM_EPOCHS = 30
DICE_WEIGHT = 0.7
FOCAL_WEIGHT = 0.3
CROP_SIZE = 512
ROOT_DIR = "sail_dataset"
CHECKPOINT_F = "checkpoint_mitb1.pth"
FINAL_MODEL_F = "segformer_final_mitb1.pth"


# ---------------------------
# Loss Definitions
# ---------------------------
class DiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0, ignore_index=-100):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)[:, 1]
        mask = (targets != self.ignore_index)
        t = (targets == 1).float()
        p = probs * mask.float()
        inter = torch.sum(p * t)
        denom = torch.sum(p) + torch.sum(t)
        dice = (2 * inter + self.smooth) / (denom + self.smooth)
        return 1 - dice


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        B, C, H, W = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(-1, C)
        targets = targets.view(-1)
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction="none"
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ---------------------------
# Line‐biased Crop Transform
# ---------------------------
class LineBiasedCrop(DualTransform):
    def __init__(self, height, width, p=0.5):
        super().__init__(p=p)
        self.height, self.width = height, width

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        H, W = mask.shape[:2]
        if np.random.rand() < self.p and mask.sum() > 0:
            ys, xs = np.where(mask == 1)
            idx = np.random.randint(len(ys))
            cy, cx = ys[idx], xs[idx]
            y0 = np.clip(cy - self.height // 2, 0, H - self.height)
            x0 = np.clip(cx - self.width // 2, 0, W - self.width)
        else:
            y0 = np.random.randint(0, H - self.height + 1)
            x0 = np.random.randint(0, W - self.width + 1)
        return {"y0": y0, "x0": x0}

    def apply(self, img, y0=0, x0=0, **kwargs):
        return img[y0:y0 + self.height, x0:x0 + self.width]

    def apply_to_mask(self, mask, y0=0, x0=0, **kwargs):
        return mask[y0:y0 + self.height, x0:x0 + self.width]

    @property
    def targets(self):
        return {"image": self.apply, "mask": self.apply_to_mask}

    def get_transform_init_args_names(self):
        return ("height", "width", "p")


# ---------------------------
# Dataset
# ---------------------------
class ImageSegmentationDataset(Dataset):
    def __init__(self, root_dir, fe, transforms=None, train=True):
        self.fe = fe
        self.train = train
        sub = "train" if train else "test"
        self.img_dir = os.path.join(root_dir, "images", sub)
        self.ann_dir = os.path.join(root_dir, "mask", sub)
        self.images = sorted(os.listdir(self.img_dir))
        self.masks = sorted(os.listdir(self.ann_dir))
        assert len(self.images) == len(self.masks)
        self.transforms = transforms
        self.dilate_kernel = np.ones((3, 3), np.uint8)

    def __len__(self):
        return len(self.images)

    @staticmethod
    def _recolor_white_to_orange(img, mask, fraction=0.7):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        ys, xs = np.where(mask == 1)
        if len(ys) > 0:
            n = int(len(ys) * fraction)
            sel = np.random.choice(len(ys), size=n, replace=False)
            hsv[ys[sel], xs[sel], 0] = 20  # Hue ≈20° for orange
            hsv[ys[sel], xs[sel], 1] = 255  # Max saturation
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def __getitem__(self, idx):
        img = cv2.cvtColor(
            cv2.imread(os.path.join(self.img_dir, self.images[idx])),
            cv2.COLOR_BGR2RGB
        )
        m = cv2.imread(os.path.join(self.ann_dir, self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        m[m == 255] = 1
        m[(m != 0) & (m != 1)] = 0
        m = cv2.dilate(m, self.dilate_kernel, iterations=1)

        # synthetic recolor 50% of train samples
        if self.train and random.random() < 0.5:
            img = self._recolor_white_to_orange(img, m)

        if self.transforms:
            aug = self.transforms(image=img, mask=m)
            img, m = aug["image"], aug["mask"]

        enc = self.fe(images=img, segmentation_maps=m, return_tensors="pt")
        enc = {k: v.squeeze(0) for k, v in enc.items()}
        enc["labels"] = enc["labels"].long()
        return enc


# ---------------------------
# Main
# ---------------------------
def main():
    # setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    logging.getLogger("transformers").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")

    print("Using device:", device)

    # class mapping
    df = pd.read_csv(os.path.join(ROOT_DIR, "class_dict_seg.csv"))
    classes = df["name"]
    id2label = classes.to_dict()
    label2id = {v: k for k, v in id2label.items()}

    # transforms & loaders
    fe = SegformerImageProcessor()

    train_tf = A.Compose([
        LineBiasedCrop(CROP_SIZE, CROP_SIZE, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=0.1, rotate=(-20, 20), p=0.7),
        A.ColorJitter(0.2, 0.2, 0.2, 0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=50, val_shift_limit=30, p=0.7),
        A.GaussianBlur(blur_limit=3, p=0.3),
    ])

    val_tf = A.Compose([A.Resize(CROP_SIZE, CROP_SIZE)])

    train_ds = ImageSegmentationDataset(ROOT_DIR, fe, train_tf, train=True)
    val_ds = ImageSegmentationDataset(ROOT_DIR, fe, val_tf, train=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    # model & optimizer
    model = SegformerForSemanticSegmentation.from_pretrained(
        f"nvidia/{ARCH}",
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        reshape_last_stage=True
    ).to(device)

    for _, p in model.segformer.encoder.named_parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(
        [{"params": [p for p in model.parameters() if p.requires_grad], "lr": INITIAL_LR}],
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    scaler = GradScaler()

    dice_loss = DiceLoss(ignore_index=-100)
    focal_loss = FocalLoss(weight=torch.tensor([0.05, 0.95], device=device), ignore_index=-100)

    # optionally resume
    start_epoch = 1
    if os.path.exists(CHECKPOINT_F):
        ckpt = torch.load(CHECKPOINT_F, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"››› Resuming from epoch {start_epoch}")

    # training loop
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        inter, union, denom = {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}
        train_true_line = 0
        train_total_valid = 0
        train_total_correct = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} Train", leave=False):
            x = batch["pixel_values"].to(device)
            y = batch["labels"].to(device)
            y[y == 255] = -100

            optimizer.zero_grad()
            with autocast():
                up = F.interpolate(
                    model(pixel_values=x).logits,
                    size=y.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                loss = DICE_WEIGHT * dice_loss(up, y) + FOCAL_WEIGHT * focal_loss(up, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * x.size(0)
            preds = up.argmax(1)
            valid = (y != -100)
            train_total_valid += valid.sum().item()
            train_total_correct += ((preds == y) & valid).sum().item()
            train_true_line += ((y == 1) & valid).sum().item()

            for cls in [0, 1]:
                pm = ((preds == cls) & valid)
                tm = ((y == cls) & valid)
                inter[cls] += (pm & tm).sum().item()
                union[cls] += (pm | tm).sum().item()
                denom[cls] += pm.sum().item() + tm.sum().item()

        train_loss = running_loss / len(train_ds)
        train_iou_line = inter[1] / union[1] if union[1] > 0 else 0.0
        train_dice_line = 2 * inter[1] / denom[1] if denom[1] > 0 else 1.0
        train_acc_line = inter[1] / train_true_line if train_true_line > 0 else 0.0
        train_acc_global = train_total_correct / train_total_valid if train_total_valid > 0 else 0.0

        # validation
        model.eval()
        v_loss = 0.0
        v_inter, v_union, v_denom = {0: 0, 1: 0}, {0: 0, 1: 0}, {0: 0, 1: 0}
        val_true_line = 0
        val_total_valid = 0
        val_total_correct = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} Val ", leave=False):
                x = batch["pixel_values"].to(device)
                y = batch["labels"].to(device)
                y[y == 255] = -100

                up = F.interpolate(
                    model(pixel_values=x).logits,
                    size=y.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )
                loss = DICE_WEIGHT * dice_loss(up, y) + FOCAL_WEIGHT * focal_loss(up, y)

                v_loss += loss.item() * x.size(0)
                preds = up.argmax(1)
                valid = (y != -100)
                val_total_valid += valid.sum().item()
                val_total_correct += ((preds == y) & valid).sum().item()
                val_true_line += ((y == 1) & valid).sum().item()

                for cls in [0, 1]:
                    pm = ((preds == cls) & valid)
                    tm = ((y == cls) & valid)
                    v_inter[cls] += (pm & tm).sum().item()
                    v_union[cls] += (pm | tm).sum().item()
                    v_denom[cls] += pm.sum().item() + tm.sum().item()

        val_loss = v_loss / len(val_ds)
        val_iou_line = v_inter[1] / v_union[1] if v_union[1] > 0 else 0.0
        val_dice_line = 2 * v_inter[1] / v_denom[1] if v_denom[1] > 0 else 1.0
        val_acc_line = v_inter[1] / val_true_line if val_true_line > 0 else 0.0
        val_acc_global = val_total_correct / val_total_valid if val_total_valid > 0 else 0.0

        scheduler.step(val_loss)

        # logging & checkpoint save
        epoch_time = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0

        print(f"\n=== Epoch {epoch}/{NUM_EPOCHS} — "
              f"LR {lr:.2e} — {epoch_time:.1f}s — PeakMem {peak_mem:.2f} GB ===")
        print(f" Train | Loss {train_loss:.4f} | Acc {train_acc_global:.4f} | "
              f"IoU[line]={train_iou_line:.4f} | Dice[line]={train_dice_line:.4f} | "
              f"Acc[line]={train_acc_line:.4f}")
        print(f" Val | Loss {val_loss:.4f} | Acc {val_acc_global:.4f} | "
              f"IoU[line]={val_iou_line:.4f} | Dice[line]={val_dice_line:.4f} | "
              f"Acc[line]={val_acc_line:.4f}\n")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
        }, CHECKPOINT_F)

        torch.cuda.empty_cache()

    # final save
    torch.save(model.state_dict(), FINAL_MODEL_F)
    print("Training complete — model saved to", FINAL_MODEL_F)


if __name__ == "__main__":
    main()
