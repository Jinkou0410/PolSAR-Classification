# run_experiments_xian.py
# Python 3.x, PyTorch 2.x, etc.
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import pandas as pd
import cv2
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import gc
import time

BACKGROUND_LABEL = 255
class_names = ["Water", "Vegetation", "Low-Density Urban", "High-Density Urban", "Developed"]
num_classes = len(class_names)

COLOR_MAP = np.array([
    [44, 83, 157], [32, 215, 80], [255, 89, 87], [255, 0, 0], [207, 75, 147], [255, 255, 255]
], dtype=np.uint8)



def label_to_color_bgr(label_map):

    h, w = label_map.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(num_classes):
        color_img[label_map == i] = COLOR_MAP[i]
    color_img[label_map == BACKGROUND_LABEL] = COLOR_MAP[-1]
    return cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

def save_dual_result_map(gt_label, pred_label, save_path):
    gt_bgr = label_to_color_bgr(gt_label)
    pred_bgr = label_to_color_bgr(pred_label)
    h, w, _ = gt_bgr.shape
    padding, header = 20, 60

    canvas = np.ones((h + header, w * 2 + padding * 3, 3), dtype=np.uint8) * 255

    canvas[header:, padding: padding + w] = gt_bgr
    canvas[header:, padding * 2 + w: padding * 2 + w * 2] = pred_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "Ground Truth", (padding + w // 4, 40),
                font, 1.0, (0, 0, 0), 2)
    cv2.putText(canvas, "Prediction", (padding * 2 + w + w // 4, 40),
                font, 1.0, (0, 0, 0), 2)
    cv2.imwrite(save_path, canvas)

def save_prediction_map(pred_label, save_path):

    pred_bgr = label_to_color_bgr(pred_label)
    cv2.imwrite(save_path, pred_bgr)

def predict_full_image(dataset, model, device, config, num_classes, background_label=255):
    H, W = dataset.label.shape
    patch_size = config['patch_size']
    pred_prob = np.zeros((num_classes, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)

    # 高斯权重
    y, x = np.mgrid[0:patch_size, 0:patch_size]
    center = patch_size / 2
    sigma = patch_size / 4
    weight = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))

    model.eval()
    with torch.no_grad():
        stride = patch_size // 4
        coords = [(i, j) for i in range(0, H - patch_size + 1, stride)
                  for j in range(0, W - patch_size + 1, stride)]
        for (i, j) in tqdm(coords, desc="📊 预测中", leave=False):
            main_patch = dataset.main_features[:, i:i+patch_size, j:j+patch_size]
            phy_patch = dataset.phy_features[i:i+patch_size, j:j+patch_size, :]
            phy_patch = np.transpose(phy_patch, (2, 0, 1))

            main_tensor = torch.from_numpy(main_patch).float().unsqueeze(0).to(device)
            phy_tensor = torch.from_numpy(phy_patch).float().unsqueeze(0).to(device)

            if hasattr(model, 'mode') or "MI-SSL" in str(type(model)):
                output = model(main_tensor, phy_tensor, mode='classify')
            else:
                output = model(main_tensor, phy_tensor)

            if isinstance(output, tuple):
                logits = output[0]
            else:
                logits = output

            if logits.dim() == 2:
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                for c in range(num_classes):
                    pred_prob[c, i:i+patch_size, j:j+patch_size] += probs[c] * weight
                count_map[i:i+patch_size, j:j+patch_size] += weight
            else:
                if logits.shape[2:] != (patch_size, patch_size):
                    logits = F.interpolate(logits, size=(patch_size, patch_size),
                                           mode='bilinear', align_corners=False)
                probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                for c in range(num_classes):
                    pred_prob[c, i:i+patch_size, j:j+patch_size] += probs[c] * weight
                count_map[i:i+patch_size, j:j+patch_size] += weight

    pred_labels = np.argmax(pred_prob / (count_map + 1e-6), axis=0).astype(np.uint8)
    for _ in range(2):
        pred_labels = cv2.medianBlur(pred_labels, 3)
    pred_labels[dataset.label == background_label] = background_label
    gc.collect()
    return pred_labels

def run_experiment(exp_name, model_class, config, device, paths):

    exp_type = exp_name.split('_')[0]
    print(f"\n🚀 [启动实验] {exp_name}")
    save_root = "./SF_Experiments"

    dataset = PolSARDataset(
        paths['coherency'],
        paths['freeman'],
        paths['label'],
        patch_size=config['patch_size']
    )

    label_map = dataset.label
    valid_pos = np.where(label_map != BACKGROUND_LABEL)
    valid_coords = np.array(list(zip(valid_pos[0], valid_pos[1])))
    num_total_valid = len(valid_coords)
    num_sampled = int(num_total_valid * config.get('sample_ratio', 0.01))

    valid_labels = label_map[valid_pos]
    cls_count = np.bincount(valid_labels, minlength=num_classes)

    labels_at_coords = valid_labels
    counts = np.bincount(labels_at_coords, minlength=num_classes)
    weights = len(valid_coords) / (counts + 1e-6)  # 类频率倒数
    sample_weights = np.array([weights[l] for l in labels_at_coords])
    sample_weights /= sample_weights.sum()
    selected_idx = np.random.choice(len(valid_coords), size=num_sampled,
                                    replace=False, p=sample_weights)
    sampled_coords = valid_coords[selected_idx]

    H, W = label_map.shape
    training_coords = []
    for (r, c) in sampled_coords:
        ri = max(0, min(r - config['patch_size']//2, H - config['patch_size']))
        ci = max(0, min(c - config['patch_size']//2, W - config['patch_size']))
        training_coords.append((ri, ci))
    training_coords = list(set(training_coords))
    patch_labels = []
    for (i, j) in training_coords:
        mi = i + config['patch_size']//2
        mj = j + config['patch_size']//2
        lbl = label_map[mi, mj] if (0 <= mi < H and 0 <= mj < W) else BACKGROUND_LABEL
        patch_labels.append(lbl)
    valid_idx = [k for k,lbl in enumerate(patch_labels) if lbl != BACKGROUND_LABEL]
    training_coords = [training_coords[k] for k in valid_idx]
    patch_labels = [patch_labels[k] for k in valid_idx]
    dataset.coords = training_coords

    class_counts = np.bincount(patch_labels, minlength=num_classes)
    sample_weights_patch = [1.0 / (class_counts[lbl] + 1e-6) for lbl in patch_labels]
    sampler = WeightedRandomSampler(weights=sample_weights_patch,
                                    num_samples=len(training_coords),
                                    replacement=True)
    dataloader = DataLoader(dataset, batch_size=config['batch_size'],
                            sampler=sampler, num_workers=0, shuffle=False)

    # 模型实例化
    if model_class == PolSARPhyNet:
        model = model_class(num_classes=num_classes,
                            use_smpc=config.get('use_smpc', False),
                            use_dpi=config.get('use_dpi', False),
                            use_ccam=config.get('use_ccam', False),
                            patch_size=config['patch_size']).to(device)
    else:
        model = model_class(in_channels=3, num_classes=num_classes).to(device)


    if config.get('use_pmsc', False):
        optimizer_pre = optim.AdamW(model.parameters(), lr=1e-3)
        start_pre = time.time()
        for epoch in tqdm(range(20), desc="Phy Pre", leave=False):
            model.train()
            for batch in dataloader:
                main = batch['main'].to(device)
                phy = batch['phy'].to(device)
                phy_gt = batch['phy_gt'].to(device)
                _, phy_pred = model(main, phy)
                if phy_pred.shape[2:] != phy_gt.shape[1:]:
                    phy_pred = F.interpolate(phy_pred, size=phy_gt.shape[1:], mode='bilinear')
                loss_p = F.cross_entropy(phy_pred, phy_gt, ignore_index=255)
                optimizer_pre.zero_grad()
                loss_p.backward()
                optimizer_pre.step()
        pre_train_time = time.time() - start_pre

    start_ft = time.time()
    criterion = CombinedLoss(lambda_phy=config['lambda_phy'], ignore_index=BACKGROUND_LABEL)
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    for epoch in range(1, 201):
        model.train()
        correct = total = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch:03d}/200", leave=False):
            main = batch['main'].to(device)
            phy = batch['phy'].to(device)
            labels = batch['label'].to(device)
            phy_gt = batch['phy_gt'].to(device)

            if "MI-SSL" in exp_name:

                logits, _ = model(main, phy, mode='classify')
                phy_pred = None
            else:
                logits, phy_pred = model(main, phy)

            if logits.dim() == 2:

                mid = labels.shape[1] // 2
                labels = labels[:, mid, mid]
                logits = logits.unsqueeze(-1).unsqueeze(-1)
                labels = labels.unsqueeze(-1).unsqueeze(-1)
            elif logits.dim() == 4:

                if logits.shape[2:] != labels.shape[1:]:
                    logits = F.interpolate(logits, size=labels.shape[1:], mode='bilinear', align_corners=False)
            if phy_pred is not None and phy_pred.shape[2:] != phy_gt.shape[1:]:
                phy_pred = F.interpolate(phy_pred, size=phy_gt.shape[1:], mode='bilinear')
            loss, _, _ = criterion(logits, labels, phy_pred, phy_gt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                _, pred = torch.max(logits, 1)
                mask = (labels != BACKGROUND_LABEL)
                correct += (pred[mask] == labels[mask]).sum().item()
                total += mask.sum().item()
        if epoch % 10 == 0 or epoch == 1:
            train_oa = 100.0 * correct / (total + 1e-6)
            fine_tune_time = time.time() - start_ft
    total_time = pre_train_time + fine_tune_time

    pred_full = predict_full_image(dataset, model, device, config, num_classes, BACKGROUND_LABEL)


    mask = (label_map != BACKGROUND_LABEL)
    y_true = label_map[mask]
    y_pred = pred_full[mask]
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))
    cls_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    oa = (y_true == y_pred).mean()
    aa = cls_acc.mean()
    kappa = cohen_kappa_score(y_true, y_pred)

    full_dataset = PolSARDataset(
        coherency_path=paths['coherency'],
        freeman_paths=paths['freeman'],
        label_path=paths['label'],
        patch_size=config['patch_size'],
        is_train=False
    )
    save_prefix = os.path.join(save_root, "TSNE", exp_name)

    result = {
        "Exp": exp_name, "OA": oa*100, "AA": aa*100, "Kappa": kappa,
        "Pre_Time": pre_train_time, "FT_Time": fine_tune_time, "Total_Time": total_time
    }
    for idx, name in enumerate(class_names):
        result[f"Acc_{name}"] = cls_acc[idx]*100

    del model, optimizer, scheduler, dataloader, dataset, pred_full
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result

def main():
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    paths = {
        'coherency': "./coherency_9ch_v7.mat",
        'freeman': ["./complex_Pd.mat",
                    "./complex_Ps.mat",
                    "./complex_Pv.mat"],
        'label': "./label.mat"
    }


    all_results = []

if __name__ == "__main__":
    main()
