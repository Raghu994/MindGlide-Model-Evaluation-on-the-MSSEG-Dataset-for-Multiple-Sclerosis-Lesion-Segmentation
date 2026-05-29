import os
import json
import torch
import time
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import binary_dilation
from monai.data import Dataset, DataLoader
from monai.inferers import SlidingWindowInferer
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, SurfaceDiceMetric
from monai.transforms import AsDiscrete

import sys
sys.path.append("/content/mindGlide/inference")
from mindglide.network import get_network
from mindglide.transforms import get_transforms, recovery_prediction
from mindglide.consts import PATCH_SIZE, PROPERTIES

def compute_boundary_iou(pred, gt, dilation_iters=2):
    if np.sum(pred) == 0 and np.sum(gt) == 0: return 1.0
    if np.sum(pred) == 0 or np.sum(gt) == 0: return 0.0
    pred_boundary = binary_dilation(pred, iterations=dilation_iters) ^ pred
    gt_boundary = binary_dilation(gt, iterations=dilation_iters) ^ gt
    intersection = np.logical_and(pred_boundary, gt_boundary).sum()
    union = np.logical_or(pred_boundary, gt_boundary).sum()
    return intersection / union if union > 0 else 0.0

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/content/mindGlide/models/_20240404_conjurer_trained_dice_7733.pt"
    dataset_path = "/content/MSSEG_Manual_Corrected_new"
    
    net = get_network(checkpoint_path=model_path, device=device).eval()
    patch_inferer = SlidingWindowInferer(roi_size=PATCH_SIZE, sw_batch_size=4, overlap=0.5, mode='gaussian')
    as_discrete = AsDiscrete(argmax=True, to_onehot=len(PROPERTIES['labels']))
    transforms = get_transforms(no_reorient=False)
    
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    hd95_metric = HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean")
    assd_metric = SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean")
    nsd_metric = SurfaceDiceMetric(class_thresholds=[1.0], include_background=False, reduction="mean")
    
    subjects = sorted([s for s in os.listdir(dataset_path) if s.startswith('sub-')])
    final_results = {}
    
    os.makedirs("/content/slicer_ready", exist_ok=True)
    
    with torch.inference_mode():
        for sub in tqdm(subjects, desc="Final Evaluation Loop"):
            flair_path = f"{dataset_path}/{sub}/anat/{sub}_acq-time01_flair.nii.gz"
            manual_path = f"{dataset_path}/{sub}/manual/{sub}_time01_manual.nii.gz"
            if not os.path.exists(flair_path) or not os.path.exists(manual_path): continue
                
            dataset = Dataset(data=[{'image': flair_path}], transform=transforms)
            batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
            
            start_time = time.time()
            predictions = patch_inferer(batch['image'].to(device), net).cpu()
            inf_time = time.time() - start_time
            
            pred = as_discrete(predictions[0])
            if batch["resample_flag"][0].item():
                pred = recovery_prediction(pred, [len(PROPERTIES['labels']), *batch["crop_shape"][0].tolist()], batch["anisotrophy_flag"][0].item())
            pred = np.argmax(pred, axis=0)
            
            pred_padded = np.zeros(batch["original_shape"][0].tolist(), dtype=pred.dtype)
            (hs, ws, ds), (he, we, de) = batch["bbox"][0].tolist()
            pred_padded[hs:he, ws:we, ds:de] = pred
            
            nifti_img = nib.Nifti1Image(pred_padded.astype(np.uint8), batch["output_affine"][0])
            orig_ornt = nib.orientations.io_orientation(batch['image_meta_dict']['affine'][0].numpy())
            curr_ornt = nib.orientations.io_orientation(batch["output_affine"][0])
            if not np.all(curr_ornt == orig_ornt):
                nifti_img = nifti_img.as_reoriented(nib.orientations.ornt_transform(curr_ornt, orig_ornt))
            
            # Save Slicer Files
            if sub in ["sub-019", "sub-021", "sub-029"]:
                nib.save(nifti_img, f"/content/slicer_ready/{sub}_prediction.nii.gz")
                
            final_pred_data = nifti_img.get_fdata()
            gold_data = nib.load(manual_path).get_fdata()
            
            pred_bin = (torch.from_numpy(final_pred_data) == 18).float().unsqueeze(0).unsqueeze(0)
            gold_bin = (torch.from_numpy(gold_data) > 0).float().unsqueeze(0).unsqueeze(0)
            
            dice_metric(y_pred=pred_bin, y=gold_bin)
            hd95_metric(y_pred=pred_bin, y=gold_bin)
            assd_metric(y_pred=pred_bin, y=gold_bin)
            nsd_metric(y_pred=pred_bin, y=gold_bin)
            
            tp = torch.sum(pred_bin * gold_bin).item()
            fp = torch.sum(pred_bin * (1 - gold_bin)).item()
            fn = torch.sum((1 - pred_bin) * gold_bin).item()
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            final_results[sub] = {
                "Dice": round(dice_metric.aggregate().item(), 4),
                "HD95": round(hd95_metric.aggregate().item(), 4),
                "ASSD": round(assd_metric.aggregate().item(), 4),
                "NSD": round(nsd_metric.aggregate().item(), 4),
                "Boundary_IoU": round(compute_boundary_iou(final_pred_data == 18, gold_data > 0), 4),
                "Precision": round(prec, 4),
                "Recall": round(rec, 4),
                "F1_score": round(2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0, 4),
                "inference_time_seconds": round(inf_time, 2)
            }
            
            for m in [dice_metric, hd95_metric, assd_metric, nsd_metric]: m.reset()

    with open("/content/mindglide_final_metrics.json", 'w') as f:
        json.dump(final_results, f, indent=4)
    print("\nComplete! Final JSON and Slicer files are ready.")

if __name__ == "__main__":
    main()
