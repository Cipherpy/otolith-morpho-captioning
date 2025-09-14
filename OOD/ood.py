#!/usr/bin/env python
# eval_ood.py ---------------------------------------------------------------
# Now with AUROC curve plotting and Grad-CAM visualisation
# --------------------------------------------------------------------------
import os, glob, sys, json, argparse, numpy as np, torch
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# --------------------------------------------------------------------------
from data   import determine_input_size, get_transforms
from models import create_model
# --------------------------------------------------------------------------

# ---------- Grad-CAM helpers ------------------------------------------------
def _register_cam_hooks(model, target_layer):
    activations, gradients = {}, {}

    def fwd_hook(_, __, output):
        activations["value"] = output.detach()

    def bwd_hook(_, grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_backward_hook(bwd_hook)
    return activations, gradients, (handle_fwd, handle_bwd)


def gradcam(model, img_tensor, class_idx, target_layer):
    """
    Compute vanilla Grad-CAM heat-map for a single image tensor.
    Returns a (1,H,W) tensor normalised to [0,1].
    """
    model.zero_grad()
    score = model(img_tensor.unsqueeze(0))[0, class_idx]
    activations, gradients, handles = _register_cam_hooks(model, target_layer)

    score.backward(retain_graph=True)

    A = activations["value"][0]          # (C,H,W)
    G = gradients["value"][0]            # (C,H,W)
    weights = G.mean(dim=(1, 2), keepdim=True)   # (C,1,1)
    cam = (weights * A).sum(dim=0)               # (H,W)

    cam = cam.clamp(min=0)                        # ReLU
    cam = (cam - cam.min()) / (cam.max() + 1e-6)  # [0,1]

    # tidy up
    for h in handles: h.remove()
    return cam.cpu()


def save_cam_overlay(img_tensor, cam, out_path, alpha=0.35):
    """
    Overlay a CAM heat-map on the input image and save as PNG.
    """
    img = img_tensor.cpu()
    if img.shape[0] == 1:          # grayscale → RGB for nicer overlay
        img = img.repeat(3, 1, 1)

    cam_rgb = torch.tensor(cm.jet(cam.numpy())[:, :, :3]).permute(2, 0, 1)
    overlay = (1 - alpha) * img + alpha * cam_rgb
    overlay = overlay / overlay.max()
    save_image(overlay, out_path)


# -------------------------- original helpers ------------------------------
def find_best_checkpoint(ckpt_dir: str) -> str:
    patterns = ["*best*.pth", "*best*.pt"]
    matches = []
    for pat in patterns:
        matches.extend(glob.glob(os.path.join(ckpt_dir, pat)))
    if not matches:
        raise RuntimeError(
            f"No checkpoint matching '*best*.pth/pt' found in {ckpt_dir}"
        )
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def build_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )


@torch.no_grad()
def collect_smax(model, loader, device, return_logits=False):
    scores, logits = [], []
    for imgs, _ in tqdm(loader, leave=False):
        imgs = imgs.to(device, non_blocking=True)
        logit = model(imgs)
        smax = softmax(logit, dim=1).max(dim=1).values
        scores.extend(smax.cpu().numpy())
        if return_logits:
            logits.append(logit.cpu())
    if return_logits:
        return np.asarray(scores), torch.cat(logits)
    return np.asarray(scores)


# ----------------------------- main ---------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Open-set / OOD evaluation for CNN models"
    )
    parser.add_argument("--model", default="resnet152")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument("--id-dir", required=True)
    parser.add_argument("--ood-root", required=True)
    parser.add_argument("--img-size", type=int, default=512)
    parser.add_argument("--grayscale", type=int, choices=[0, 1], default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tau", type=float, default=0.5)

    # ---- new flags ----
    parser.add_argument("--plot-roc", action="store_true",
                        help="Save ROC curve PNG alongside results JSON")
    parser.add_argument("--gradcam", type=int, default=0,
                        help="Number of ID & OOD images to visualise with Grad-CAM")
    parser.add_argument("--gradcam-out", default="gradcam_out",
                        help="Folder for saving Grad-CAM overlays")

    args = parser.parse_args()
    os.makedirs(args.gradcam_out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = determine_input_size(args.model, args.img_size)
    tf_dict = get_transforms(
        img_size, augment=False, grayscale=bool(args.grayscale)
    )
    test_tf = tf_dict["test"]

    # ------------------------------------------------------------------
    # resolve checkpoint path
    # ------------------------------------------------------------------
    if args.checkpoint:
        ckpt_path = args.checkpoint
    elif args.ckpt_dir:
        ckpt_path = find_best_checkpoint(args.ckpt_dir)
        print(f"[*] Auto-selected best checkpoint: {ckpt_path}")
    else:
        sys.exit("ERROR: provide --checkpoint FILE or --ckpt-dir DIR")

    # ------------------------------------------------------------------
    # build model & load weights
    # ------------------------------------------------------------------
    print("[*] Loading dataset class list …")
    id_ds = datasets.ImageFolder(args.id_dir, test_tf)
    num_classes = len(id_ds.classes)
    print(f"Found {num_classes} ID classes")

    print("[*] Building model …")
    model = create_model(
        args.model, num_classes, pretrained=False, freeze_base=False
    )
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()

    # ------------------  ID scores ------------------------------------
    print("[*] Scoring ID test images …")
    id_loader = build_loader(id_ds, args.batch_size, args.num_workers)
    id_scores = collect_smax(model, id_loader, device)

    avg_conf_id = id_scores.mean()
    fpr_id = (id_scores < args.tau).sum() / id_scores.size

    # ------------------  OOD subsets ----------------------------------
    subsets = ["deepsea", "shallow_marine", "freshwater"]
    ood_scores_all, is_ood_flags = [], []
    metrics = {}

    # keep one small loader for Grad-CAM later
    ood_example_loaders = {}

    for sub in subsets:
        sub_dir = os.path.join(args.ood_root, sub)
        if not os.path.isdir(sub_dir):
            print(f"WARNING: {sub_dir} not found – skipping")
            continue
        ds = datasets.ImageFolder(sub_dir, test_tf)
        if len(ds) == 0:
            print(f"WARNING: no images in {sub_dir} – skipping")
            continue

        print(f"[*] Scoring OOD subset '{sub}' ({len(ds)} images)")
        loader = build_loader(ds, args.batch_size, args.num_workers)
        scores = collect_smax(model, loader, device)
        tpr = (scores < args.tau).sum() / scores.size
        metrics[sub] = {
            "n_images": len(scores),
            "TPR_OOD": round(float(tpr), 3),
            "avg_conf": round(float(scores.mean()), 3),
        }
        ood_scores_all.append(scores)
        is_ood_flags.extend([1] * len(scores))

        if args.gradcam > 0:
            ood_example_loaders[sub] = DataLoader(
                ds, batch_size=1, shuffle=True)

    # concatenate for AUROC / AUPR
    ood_scores_all = (
        np.concatenate(ood_scores_all) if len(ood_scores_all) else np.array([])
    )
    all_scores = np.concatenate([id_scores, ood_scores_all])
    is_ood_flags.extend([0] * len(id_scores))
    is_ood_flags = np.asarray(is_ood_flags)

    auroc = roc_auc_score(is_ood_flags, -all_scores)
    aupr_in = average_precision_score(1 - is_ood_flags, all_scores)
    aupr_out = average_precision_score(is_ood_flags, -all_scores)

    # ------------------  ROC curve plot --------------------------------
    if args.plot_roc:
        fpr, tpr, _ = roc_curve(is_ood_flags, -all_scores)
        plt.figure(figsize=(4, 4))
        plt.plot(fpr, tpr, lw=2, label=f"AUROC = {auroc:.3f}")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("AUROC (ResNet152)")
        plt.grid(True, ls="--", alpha=.4)
        plt.legend(loc="lower right")
        roc_path = os.path.splitext(ckpt_path)[0] + "_roc_curve.png"
        plt.tight_layout()
        plt.savefig(roc_path, dpi=300)
        print(f"[*] Saved ROC curve to {roc_path}")

    # ------------------  Grad-CAM --------------------------------------
    if args.gradcam > 0:
        print(f"[*] Generating Grad-CAM overlays (N = {args.gradcam}) …")

        # pick the last conv layer automatically (works for most torchvision CNNs)
        target_layer = None
        for m in reversed(model.modules()):
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
                break
        assert target_layer is not None, "Could not locate a Conv2d layer for CAM"

        tf_inv = transforms.Compose([
            transforms.Normalize(mean=[0, 0, 0], std=[1/0.5]*3),
            transforms.Normalize(mean=[-0.5]*3, std=[1,1,1])
        ]) if test_tf.transforms[0].__class__.__name__ == "Normalize" else lambda x: x

        # --- ID examples ------------------------------------------------
        id_cam_loader = DataLoader(id_ds, batch_size=1, shuffle=True)
        for i, (img, lbl) in enumerate(it := tqdm(id_cam_loader, total=args.gradcam, desc="[ID]")):
            img = img[0]
            cam = gradcam(model, img.to(device), class_idx=lbl.item(), target_layer=target_layer)
            save_cam_overlay(tf_inv(img), cam,
                             os.path.join(args.gradcam_out, f"id_{i}.png"))
            if i + 1 >= args.gradcam:
                break

        # --- one OOD subset (take whichever exists first) --------------
        for sub, loader in ood_example_loaders.items():
            for i, (img, _) in enumerate(it := tqdm(loader, total=args.gradcam, desc=f"[OOD:{sub}]")):
                img = img[0]
                # class index = argmax => “most confident wrong” class
                with torch.no_grad():
                    logits = model(img.unsqueeze(0).to(device))
                cls_idx = logits.argmax(1).item()
                cam = gradcam(model, img.to(device), class_idx=cls_idx, target_layer=target_layer)
                save_cam_overlay(tf_inv(img), cam,
                                 os.path.join(args.gradcam_out, f"ood_{sub}_{i}.png"))
                if i + 1 >= args.gradcam:
                    break
            break  # only one subset

        print(f"[*] CAM images saved in '{args.gradcam_out}/'")

    # ------------------  print table ----------------------------------
    print(
        "\n=== Open-set metrics "
        f"(τ = {args.tau:.2f}, img {img_size}², model '{args.model}') ==="
    )
    print(f"ID images: {len(id_scores)}  |  Avg-conf ID = {avg_conf_id:.3f}")
    print(f"FPR_ID    = {fpr_id:.3f}\n")
    print("{:<15} {:>6} {:>10} {:>15}".format("OOD subset", "N", "TPR_OOD", "Avg.conf"))
    for sub in subsets:
        if sub in metrics:
            m = metrics[sub]
            print(
                "{:<15} {:>6} {:>10.3f} {:>15.3f}".format(
                    sub, m["n_images"], m["TPR_OOD"], m["avg_conf"]
                )
            )
    print("---------------------------------------------------------------")
    print(f"Global  AUROC   : {auroc:.3f}")
    print(f"AUPR_In         : {aupr_in:.3f}")
    print(f"AUPR_Out        : {aupr_out:.3f}")
    if len(ood_scores_all):
        print(f"Avg conf OOD     : {ood_scores_all.mean():.3f}")

    # ------------------ save JSON -------------------------------------
    metrics.update(
        {
            "Avg_conf_ID": round(float(avg_conf_id), 3),
            "FPR_ID": round(float(fpr_id), 3),
            "AUROC": round(float(auroc), 3),
            "AUPR_In": round(float(aupr_in), 3),
            "AUPR_Out": round(float(aupr_out), 3),
            "tau": args.tau,
        }
    )
    json_out = os.path.splitext(ckpt_path)[0] + "_OOD_results.json"
    with open(json_out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved detailed results to {json_out}")


if __name__ == "__main__":
    main()
# python ood.py \
#   --id-dir /home/reshma/Otolith/otolith_Final/model_data/test \
#   --ood-root /home/reshma/Otolith/otolith_Final/OOD \
#   --ckpt-dir /home/reshma/Otolith/otolith_Final/src/outputs/20250515_095917/models/
