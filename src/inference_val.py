import argparse
import glob
import os

import clip
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from chils_eval import run_chils
from src.dataset.data_utils import get_dataset
from src.utils.labelsets_utils import get_true_cifar20_labelset, load_label_set
from src.models.L2CLIP import CLIPWithL2P


def get_classnames(dataset):
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    raise AttributeError("The selected dataset does not expose class names via a `classes` attribute.")


def get_template(dataset_name):
    if dataset_name.lower() == "cifar100":
        return [
            "a photo of a {}.",
            "a blurry photo of a {}.",
            "a black and white photo of a {}.",
            "a low contrast photo of a {}.",
            "a high contrast photo of a {}.",
            "a bad photo of a {}.",
            "a good photo of a {}.",
            "a photo of a small {}.",
            "a photo of a big {}.",
            "a photo of the {}.",
            "a blurry photo of the {}.",
            "a black and white photo of the {}.",
            "a low contrast photo of the {}.",
            "a high contrast photo of the {}.",
            "a bad photo of the {}.",
            "a good photo of the {}.",
            "a photo of the small {}.",
            "a photo of the big {}.",
        ]
    return ["a photo of a {}."]


def _normalize_label_name(name):
    return str(name).replace("_", " ").strip().lower()


def resolve_checkpoint(work_dir, ckpt_path=None):
    if ckpt_path:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    best_ckpt = os.path.join(work_dir, "../best_prompt.pt")
    if os.path.isfile(best_ckpt):
        return best_ckpt

    candidates = glob.glob(os.path.join(work_dir, "*.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found in '{work_dir}'. Expected 'best_prompt.pt' or any '*.pt' file."
        )

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


class PlainCLIPWithCHiLS(nn.Module):
    """Minimal CLIP wrapper for evaluating plain zero-shot CLIP with the CHiLS pipeline."""

    def __init__(self, device, classnames, template, clip_model):
        super().__init__()
        self.device = device
        self.bb, _ = clip.load(clip_model, device=device)
        self.classnames = classnames
        self.template = template
        self.text_features = self._build_text_features()

    def _build_text_features(self):
        templates = [self.template] if isinstance(self.template, str) else list(self.template)
        with torch.no_grad():
            zeroshot_weights = []
            for classname in self.classnames:
                texts = [template.format(classname) for template in templates]
                tokens = clip.tokenize(texts).to(self.device)
                class_embeddings = self.bb.encode_text(tokens)
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            return torch.stack(zeroshot_weights, dim=0)

    def encode_image_with_prompt(self, image):
        with torch.no_grad():
            return self.bb.encode_image(image.to(self.device))

    def emb_forward(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if image.dim() == 4:
            image_features = self.encode_image_with_prompt(image)
        elif image.dim() == 2:
            image_features = image.to(self.device)
        else:
            raise ValueError("emb_forward expects a 4D image tensor or 2D feature tensor")

        return image_features / image_features.norm(dim=-1, keepdim=True)

    def forward(self, image):
        image_features = self.emb_forward(image)
        logits = 100.0 * (image_features @ self.text_features.T)
        return logits


def main():
    parser = argparse.ArgumentParser(description="Quick validation inference with checkpointed prompt model")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--clip_model", default="ViT-B/16")
    parser.add_argument("--work_dir", default=".")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--show_probs", action="store_true", help="Also print softmax probabilities for top-k.")
    parser.add_argument("--eval_mode", default="prompted", choices=["prompted", "plain"],
                        help="Use the prompted L2P model or plain CLIP for the validation pass.")
    parser.add_argument("--superclass_eval", action="store_true",
                        help="Run CIFAR superclass CHiLS evaluation at the end.")
    parser.add_argument("--label_sets_dir", default="label_sets")
    parser.add_argument("--chils_labelset", default="true",
                        help="CIFAR20 label-set variant: 'true' or numeric size (e.g. 10).")
    parser.add_argument("--chils_reweighter", default="normal", choices=["normal", "hat", "supagg"])
    parser.add_argument("--chils_experiment", default="true", choices=["true", "gpt", "true_lin", "gpt_lin"])
    parser.add_argument("--chils_best_possible", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    _, preprocess = clip.load(args.clip_model, device=device)

    val_dataset = get_dataset(
        data_dir=args.data_dir,
        dataset=args.dataset,
        train=False,
        transform=preprocess,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    classnames = get_classnames(val_dataset)
    template = get_template(args.dataset)

    sub2super = None
    breeds_classes = None
    fine_to_super_idx = None
    if args.superclass_eval:
        if args.dataset.lower() != "cifar100":
            raise ValueError("--superclass_eval is currently supported only for --dataset cifar100")

        if args.chils_labelset == "true":
            sub2super = get_true_cifar20_labelset()
        else:
            sub2super = load_label_set("cifar20", int(args.chils_labelset), label_sets_dir=args.label_sets_dir)

        breeds_classes = list(sub2super.keys())
        coarse_to_idx = {coarse_name: idx for idx, coarse_name in enumerate(breeds_classes)}

        true_set = get_true_cifar20_labelset()
        fine_to_coarse_name = {}
        for coarse_name, fine_names in true_set.items():
            for fine_name in fine_names:
                fine_to_coarse_name[_normalize_label_name(fine_name)] = coarse_name

        fine_to_super_idx = {}
        for fine_idx, fine_name in enumerate(classnames):
            norm_fine = _normalize_label_name(fine_name)
            if norm_fine not in fine_to_coarse_name:
                raise ValueError(f"CIFAR fine class '{fine_name}' does not map to any superclass")
            coarse_name = fine_to_coarse_name[norm_fine]
            if coarse_name not in coarse_to_idx:
                raise ValueError(f"Superclass '{coarse_name}' missing from selected CHiLS label-set keys")
            fine_to_super_idx[fine_idx] = coarse_to_idx[coarse_name]

    if args.eval_mode == "prompted":
        net = CLIPWithL2P(
            device=device,
            classnames=classnames,
            template=template,
            clip_model=args.clip_model,
            prompt_pool_size=50
        )
        ckpt_dir = os.path.join(args.work_dir, "../checkpoints", args.clip_model)
        checkpoint_path = os.path.join(ckpt_dir, "best_50_prompt.pt")
        ckpt_path = resolve_checkpoint(args.work_dir, checkpoint_path)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        net.load_state_dict(state_dict, strict=True)
        epoch = checkpoint.get("epoch", "?") if isinstance(checkpoint, dict) else "?"
        best_val_acc = checkpoint.get("best_val_acc", "?") if isinstance(checkpoint, dict) else "?"
        print(f"Loaded checkpoint: {ckpt_path}")
        print(f"Checkpoint meta -> epoch: {epoch}, best_val_acc: {best_val_acc}")
        print("Evaluation mode: prompted CLIP + L2P")
    else:
        if args.ckpt is not None:
            print(f"Ignoring --ckpt in plain mode: {args.ckpt}")
        net = PlainCLIPWithCHiLS(
            device=device,
            classnames=classnames,
            template=template,
            clip_model=args.clip_model,
        )
        print("Evaluation mode: plain CLIP baseline (no prompts)")
    net.eval()

    shown = 0
    print_limit_hit = False
    all_features = []
    all_super_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation inference", total=len(val_loader)):
            images, labels, indices = batch[:3]
            images = images.to(device)
            labels = labels.to(device)

            if args.superclass_eval:
                image_features = net.emb_forward(images)
                text_features = net.text_features.to(device=image_features.device, dtype=image_features.dtype)
                logits = 100.0 * (image_features @ text_features.T)
                all_features.append(image_features.detach().cpu())
                batch_super_labels = [fine_to_super_idx[int(x)] for x in labels.detach().cpu().tolist()]
                all_super_labels.append(np.array(batch_super_labels, dtype=np.int64))
            else:
                logits = net(images)
            topk_vals, topk_idx = torch.topk(logits, k=min(args.top_k, logits.shape[1]), dim=1)

            if args.show_probs:
                probs = torch.softmax(logits, dim=1)
                topk_probs = torch.gather(probs, dim=1, index=topk_idx)

            if print_limit_hit:
                continue

            for i in range(images.shape[0]):
                sample_idx = int(indices[i].item())
                gt_idx = int(labels[i].item())
                gt_name = classnames[gt_idx] if 0 <= gt_idx < len(classnames) else f"class_{gt_idx}"

                pred_parts = []
                for rank in range(topk_idx.shape[1]):
                    cls_idx = int(topk_idx[i, rank].item())
                    cls_name = classnames[cls_idx] if 0 <= cls_idx < len(classnames) else f"class_{cls_idx}"
                    logit_val = float(topk_vals[i, rank].item())

                    if args.show_probs:
                        prob_val = float(topk_probs[i, rank].item())
                        pred_parts.append(
                            f"{rank + 1}:{cls_name} (id={cls_idx}, logit={logit_val:.4f}, p={prob_val:.4f})")
                    else:
                        pred_parts.append(f"{rank + 1}:{cls_name} (id={cls_idx}, logit={logit_val:.4f})")

                print(f"idx={sample_idx} | gt={gt_name} (id={gt_idx}) | top{topk_idx.shape[1]} -> " + " | ".join(
                    pred_parts))

                shown += 1
                if shown >= args.max_samples and not print_limit_hit:
                    print(f"Stopped after {shown} samples (max_samples={args.max_samples}).")
                    print_limit_hit = True
                if print_limit_hit:
                    break

    if args.superclass_eval:
        features_for_chils = torch.cat(all_features, dim=0)
        labels_for_chils = np.concatenate(all_super_labels, axis=0)
        chils_out, _ = run_chils(
            clip_with_prompt=net,
            features=features_for_chils,
            labels=labels_for_chils,
            sub2super=sub2super,
            breeds_classes=breeds_classes,
            reweighter=args.chils_reweighter,
            experiment=args.chils_experiment,
            best_poss=args.chils_best_possible,
        )
        print("\n[CHiLS superclass eval]")
        print(
            f"labelset={args.chils_labelset} | reweighter={args.chils_reweighter} | experiment={args.chils_experiment}")
        for key, value in chils_out.items():
            print(f"{key}: {float(value):.4f}")


if __name__ == "__main__":
    main()
