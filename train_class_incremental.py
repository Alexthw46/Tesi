import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import clip
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.dataset.data_utils import get_dataset
from src.inference_val import get_template
from src.models.L2CLIP import CLIPWithL2P
from src.chils_eval import run_chils
from src.utils.labelsets_utils import get_true_cifar20_labelset, load_label_set


def get_classnames(dataset) -> List[str]:
    if hasattr(dataset, "classes"):
        return list(dataset.classes)
    raise AttributeError("Dataset does not expose class names via `classes`.")


def get_targets(dataset) -> List[int]:
    if hasattr(dataset, "targets"):
        return [int(x) for x in dataset.targets]
    if hasattr(dataset, "labels"):
        return [int(x) for x in dataset.labels]
    raise AttributeError("Dataset does not expose labels via `targets` or `labels`.")


def build_class_order(num_classes: int, seed: int, random_order: bool) -> List[int]:
    if not random_order:
        return list(range(num_classes))
    rng = np.random.RandomState(seed)
    return rng.permutation(num_classes).tolist()


def build_task_splits(class_order: Sequence[int], classes_per_task: int, num_tasks: int = -1) -> List[List[int]]:
    if classes_per_task <= 0:
        raise ValueError("classes_per_task must be > 0")

    full_num_tasks = len(class_order) // classes_per_task
    if full_num_tasks == 0:
        raise ValueError("Not enough classes for even one task")

    if num_tasks <= 0:
        num_tasks = full_num_tasks
    num_tasks = min(num_tasks, full_num_tasks)

    splits = []
    for task_id in range(num_tasks):
        start = task_id * classes_per_task
        end = start + classes_per_task
        splits.append(list(class_order[start:end]))
    return splits


def build_subset_indices(
    targets: Sequence[int],
    class_ids: Sequence[int],
    max_per_class: int = -1,
) -> List[int]:
    class_set = set(class_ids)
    if max_per_class is None or max_per_class <= 0:
        return [i for i, y in enumerate(targets) if int(y) in class_set]

    counts = {c: 0 for c in class_set}
    selected = []
    for i, y in enumerate(targets):
        yi = int(y)
        if yi in class_set and counts[yi] < max_per_class:
            selected.append(i)
            counts[yi] += 1
    return selected


def remap_labels_to_seen(labels: torch.Tensor, class_to_seen_idx: Dict[int, int], device: str) -> torch.Tensor:
    mapped = [class_to_seen_idx[int(x)] for x in labels.detach().cpu().tolist()]
    return torch.tensor(mapped, dtype=torch.long, device=device)


def refresh_text_features(model: CLIPWithL2P, all_classnames: Sequence[str], seen_classes: Sequence[int]) -> None:
    model.classnames = [all_classnames[c] for c in seen_classes]
    with torch.no_grad():
        model.text_features = model._build_text_features()


class ReplayBuffer:
    """Simple reservoir replay buffer over already-preprocessed image tensors."""

    def __init__(self, capacity: int):
        self.capacity = max(0, int(capacity))
        self.images: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.num_seen = 0

    def add_batch(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        if self.capacity <= 0:
            return

        images_cpu = images.detach().cpu()
        labels_cpu = labels.detach().cpu().tolist()

        for img, lbl in zip(images_cpu, labels_cpu):
            self.num_seen += 1
            if len(self.images) < self.capacity:
                self.images.append(img.clone())
                self.labels.append(int(lbl))
                continue

            j = np.random.randint(0, self.num_seen)
            if j < self.capacity:
                self.images[j] = img.clone()
                self.labels[j] = int(lbl)

    def sample(self, batch_size: int, device: str):
        if len(self.images) == 0 or batch_size <= 0:
            return None

        k = min(batch_size, len(self.images))
        idx_np = np.random.choice(len(self.images), size=k, replace=False)
        idx_list = [int(i) for i in np.atleast_1d(idx_np).tolist()]
        imgs = torch.stack([self.images[i] for i in idx_list], dim=0).to(device)
        lbls = torch.tensor([self.labels[i] for i in idx_list], dtype=torch.long, device=device)
        return imgs, lbls

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "images": self.images,
            "labels": self.labels,
            "num_seen": self.num_seen,
        }


def train_one_task(
    model: CLIPWithL2P,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    class_to_seen_idx: Dict[int, int],
    device: str,
    replay_buffer: Optional[ReplayBuffer],
    replay_mode: str,
    replay_batch_size: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0

    for batch in tqdm(train_loader, desc="Train", total=len(train_loader), leave=False):
        images, labels = batch[0].to(device), batch[1].to(device)
        labels_seen = remap_labels_to_seen(labels, class_to_seen_idx, device)

        if replay_mode == "reservoir" and replay_buffer is not None:
            replay = replay_buffer.sample(replay_batch_size, device)
            if replay is not None:
                rep_images, rep_labels_global = replay
                rep_labels_seen = remap_labels_to_seen(rep_labels_global, class_to_seen_idx, device)
                images = torch.cat([images, rep_images], dim=0)
                labels_seen = torch.cat([labels_seen, rep_labels_seen], dim=0)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits.float(), labels_seen)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels_seen.size(0)
        total += labels_seen.size(0)
        correct += (logits.argmax(dim=1) == labels_seen).sum().item()

        # Feed only current-task samples to the replay buffer.
        if replay_mode == "reservoir" and replay_buffer is not None:
            replay_buffer.add_batch(batch[0], batch[1])

    return total_loss / max(total, 1), correct / max(total, 1)


def evaluate_subset(
    model: CLIPWithL2P,
    dataset,
    targets: Sequence[int],
    eval_classes: Sequence[int],
    class_to_seen_idx: Dict[int, int],
    batch_size: int,
    num_workers: int,
    device: str,
    max_per_class: int,
) -> float:
    eval_indices = build_subset_indices(targets, eval_classes, max_per_class=max_per_class)
    if not eval_indices:
        return 0.0

    loader = DataLoader(
        Subset(dataset, eval_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Eval", total=len(loader), leave=False):
            images, labels = batch[0].to(device), batch[1].to(device)
            logits = model(images)
            labels_seen = remap_labels_to_seen(labels, class_to_seen_idx, device)
            preds = logits.argmax(dim=1)
            correct += (preds == labels_seen).sum().item()
            total += labels_seen.size(0)

    return correct / max(total, 1)


def compute_incremental_metrics(acc_matrix: np.ndarray) -> Dict[str, float]:
    num_tasks = acc_matrix.shape[0]
    final_row = acc_matrix[num_tasks - 1]

    avg_inc_acc = float(np.nanmean(final_row))

    if num_tasks <= 1:
        avg_forgetting = 0.0
    else:
        forgetting = []
        for task_id in range(num_tasks - 1):
            best_before_final = np.nanmax(acc_matrix[: num_tasks - 1, task_id])
            final_perf = acc_matrix[num_tasks - 1, task_id]
            forgetting.append(float(best_before_final - final_perf))
        avg_forgetting = float(np.mean(forgetting)) if forgetting else 0.0

    return {
        "avg_incremental_accuracy": avg_inc_acc,
        "avg_forgetting": avg_forgetting,
    }


def _normalize_label_name(name: str) -> str:
    return str(name).replace("_", " ").strip().lower()


def build_chils_mappings(all_classnames: Sequence[str], chils_labelset: str, label_sets_dir: str):
    if chils_labelset == "true":
        sub2super = get_true_cifar20_labelset()
    else:
        sub2super = load_label_set("cifar20", int(chils_labelset), label_sets_dir=label_sets_dir)

    breeds_classes = list(sub2super.keys())
    coarse_to_idx = {coarse_name: idx for idx, coarse_name in enumerate(breeds_classes)}

    true_set = get_true_cifar20_labelset()
    fine_to_coarse_name = {}
    for coarse_name, fine_names in true_set.items():
        for fine_name in fine_names:
            fine_to_coarse_name[_normalize_label_name(fine_name)] = coarse_name

    fine_to_super_idx = {}
    for fine_idx, fine_name in enumerate(all_classnames):
        norm_fine = _normalize_label_name(fine_name)
        if norm_fine not in fine_to_coarse_name:
            raise ValueError(f"CIFAR fine class '{fine_name}' does not map to any superclass")
        coarse_name = fine_to_coarse_name[norm_fine]
        if coarse_name not in coarse_to_idx:
            raise ValueError(f"Superclass '{coarse_name}' missing from selected CHiLS label-set keys")
        fine_to_super_idx[fine_idx] = coarse_to_idx[coarse_name]

    return sub2super, breeds_classes, fine_to_super_idx


def evaluate_chils_subset(
    model: CLIPWithL2P,
    dataset,
    targets: Sequence[int],
    eval_classes: Sequence[int],
    fine_to_super_idx: Dict[int, int],
    sub2super,
    breeds_classes,
    reweighter: str,
    experiment: str,
    best_poss: bool,
    batch_size: int,
    num_workers: int,
    device: str,
    max_per_class: int,
):
    eval_indices = build_subset_indices(targets, eval_classes, max_per_class=max_per_class)
    if not eval_indices:
        return {}

    loader = DataLoader(
        Subset(dataset, eval_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
    )

    model.eval()
    all_features = []
    all_super_labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="CHiLS Eval", total=len(loader), leave=False):
            images, labels = batch[0].to(device), batch[1].to(device)
            image_features = model.emb_forward(images)
            all_features.append(image_features.detach().cpu())
            batch_super_labels = [fine_to_super_idx[int(x)] for x in labels.detach().cpu().tolist()]
            all_super_labels.append(np.array(batch_super_labels, dtype=np.int64))

    features_for_chils = torch.cat(all_features, dim=0)
    labels_for_chils = np.concatenate(all_super_labels, axis=0)
    chils_out, _ = run_chils(
        clip_with_prompt=model,
        features=features_for_chils,
        labels=labels_for_chils,
        sub2super=sub2super,
        breeds_classes=breeds_classes,
        reweighter=reweighter,
        experiment=experiment,
        best_poss=best_poss,
    )
    return chils_out


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Class-incremental training for CLIP+L2P prompt tuning")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--clip_model", default="ViT-B/16")
    parser.add_argument("--work_dir", default=".")

    parser.add_argument("--classes_per_task", type=int, default=10)
    parser.add_argument("--num_tasks", type=int, default=-1)
    parser.add_argument("--class_order_seed", type=int, default=0)
    parser.add_argument("--random_class_order", action="store_true")

    parser.add_argument("--epochs_per_task", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    parser.add_argument("--prompt_pool_size", type=int, default=50)

    parser.add_argument("--replay_mode", default="none", choices=["none", "reservoir"])
    parser.add_argument("--memory_size", type=int, default=0)
    parser.add_argument("--replay_batch_size", type=int, default=32)

    parser.add_argument("--max_train_per_class", type=int, default=-1)
    parser.add_argument("--max_eval_per_class", type=int, default=-1)

    parser.add_argument("--save_every_task", action="store_true")

    parser.add_argument("--superclass_eval", action="store_true",
                        help="Run CHiLS-style superclass evaluation on the seen-class test subset each task.")
    parser.add_argument("--label_sets_dir", default="label_sets")
    parser.add_argument("--chils_labelset", default="true",
                        help="CIFAR20 label-set variant: 'true' or numeric size (e.g. 10).")
    parser.add_argument("--chils_reweighter", default="normal", choices=["normal", "hat", "supagg"])
    parser.add_argument("--chils_experiment", default="true", choices=["true", "gpt", "true_lin", "gpt_lin"])
    parser.add_argument("--chils_best_possible", action="store_true")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.dataset.lower() != "cifar100":
        raise ValueError("This script currently targets CIFAR100-style class-incremental setup.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    np.random.seed(args.class_order_seed)
    torch.manual_seed(args.class_order_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.class_order_seed)

    _, preprocess = clip.load(args.clip_model, device=device)

    train_dataset = get_dataset(args.data_dir, args.dataset, train=True, transform=preprocess)
    test_dataset = get_dataset(args.data_dir, args.dataset, train=False, transform=preprocess)

    all_classnames = get_classnames(train_dataset)
    num_classes = len(all_classnames)

    class_order = build_class_order(num_classes, args.class_order_seed, args.random_class_order)
    task_splits = build_task_splits(class_order, args.classes_per_task, num_tasks=args.num_tasks)
    num_tasks = len(task_splits)

    print(f"Classes: {num_classes} | Tasks: {num_tasks} | Classes/task: {args.classes_per_task}")

    model = CLIPWithL2P(
        device=device,
        classnames=[all_classnames[c] for c in task_splits[0]],
        template=get_template(args.dataset),
        clip_model=args.clip_model,
        prompt_pool_size=args.prompt_pool_size,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.prompt.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    replay_buffer = None
    if args.replay_mode == "reservoir":
        replay_buffer = ReplayBuffer(capacity=args.memory_size)

    train_targets = get_targets(train_dataset)
    test_targets = get_targets(test_dataset)

    sub2super = None
    breeds_classes = None
    fine_to_super_idx = None
    chils_history = []
    if args.superclass_eval:
        if args.dataset.lower() != "cifar100":
            raise ValueError("--superclass_eval is currently supported only for --dataset cifar100")
        sub2super, breeds_classes, fine_to_super_idx = build_chils_mappings(
            all_classnames,
            args.chils_labelset,
            args.label_sets_dir,
        )

    seen_classes: List[int] = []
    acc_matrix = np.full((num_tasks, num_tasks), np.nan, dtype=np.float32)

    ckpt_dir = os.path.join(args.work_dir, "checkpoints", args.clip_model.replace("/", "_"), "class_incremental")
    os.makedirs(ckpt_dir, exist_ok=True)

    for task_id, task_classes in enumerate(task_splits):
        print("\n" + "=" * 80)
        print(f"Task {task_id + 1}/{num_tasks} | New classes: {task_classes}")

        seen_classes.extend(task_classes)
        seen_classes = sorted(set(seen_classes))
        class_to_seen_idx = {cls_id: idx for idx, cls_id in enumerate(seen_classes)}

        refresh_text_features(model, all_classnames, seen_classes)

        train_indices = build_subset_indices(
            train_targets,
            task_classes,
            max_per_class=args.max_train_per_class,
        )
        train_loader = DataLoader(
            Subset(train_dataset, train_indices),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
        )

        for epoch in range(1, args.epochs_per_task + 1):
            train_loss, train_acc = train_one_task(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                class_to_seen_idx=class_to_seen_idx,
                device=device,
                replay_buffer=replay_buffer,
                replay_mode=args.replay_mode,
                replay_batch_size=args.replay_batch_size,
            )
            print(
                f"Task {task_id + 1} | Epoch {epoch}/{args.epochs_per_task} | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}"
            )

        for eval_task_id in range(task_id + 1):
            eval_classes = task_splits[eval_task_id]
            acc = evaluate_subset(
                model=model,
                dataset=test_dataset,
                targets=test_targets,
                eval_classes=eval_classes,
                class_to_seen_idx=class_to_seen_idx,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                device=device,
                max_per_class=args.max_eval_per_class,
            )
            acc_matrix[task_id, eval_task_id] = acc
            print(f"Task {task_id + 1} -> eval on task {eval_task_id + 1}: acc={acc:.4f}")

        seen_acc = evaluate_subset(
            model=model,
            dataset=test_dataset,
            targets=test_targets,
            eval_classes=seen_classes,
            class_to_seen_idx=class_to_seen_idx,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            device=device,
            max_per_class=args.max_eval_per_class,
        )
        print(f"Task {task_id + 1} -> cumulative seen-class acc={seen_acc:.4f}")

        if args.superclass_eval:
            assert fine_to_super_idx is not None
            assert sub2super is not None
            assert breeds_classes is not None
            chils_out = evaluate_chils_subset(
                model=model,
                dataset=test_dataset,
                targets=test_targets,
                eval_classes=seen_classes,
                fine_to_super_idx=fine_to_super_idx,
                sub2super=sub2super,
                breeds_classes=breeds_classes,
                reweighter=args.chils_reweighter,
                experiment=args.chils_experiment,
                best_poss=args.chils_best_possible,
                batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                device=device,
                max_per_class=args.max_eval_per_class,
            )
            chils_history.append({"task": task_id + 1, **{k: float(v) for k, v in chils_out.items()}})
            print(f"Task {task_id + 1} -> CHiLS seen-subset metrics: {chils_out}")

        if args.save_every_task:
            task_ckpt = os.path.join(ckpt_dir, f"task_{task_id + 1}.pt")
            torch.save(
                {
                    "task": task_id + 1,
                    "seen_classes": seen_classes,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "replay_mode": args.replay_mode,
                    "replay_buffer": replay_buffer.state_dict() if replay_buffer is not None else None,
                    "chils_metrics": chils_history[-1] if (args.superclass_eval and chils_history) else None,
                },
                task_ckpt,
            )
            print(f"Saved task checkpoint: {task_ckpt}")

    metrics = compute_incremental_metrics(acc_matrix)
    print("\nFinal metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    final_ckpt = os.path.join(ckpt_dir, "final.pt")
    torch.save(
        {
            "seen_classes": seen_classes,
            "acc_matrix": acc_matrix,
            "metrics": metrics,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "replay_mode": args.replay_mode,
            "replay_buffer": replay_buffer.state_dict() if replay_buffer is not None else None,
            "chils_history": chils_history,
            "args": vars(args),
        },
        final_ckpt,
    )
    print(f"Saved final checkpoint: {final_ckpt}")

    metrics_json_path = os.path.join(ckpt_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "task_splits": task_splits,
                "acc_matrix": acc_matrix.tolist(),
                "metrics": metrics,
                "seen_classes": seen_classes,
                "chils_history": chils_history,
            },
            f,
            indent=2,
        )
    print(f"Saved metrics JSON: {metrics_json_path}")

    metrics_csv_path = os.path.join(ckpt_dir, "acc_matrix.csv")
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["train_task"] + [f"eval_task_{i + 1}" for i in range(num_tasks)]
        writer.writerow(header)
        for i in range(num_tasks):
            row = [f"task_{i + 1}"] + [acc_matrix[i, j] for j in range(num_tasks)]
            writer.writerow(row)
    print(f"Saved accuracy matrix CSV: {metrics_csv_path}")

    if args.superclass_eval and chils_history:
        chils_csv_path = os.path.join(ckpt_dir, "chils_metrics.csv")
        with open(chils_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(chils_history[0].keys()))
            writer.writeheader()
            writer.writerows(chils_history)
        print(f"Saved CHiLS metrics CSV: {chils_csv_path}")


if __name__ == "__main__":
    main()

