# Tesi - Class Incremental Training

This repository now includes a class-incremental training entrypoint for CLIP+L2P prompt tuning:

- `train_class_incremental.py`

## What it does

- Splits CIFAR100 classes into sequential tasks (for example 10 classes per task).
- Trains prompt parameters task-by-task.
- Optionally uses reservoir replay memory.
- Evaluates task-wise and cumulative seen-class accuracy after each task.
- Saves checkpoints and metrics (`.pt`, `.json`, `.csv`).

## Quick smoke check

```powershell
python smoke_test_train_class_incremental.py
```

## Run class-incremental training (example)

```powershell
python train_class_incremental.py `
  --data_dir data `
  --dataset cifar100 `
  --clip_model ViT-B/16 `
  --classes_per_task 10 `
  --epochs_per_task 2 `
  --batch_size 64 `
  --replay_mode reservoir `
  --memory_size 2000 `
  --replay_batch_size 32 `
  --save_every_task
```

## Notes

- The script expects a CIFAR100-like dataset with `classes` and `targets` fields (standard torchvision CIFAR100).
- Output is saved under:
  - `checkpoints/<clip_model_with_slash_replaced>/class_incremental/`

