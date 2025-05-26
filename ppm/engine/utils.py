import os
import torch

from pathlib import Path
from typing import Union


def save_checkpoint(checkpoint: dict, experiment_id: Union[str, int]) -> Path:
    """
    Save the checkpoint dict as a .pth file under models/suffix/<experiment_id>.pth.
    If VSC_DATA is set, it’s used as the base folder; otherwise the current dir is used.
    """
    base_dir = Path(os.getenv("VSC_SCRATCH", "."))
    save_path = base_dir / "persisted_models" / "suffix" / f"{experiment_id}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(checkpoint, save_path)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        try:
            base_dir = Path(os.getenv("VSC_DATA", "."))
            save_path = (
                base_dir / "persisted_models" / "suffix" / f"{experiment_id}.pth"
            )
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, save_path)
        except Exception as e:
            print(f"Error saving checkpoint to VSC_DATA: {e}")

    return save_path


def load_checkpoint(ckpt_path: str, map_location=None):
    """Loads torch model from checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(" [*] Loading checkpoint from %s succeed!" % ckpt_path)
    return ckpt


def save_confidence_level(
    model,
    test_loader,
    config,
):
    import torch.nn.functional as F

    device = config["device"]

    n_classes = len(test_loader.dataset.log.itos["activity"])
    confidence_sum = torch.zeros(n_classes)
    total_count = torch.zeros(n_classes)
    total_count_ground_truth = torch.zeros(n_classes)
    accuracy_sum = torch.zeros(n_classes)

    def to_device(*args):
        return (item.to(device) for item in args)

    model.eval()
    with torch.inference_mode():
        for items in test_loader:
            x_cat, x_num, y_cat, y_num = to_device(*items)

            attention_mask = x_cat[..., 0] != 0

            outputs, _ = model(
                x_cat=x_cat, x_num=x_num, attention_mask=attention_mask.long()
            )

            mask_flat = attention_mask.view(-1)
            y_cat_flat = y_cat.view(-1)[mask_flat]

            for target in test_loader.dataset.log.targets.categorical:
                probs = F.softmax(outputs[target], dim=-1)
                max_probs, preds = probs.max(dim=-1)

                preds_flat = preds.view(-1)[mask_flat].cpu()
                probs_flat = max_probs.view(-1)[mask_flat].cpu()

                total_count += torch.bincount(preds_flat, minlength=n_classes)
                confidence_sum += torch.bincount(
                    preds_flat, weights=probs_flat, minlength=n_classes
                )
                total_count_ground_truth += torch.bincount(
                    y_cat_flat.cpu(), minlength=n_classes
                )

                accuracy_sum += torch.bincount(
                    preds_flat,
                    weights=(preds_flat == y_cat_flat.cpu()).float(),
                    minlength=n_classes,
                )

    avg_confidence = torch.where(
        total_count > 0, confidence_sum / total_count, torch.zeros(n_classes)
    )
    avg_accuracy = torch.where(
        total_count > 0, accuracy_sum / total_count, torch.zeros(n_classes)
    )

    import pandas as pd
    import os

    csv_path = "notebooks/confidence_eval.csv"
    results = pd.read_csv(csv_path) if os.path.exists(csv_path) else pd.DataFrame()

    new_rows = []
    for cls in range(n_classes):
        new_rows.append(
            {
                "log": config["log"],
                "backbone": config["backbone"],
                "activity": test_loader.dataset.log.itos["activity"][cls],
                "avg_confidence": avg_confidence[cls].item(),
                "avg_accuracy": avg_accuracy[cls].item(),
                "predicted_count": total_count[cls].item(),
                "true_count": total_count_ground_truth[cls].item(),
            }
        )

    results = pd.concat([results, pd.DataFrame(new_rows)], ignore_index=True)
    results.to_csv(csv_path, index=False)
