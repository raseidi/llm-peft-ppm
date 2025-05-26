import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

from ppm.engine.utils import save_checkpoint, save_confidence_level
from ppm.metrics import MetricsTracker

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def train_step(
    model,
    data_loader,
    optimizer,
    tracker: MetricsTracker,
    device="cuda",
    scaler=None,
    grad_clip=None,
):
    model.train()
    metrics = {
        target: {metric: 0.0 for metric in tracker.metrics[target]}
        for target in tracker.metrics
        if target.startswith("train")
    }
    total_targets = 0
    for batch, items in enumerate(data_loader):
        x_cat, x_num, y_cat, y_num = items
        x_cat, x_num, y_cat, y_num = (
            x_cat.to(device),
            x_num.to(device),
            y_cat.to(device),
            y_num.to(device),
        )

        attention_mask = (x_cat[..., 0] != 0).long()
        total_targets += attention_mask.sum().item()

        optimizer.zero_grad()
        # with torch.autocast(device_type=device, dtype=torch.float16):
        out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)

        batch_loss = 0.0
        mask = attention_mask.bool().view(-1)
        for ix, target in enumerate(data_loader.dataset.log.targets.categorical):
            loss = F.cross_entropy(
                out[target].view(-1, out[target].size(-1)),
                y_cat[..., ix].view(-1),
                ignore_index=model.padding_idx,
                reduction="sum",
            )
            predictions = torch.argmax(out[target], dim=-1)
            acc = (
                (predictions.view(-1)[mask] == y_cat[..., ix].view(-1)[mask])
                .sum()
                .item()
            )

            batch_loss += loss
            metrics[f"train_{target}"]["loss"] += loss.item()
            metrics[f"train_{target}"]["acc"] += acc

        for ix, target in enumerate(data_loader.dataset.log.targets.numerical):
            loss = F.mse_loss(
                out[target].view(-1)[mask],
                y_num[..., ix].view(-1)[mask],
                reduction="sum",
            )
            batch_loss += loss
            metrics[f"train_{target}"]["loss"] += loss.item()

        batch_loss.backward()
        if grad_clip:
            clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    for target in metrics:
        for k in metrics[target].keys():
            metrics[target][k] /= total_targets

        tracker.update(target, **metrics[target])

    return tracker


def eval_step(model, data_loader, tracker: MetricsTracker, device="cuda"):
    model.eval()
    metrics = {
        target: {metric: 0.0 for metric in tracker.metrics[target]}
        for target in tracker.metrics
        if target.startswith("test")
    }
    total_targets = 0

    with torch.inference_mode():
        for batch, items in enumerate(data_loader):
            x_cat, x_num, y_cat, y_num = items
            x_cat, x_num, y_cat, y_num = (
                x_cat.to(device),
                x_num.to(device),
                y_cat.to(device),
                y_num.to(device),
            )

            attention_mask = (x_cat[..., 0] != 0).long()
            total_targets += attention_mask.sum().item()

            # with torch.autocast(device_type=device, dtype=torch.float16):
            out, _ = model(x_cat=x_cat, x_num=x_num, attention_mask=attention_mask)

            batch_loss = 0.0
            mask = attention_mask.bool().view(-1)
            for ix, target in enumerate(data_loader.dataset.log.targets.categorical):
                loss = F.cross_entropy(
                    out[target].view(-1, out[target].size(-1)),
                    y_cat[..., ix].view(-1),
                    ignore_index=model.padding_idx,
                    reduction="sum",
                )
                predictions = torch.argmax(out[target], dim=-1)
                acc = (
                    (predictions.view(-1)[mask] == y_cat[..., ix].view(-1)[mask])
                    .sum()
                    .item()
                )

                batch_loss += loss
                metrics[f"test_{target}"]["loss"] += loss.item()
                metrics[f"test_{target}"]["acc"] += acc

            for ix, target in enumerate(data_loader.dataset.log.targets.numerical):
                loss = F.mse_loss(
                    out[target].view(-1)[mask],
                    y_num[..., ix].view(-1)[mask],
                    reduction="sum",
                )
                batch_loss += loss
                metrics[f"test_{target}"]["loss"] += loss.item()

    for target in metrics:
        for k in metrics[target].keys():
            metrics[target][k] /= total_targets

        tracker.update(target, **metrics[target])

    return tracker


def train_engine(
    model: Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    optimizer: Optimizer,
    config: dict,
    use_wandb: bool,
    persist_model: bool,
):
    model.to(config["device"])

    categorical_target_metrics = {
        f"{split}_{target}": ["loss", "acc"]
        for split in ["train", "test"]
        for target in train_loader.dataset.log.targets.categorical
    }
    numerical_target_metrics = {
        f"{split}_{target}": ["loss"]
        for split in ["train", "test"]
        for target in train_loader.dataset.log.targets.numerical
    }
    tracker = MetricsTracker({**categorical_target_metrics, **numerical_target_metrics})

    best_loss = torch.inf
    no_improvement = 0
    for epoch in range(config["epochs"]):
        tracker = train_step(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=config["device"],
            tracker=tracker,
            grad_clip=config["grad_clip"],
        )
        tracker = eval_step(
            model=model,
            data_loader=test_loader,
            device=config["device"],
            tracker=tracker,
        )

        print(
            f"Epoch {epoch}: ",
            " | ".join(
                f"{k}: {v:.4f}" for k, v in tracker.latest().items() if "best" not in k
            ),
        )

        if WANDB_AVAILABLE and use_wandb:
            wandb.log(tracker.latest())

        loss_key = (
            "test_next_activity_loss"
            if "test_next_activity_loss" in tracker.metrics
            else "test_next_remaining_time_loss"
        )
        activity_loss = tracker.latest()[loss_key]
        if persist_model:
            if activity_loss < best_loss:
                cpkt = {
                    "epoch": epoch,
                    "net": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "stoi": train_loader.dataset.log.stoi,
                    "itos": train_loader.dataset.log.itos,
                }
                save_checkpoint(
                    checkpoint=cpkt,
                    experiment_id="{}_{}".format(config["log"], config["backbone"]),
                )

        if activity_loss < best_loss:
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= 15:
                break

        best_loss = min(best_loss, tracker.latest()[loss_key])

    optimizer.zero_grad()
    # save_confidence_level(model, test_loader, config)