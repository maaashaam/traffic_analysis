from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from salaryreg.config import ModelConfig
from salaryreg.io_npy import load_x, load_y
from salaryreg.nn import FCNRegressor
from salaryreg.nn_out import Scaler, save_model, save_scaler


@dataclass(frozen=True)
class TrainParams:
    test_size: float = 0.2
    val_size: float = 0.15
    seed: int = 42

    hidden_dims: tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.05

    epochs: int = 120
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    use_log_target: bool = True

    experiment_name: str = "LIne Regression HH"
    registered_model_name: str = "mashkovtseva_maria_fcn"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(batch, device: torch.device):
    xb, yb = batch
    return xb.to(device), yb.to(device)


def _eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="salaryreg.train_cli",
        description="Train FCN regressor with MLflow tracking (experiment: LIne Regression HH)",
    )
    parser.add_argument("x_path", type=str, help="Path to x_data.npy")
    parser.add_argument("y_path", type=str, help="Path to y_data.npy")

    parser.add_argument("--model-name", type=str, default=None, help="MLflow registered model name (<fio>_fcn)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-log", action="store_true", help="Disable log1p target")
    args = parser.parse_args()

    base_cfg = ModelConfig()
    p = TrainParams(
        seed=args.seed if args.seed is not None else TrainParams.seed,
        epochs=args.epochs if args.epochs is not None else TrainParams.epochs,
        batch_size=args.batch_size if args.batch_size is not None else TrainParams.batch_size,
        lr=args.lr if args.lr is not None else TrainParams.lr,
        weight_decay=args.weight_decay if args.weight_decay is not None else TrainParams.weight_decay,
        use_log_target=not args.no_log,
        experiment_name=getattr(base_cfg, "experiment_name", TrainParams.experiment_name),
        registered_model_name=args.model_name
        or getattr(base_cfg, "registered_model_name", TrainParams.registered_model_name),
    )

    seed_everything(p.seed)

    x = load_x(Path(args.x_path))
    y = load_y(Path(args.y_path)).reshape(-1)

    mask = np.isfinite(x).all(axis=1) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    # split train/test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=p.test_size, random_state=p.seed
    )

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=p.val_size,
        random_state=p.seed,
    )

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-8] = 1.0
    scaler = Scaler(mean=mean, std=std, use_log_target=p.use_log_target)

    x_train_n = scaler.transform(x_train).astype(np.float32, copy=False)
    x_val_n = scaler.transform(x_val).astype(np.float32, copy=False)
    x_test_n = scaler.transform(x_test).astype(np.float32, copy=False)

    y_train_t = np.log1p(y_train) if p.use_log_target else y_train
    y_val_t = np.log1p(y_val) if p.use_log_target else y_val
    y_test_t = np.log1p(y_test) if p.use_log_target else y_test

    # datasets
    ds_train = TensorDataset(
        torch.from_numpy(x_train_n),
        torch.from_numpy(y_train_t.reshape(-1, 1).astype(np.float32)),
    )
    dl_train = DataLoader(ds_train, batch_size=p.batch_size, shuffle=True)

    # model
    model = FCNRegressor(
        input_dim=x.shape[1],
        hidden_dims=list(p.hidden_dims),
        dropout=p.dropout,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=p.lr, weight_decay=p.weight_decay)
    loss_fn = nn.SmoothL1Loss(beta=1.0)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4
    )

    project_dir = Path(__file__).resolve().parents[1]
    resources_dir = project_dir / "resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    model_path = resources_dir / "model.pt"
    scaler_path = resources_dir / "scaler.npz"

    mlflow.set_tracking_uri("http://kamnsv.com:55000")
    mlflow.set_experiment(p.experiment_name)
    run_name = f"{p.registered_model_name}_lr{p.lr}_bs{p.batch_size}_ep{p.epochs}"

    start_time = time.time()

    with mlflow.start_run(run_name=run_name) as run:
        # log params
        mlflow.log_params(
            {
                "model_name": p.registered_model_name,
                "model_type": "fcn",
                "hidden_dims": list(p.hidden_dims),
                "dropout": p.dropout,
                "lr": p.lr,
                "weight_decay": p.weight_decay,
                "batch_size": p.batch_size,
                "epochs": p.epochs,
                "patience": p.patience,
                "seed": p.seed,
                "use_log_target": p.use_log_target,
                "n_features": int(x.shape[1]),
                "train_rows": int(x_train.shape[0]),
                "val_rows": int(x_val.shape[0]),
                "test_rows": int(x_test.shape[0]),
            }
        )

        best_val = float("inf")
        best_epoch = 0
        best_state = None

        model.train()
        for epoch in range(1, p.epochs + 1):
            total = 0.0
            for batch in dl_train:
                xb, yb = _to_device(batch, device)

                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()

                total += float(loss.item()) * xb.size(0)

            train_loss = total / len(ds_train)

            model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(x_val_n).to(device)
                yv = torch.from_numpy(y_val_t.reshape(-1, 1).astype(np.float32)).to(device)
                pv = model(xv)
                val_loss = float(loss_fn(pv, yv).item())
            model.train()

            scheduler.step(val_loss)
            lr_now = float(opt.param_groups[0]["lr"])

            mlflow.log_metric("train_loss", float(train_loss), step=epoch)
            mlflow.log_metric("val_loss", float(val_loss), step=epoch)
            mlflow.log_metric("lr", lr_now, step=epoch)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            if epoch - best_epoch >= p.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(x_test_n).to(device)
            pred_t = model(xt).detach().cpu().numpy().reshape(-1)

        y_pred = np.expm1(pred_t) if p.use_log_target else pred_t
        y_pred = np.clip(y_pred, base_cfg.clip_min_pred, None)

        metrics = _eval_metrics(y_test, y_pred)
        mlflow.log_metric("mae_test", metrics["mae"])
        mlflow.log_metric("rmse_test", metrics["rmse"])

        mlflow.log_metric("r2_score_test", metrics["r2"])

        save_model(model_path, model.cpu())
        save_scaler(scaler_path, scaler)

        mlflow.log_artifact(str(model_path), artifact_path="resources")
        mlflow.log_artifact(str(scaler_path), artifact_path="resources")

        mlflow.pytorch.log_model(
            pytorch_model=model.cpu(),
            artifact_path="model",
            registered_model_name=p.registered_model_name,
        )

        summary = {
            "run_id": run.info.run_id,
            "experiment": p.experiment_name,
            "registered_model_name": p.registered_model_name,
            "best_epoch": best_epoch,
            "val_loss_best": best_val,
            "metrics_test": metrics,
            "train_params": asdict(p),
        }
        tmp = project_dir / "run_summary.json"
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(tmp), artifact_path="meta")
        tmp.unlink(missing_ok=True)

        elapsed = time.time() - start_time
        mlflow.log_metric("train_seconds", float(elapsed))

        print("RUN_ID:", run.info.run_id)
        print("r2_score_test:", metrics["r2"])
        print("Saved:", model_path)
        print("Saved:", scaler_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())