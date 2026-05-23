from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import mobilenet_v2, resnet18

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).parent
PREPROCESSED_PATH = PROJECT_DIR / "preprocessed" / "cifar10_train_val_preprocessed.npz"
MODELS_DIR = PROJECT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
TRAIN_LIMIT = 12000
VAL_LIMIT = 3000
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def set_random_seed():
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    print(f'CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)



def load_preprocessed_data():
    print_header("ЗАГРУЗКА ПРЕДОБРАБОТАННЫХ ДАННЫХ")

    if not PREPROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Файл не найден: {PREPROCESSED_PATH}. Сначала запустите preprocessing.py."
        )

    data = np.load(PREPROCESSED_PATH, allow_pickle=True)
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_val = data["x_val"]
    y_val = data["y_val"]
    class_names = data["class_names"]

    print("x_train:", x_train.shape)
    print("y_train:", y_train.shape)
    print("x_val:", x_val.shape)
    print("y_val:", y_val.shape)
    print("Классы:", list(class_names))

    return x_train, y_train, x_val, y_val, class_names


def stratified_limit(x, y, limit):
    if limit is None or limit >= len(y):
        return x, y

    rng = np.random.default_rng(RANDOM_STATE)
    selected_indices = []
    classes = np.unique(y)
    per_class = max(1, limit // len(classes))

    for class_id in classes:
        class_indices = np.where(y == class_id)[0]
        take_count = min(per_class, len(class_indices))
        selected_indices.extend(rng.choice(class_indices, size=take_count, replace=False))

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    return x[selected_indices], y[selected_indices]


def to_torch_images(x):
    if x.ndim != 4:
        raise ValueError("CNN ожидает изображения формы (N, height, width, channels).")

    x_channels_first = np.transpose(x, (0, 3, 1, 2))
    return torch.tensor(x_channels_first, dtype=torch.float32)


def make_loader(x, y, batch_size, shuffle):
    x_tensor = to_torch_images(x)
    y_tensor = torch.tensor(y, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_resnet18(class_count):
    model = resnet18(weights=None)

    # CIFAR-10 имеет размер 32x32: делаем первый слой мягче и отключаем maxpool.
    model.conv1 = nn.Conv2d(
        3,
        64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, class_count)
    return model


def build_mobilenet_v2(class_count):
    model = mobilenet_v2(weights=None)

    # MobileNetV2 тоже готовая CNN. Для маленьких 32x32 изображений уменьшаем
    # агрессивность первого свёрточного слоя: stride 1 вместо stride 2.
    model.features[0][0].stride = (1, 1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, class_count)
    return model


def make_cnn_models(class_count):
    return {
        "ResNet18": build_resnet18(class_count),
        "MobileNetV2": build_mobilenet_v2(class_count),
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)

    total_loss = 0.0
    correct = 0
    total = 0
    all_true = []
    all_pred = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        if is_training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_training):
            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            if is_training:
                loss.backward()
                optimizer.step()

        predictions = logits.argmax(dim=1)
        batch_size = y_batch.size(0)

        total_loss += loss.item() * batch_size
        correct += (predictions == y_batch).sum().item()
        total += batch_size
        all_true.extend(y_batch.detach().cpu().numpy())
        all_pred.extend(predictions.detach().cpu().numpy())

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "y_true": np.array(all_true),
        "y_pred": np.array(all_pred),
    }


def train_cnn_model(model_name, model, train_loader, val_loader, class_names, device):
    print_header(f"{model_name}: ОБУЧЕНИЕ CNN")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val_accuracy = -1.0
    best_model_path = MODELS_DIR / f"{model_name}_best.pth"

    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
        }
        history.append(row)

        if val_metrics["accuracy"] > best_val_accuracy:
            best_val_accuracy = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": list(class_names),
                    "architecture": model_name,
                },
                best_model_path,
            )

        print(
            f"Epoch {epoch:02d}: "
            f"train_acc={row['train_accuracy']:.4f}, "
            f"val_acc={row['val_accuracy']:.4f}, "
            f"train_loss={row['train_loss']:.4f}, "
            f"val_loss={row['val_loss']:.4f}"
        )

    history_df = pd.DataFrame(history)
    history_path = MODELS_DIR / f"{model_name}_epoch_history.csv"
    history_df.to_csv(history_path, index=False, encoding="utf-8")

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = run_epoch(model, val_loader, criterion, device)

    print(f"\nИтоговый отчет {model_name}:")
    print(
        classification_report(
            final_metrics["y_true"],
            final_metrics["y_pred"],
            target_names=class_names,
        )
    )

    plot_training_history(history_df, model_name)
    save_confusion_matrix(
        final_metrics["y_true"],
        final_metrics["y_pred"],
        class_names,
        model_name,
    )

    best_epoch = history_df.sort_values("val_accuracy", ascending=False).iloc[0]
    print(f"История {model_name} сохранена:", history_path)
    print(f"Лучшая модель {model_name} сохранена:", best_model_path)

    return {
        "model": model_name,
        "best_epoch": int(best_epoch["epoch"]),
        "val_accuracy": float(best_epoch["val_accuracy"]),
        "val_loss": float(best_epoch["val_loss"]),
        "model_path": str(best_model_path),
        "history_path": str(history_path),
    }


def plot_training_history(history_df, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o", label="train loss")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], marker="o", label="val loss")
    axes[0].set_title(f"{model_name} loss по эпохам")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], marker="o", label="train accuracy")
    axes[1].plot(history_df["epoch"], history_df["val_accuracy"], marker="o", label="val accuracy")
    axes[1].set_title(f"{model_name} accuracy по эпохам")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    output_path = MODELS_DIR / f"{model_name}_loss_accuracy_by_epoch.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"График {model_name} loss/accuracy сохранен:", output_path)


def save_confusion_matrix(y_true, y_pred, class_names, model_name):
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(9, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(f"Confusion matrix: {model_name}")
    plt.tight_layout()

    output_path = MODELS_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Матрица ошибок сохранена:", output_path)


def write_modeling_report(results_df):
    best_model = results_df.sort_values("val_accuracy", ascending=False).iloc[0]
    report_path = MODELS_DIR / "modeling_report.md"

    rows = "\n".join(
        f"| {row.model} | {row.best_epoch} | {row.val_accuracy:.4f} | {row.val_loss:.4f} |"
        for row in results_df.itertuples(index=False)
    )

    report = f"""# Построение моделей машинного обучения

## Использованные модели

В работе используются две готовые сверточные нейронные сети из `torchvision`:

- `ResNet18`
- `MobileNetV2`

Обе модели обучаются на предобработанных изображениях CIFAR-10 размером 32x32.

## Данные

- Источник: `preprocessed/cifar10_train_val_preprocessed.npz`.
- Изображения уже приведены к `float32` и нормализованы в диапазон от 0 до 1.
- Для обучения использовано train: {TRAIN_LIMIT}.
- Для проверки использовано validation: {VAL_LIMIT}.

## Сравнение CNN

| Модель | Лучшая эпоха | Validation accuracy | Validation loss |
|---|---:|---:|---:|
{rows}

## Лучшая модель

- Модель: `{best_model["model"]}`
- Validation accuracy: {best_model["val_accuracy"]:.4f}
- Validation loss: {best_model["val_loss"]:.4f}

## Сохраненные файлы

- `models/ResNet18_best.pth`
- `models/MobileNetV2_best.pth`
- `models/ResNet18_epoch_history.csv`
- `models/MobileNetV2_epoch_history.csv`
- графики loss/accuracy и матрицы ошибок для каждой модели
"""

    report_path.write_text(report, encoding="utf-8")
    print("Отчет сохранен:", report_path)


def main():
    set_random_seed()
    x_train, y_train, x_val, y_val, class_names = load_preprocessed_data()

    x_train, y_train = stratified_limit(x_train, y_train, TRAIN_LIMIT)
    x_val, y_val = stratified_limit(x_val, y_val, VAL_LIMIT)

    print("Train для CNN:", x_train.shape)
    print("Validation для CNN:", x_val.shape)

    train_loader = make_loader(x_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(x_val, y_val, BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Устройство:", device)

    rows = []
    for model_name, model in make_cnn_models(class_count=len(class_names)).items():
        rows.append(
            train_cnn_model(
                model_name,
                model,
                train_loader,
                val_loader,
                class_names,
                device,
            )
        )

    results_df = pd.DataFrame(rows).sort_values("val_accuracy", ascending=False)
    results_path = MODELS_DIR / "cnn_model_results.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8")
    write_modeling_report(results_df)

    print_header("ЭТАП МОДЕЛИРОВАНИЯ ЗАВЕРШЕН")
    print("Результаты сохранены в папку:", MODELS_DIR)


if __name__ == "__main__":
    main()
