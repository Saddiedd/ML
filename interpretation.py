from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix

from modeling import (
    BATCH_SIZE,
    MODELS_DIR,
    PREPROCESSED_PATH,
    VAL_LIMIT,
    build_mobilenet_v2,
    build_resnet18,
    make_loader,
    stratified_limit,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INTERPRETATION_DIR = MODELS_DIR / "interpretation"
INTERPRETATION_DIR.mkdir(exist_ok=True)

RESULTS_PATH = MODELS_DIR / "cnn_model_results.csv"
REPORT_PATH = MODELS_DIR / "interpretation_report.md"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def load_validation_data():
    if not PREPROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Файл не найден: {PREPROCESSED_PATH}. Сначала запустите preprocessing.py."
        )

    data = np.load(PREPROCESSED_PATH, allow_pickle=True)
    x_val = data["x_val"]
    y_val = data["y_val"]
    class_names = data["class_names"]

    x_val, y_val = stratified_limit(x_val, y_val, VAL_LIMIT)
    return x_val, y_val, class_names


def choose_best_model():
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Файл не найден: {RESULTS_PATH}. Сначала запустите modeling.py."
        )

    results_df = pd.read_csv(RESULTS_PATH)
    best_row = results_df.sort_values("val_accuracy", ascending=False).iloc[0]
    return best_row, results_df


def build_model(architecture, class_count):
    if architecture == "ResNet18":
        return build_resnet18(class_count)
    if architecture == "MobileNetV2":
        return build_mobilenet_v2(class_count)
    raise ValueError(f"Неизвестная архитектура: {architecture}")


def load_trained_model(model_name, model_path, class_count):
    checkpoint = torch.load(model_path, map_location=DEVICE)
    architecture = checkpoint.get("architecture", model_name)
    model = build_model(architecture, class_count)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


def predict_dataset(model, x_val, y_val):
    loader = make_loader(x_val, y_val, BATCH_SIZE, shuffle=False)
    y_true = []
    y_pred = []
    confidences = []

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(DEVICE)
            logits = model(x_batch)
            probabilities = torch.softmax(logits, dim=1)
            batch_confidences, batch_predictions = torch.max(probabilities, dim=1)

            y_true.extend(y_batch.numpy())
            y_pred.extend(batch_predictions.cpu().numpy())
            confidences.extend(batch_confidences.cpu().numpy())

    return np.array(y_true), np.array(y_pred), np.array(confidences)


def make_confusion_table(y_true, y_pred, class_names):
    matrix = confusion_matrix(y_true, y_pred)
    rows = []

    for true_id, true_name in enumerate(class_names):
        for pred_id, pred_name in enumerate(class_names):
            count = int(matrix[true_id, pred_id])
            if true_id != pred_id and count > 0:
                rows.append(
                    {
                        "true_class": str(true_name),
                        "predicted_class": str(pred_name),
                        "count": count,
                    }
                )

    return pd.DataFrame(rows).sort_values("count", ascending=False)


def make_class_metrics(y_true, y_pred, class_names):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    rows = []
    for class_name in class_names:
        metrics = report[str(class_name)]
        rows.append(
            {
                "class": str(class_name),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1-score"],
                "support": int(metrics["support"]),
            }
        )

    return pd.DataFrame(rows).sort_values("f1_score", ascending=False)


def save_prediction_examples(x_val, y_true, y_pred, confidences, class_names, correct):
    mask = y_true == y_pred if correct else y_true != y_pred
    indices = np.where(mask)[0]

    if len(indices) == 0:
        return None

    selected = indices[np.argsort(confidences[indices])[::-1][:12]]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()

    for axis, index in zip(axes, selected):
        axis.imshow(x_val[index])
        axis.axis("off")
        true_name = class_names[y_true[index]]
        pred_name = class_names[y_pred[index]]
        axis.set_title(
            f"true: {true_name}\npred: {pred_name}\nconf: {confidences[index]:.1%}",
            fontsize=9,
        )

    for axis in axes[len(selected) :]:
        axis.axis("off")

    file_name = "correct_predictions.png" if correct else "wrong_predictions.png"
    output_path = INTERPRETATION_DIR / file_name
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def write_report(
    best_row,
    results_df,
    class_metrics_df,
    confusions_df,
    y_true,
    y_pred,
    confidences,
    correct_examples_path,
    wrong_examples_path,
):
    accuracy = float((y_true == y_pred).mean())
    correct_confidence = float(confidences[y_true == y_pred].mean())
    wrong_confidence = float(confidences[y_true != y_pred].mean())

    best_classes = class_metrics_df.head(3)
    worst_classes = class_metrics_df.tail(3).sort_values("f1_score")
    top_confusions = confusions_df.head(10)

    model_rows = "\n".join(
        f"| {row.model} | {row.best_epoch} | {row.val_accuracy:.4f} | {row.val_loss:.4f} |"
        for row in results_df.itertuples(index=False)
    )
    best_class_rows = "\n".join(
        f"| {row['class']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} |"
        for _, row in best_classes.iterrows()
    )
    worst_class_rows = "\n".join(
        f"| {row['class']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} |"
        for _, row in worst_classes.iterrows()
    )
    confusion_rows = "\n".join(
        f"| {row.true_class} | {row.predicted_class} | {row.count} |"
        for row in top_confusions.itertuples(index=False)
    )

    report = f"""# Интерпретация и использование модели

## Выбранная модель

Для использования выбрана модель `{best_row["model"]}`, так как она показала лучший результат на validation-наборе.

| Модель | Лучшая эпоха | Validation accuracy | Validation loss |
|---|---:|---:|---:|
{model_rows}

## Качество выбранной модели

- Accuracy на validation-наборе: {accuracy:.4f}
- Средняя уверенность на правильных ответах: {correct_confidence:.1%}
- Средняя уверенность на ошибочных ответах: {wrong_confidence:.1%}

Если уверенность на ошибках высокая, это означает, что модель иногда ошибается достаточно уверенно. Такие случаи важны для ручной проверки.

## Лучше всего распознаются

| Класс | Precision | Recall | F1-score |
|---|---:|---:|---:|
{best_class_rows}

## Хуже всего распознаются

| Класс | Precision | Recall | F1-score |
|---|---:|---:|---:|
{worst_class_rows}

## Самые частые ошибки

| Истинный класс | Предсказанный класс | Количество |
|---|---|---:|
{confusion_rows}

## Практическое использование

Практическое использование модели реализовано в `telegram_bot.py`: пользователь отправляет фотографию, изображение приводится к формату 32x32 RGB, после чего модель возвращает наиболее вероятный класс и уверенность.

## Сохраненные материалы

- `{INTERPRETATION_DIR / "class_metrics.csv"}`
- `{INTERPRETATION_DIR / "top_confusions.csv"}`
- `{correct_examples_path}`
- `{wrong_examples_path}`
"""

    REPORT_PATH.write_text(report, encoding="utf-8")
    print("Отчет сохранен:", REPORT_PATH)


def main():
    print_header("ИНТЕРПРЕТАЦИЯ МОДЕЛИ")
    print("Устройство:", DEVICE)

    best_row, results_df = choose_best_model()
    x_val, y_val, class_names = load_validation_data()

    model = load_trained_model(
        best_row["model"],
        best_row["model_path"],
        class_count=len(class_names),
    )

    y_true, y_pred, confidences = predict_dataset(model, x_val, y_val)

    class_metrics_df = make_class_metrics(y_true, y_pred, class_names)
    confusions_df = make_confusion_table(y_true, y_pred, class_names)

    class_metrics_path = INTERPRETATION_DIR / "class_metrics.csv"
    confusions_path = INTERPRETATION_DIR / "top_confusions.csv"
    predictions_path = INTERPRETATION_DIR / "validation_predictions.csv"

    class_metrics_df.to_csv(class_metrics_path, index=False, encoding="utf-8")
    confusions_df.to_csv(confusions_path, index=False, encoding="utf-8")
    pd.DataFrame(
        {
            "true_class": [class_names[index] for index in y_true],
            "predicted_class": [class_names[index] for index in y_pred],
            "confidence": confidences,
        }
    ).to_csv(predictions_path, index=False, encoding="utf-8")

    correct_examples_path = save_prediction_examples(
        x_val, y_true, y_pred, confidences, class_names, correct=True
    )
    wrong_examples_path = save_prediction_examples(
        x_val, y_true, y_pred, confidences, class_names, correct=False
    )

    write_report(
        best_row,
        results_df,
        class_metrics_df,
        confusions_df,
        y_true,
        y_pred,
        confidences,
        correct_examples_path,
        wrong_examples_path,
    )

    print("Метрики классов сохранены:", class_metrics_path)
    print("Частые ошибки сохранены:", confusions_path)
    print("Предсказания сохранены:", predictions_path)


if __name__ == "__main__":
    main()
