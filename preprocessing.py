from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# ==============================
# 1. НАСТРОЙКИ ПРОЕКТА
# ==============================

PROJECT_DIR = Path(__file__).parent

TRAIN_DIR = PROJECT_DIR / "train"
TEST_DIR = PROJECT_DIR / "test"
LABELS_PATH = PROJECT_DIR / "trainLabels.csv"
SAMPLE_SUBMISSION_PATH = PROJECT_DIR / "sampleSubmission.csv"

PROCESSED_DIR = PROJECT_DIR / "preprocessed"
PROCESSED_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
VALIDATION_SIZE = 0.2
IMAGE_SIZE = (32, 32)


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ==============================
# 2. РАЗМЕТКА: LABEL ENCODER
# ==============================

def load_and_encode_labels():
    print_header("ЗАГРУЗКА И КОДИРОВАНИЕ МЕТОК")

    labels_df = pd.read_csv(LABELS_PATH)

    encoder = LabelEncoder()
    labels_df["label_id"] = encoder.fit_transform(labels_df["label"])
    class_names = encoder.classes_

    mapping_df = pd.DataFrame(
        {
            "label_id": np.arange(len(class_names)),
            "label": class_names,
        }
    )
    mapping_path = PROCESSED_DIR / "label_mapping.csv"
    mapping_df.to_csv(mapping_path, index=False, encoding="utf-8")

    print("Количество изображений:", len(labels_df))
    print("Количество классов:", len(class_names))
    print("Кодировка классов:")
    print(mapping_df)
    print("Кодировка сохранена:", mapping_path)

    return labels_df, class_names


# ==============================
# 3. TRAIN/VALIDATION: TRAIN_TEST_SPLIT
# ==============================

def make_train_validation_split(labels_df):
    print_header("СТРАТИФИЦИРОВАННОЕ РАЗБИЕНИЕ")

    train_df, val_df = train_test_split(
        labels_df,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels_df["label_id"],
        shuffle=True,
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_df.to_csv(PROCESSED_DIR / "train_split.csv", index=False, encoding="utf-8")
    val_df.to_csv(PROCESSED_DIR / "validation_split.csv", index=False, encoding="utf-8")

    print("Train:", len(train_df))
    print("Validation:", len(val_df))
    print("\nРаспределение классов в train:")
    print(train_df["label"].value_counts().sort_index())
    print("\nРаспределение классов в validation:")
    print(val_df["label"].value_counts().sort_index())

    return train_df, val_df


# ==============================
# 4. ИЗОБРАЖЕНИЯ: PIL + NUMPY
# ==============================

def preprocess_image(image_id, directory):
    image_path = directory / f"{int(image_id)}.png"

    if not image_path.exists():
        raise FileNotFoundError(f"Не найден файл: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        return np.asarray(image, dtype=np.float32) / 255.0


def build_image_array(data_df, directory):
    image_ids = data_df["id"].to_numpy(dtype=np.int64)
    labels = data_df["label_id"].to_numpy(dtype=np.int64)
    images = np.empty((len(data_df), IMAGE_SIZE[1], IMAGE_SIZE[0], 3), dtype=np.float32)

    for index, image_id in enumerate(image_ids):
        images[index] = preprocess_image(image_id, directory)

        if (index + 1) % 5000 == 0:
            print(f"Обработано изображений: {index + 1}/{len(data_df)}")

    return images, labels, image_ids


def preprocess_train_validation(train_df, val_df, class_names):
    print_header("ПРЕДОБРАБОТКА TRAIN И VALIDATION")

    print("Загрузка train...")
    x_train, y_train, train_ids = build_image_array(train_df, TRAIN_DIR)

    print("Загрузка validation...")
    x_val, y_val, val_ids = build_image_array(val_df, TRAIN_DIR)

    output_path = PROCESSED_DIR / "cifar10_train_val_preprocessed.npz"
    np.savez_compressed(
        output_path,
        x_train=x_train,
        y_train=y_train,
        train_ids=train_ids,
        x_val=x_val,
        y_val=y_val,
        val_ids=val_ids,
        class_names=class_names,
    )

    summary = {
        "output_path": output_path,
        "x_train_shape": x_train.shape,
        "x_val_shape": x_val.shape,
        "train_pixel_min": float(x_train.min()),
        "train_pixel_max": float(x_train.max()),
        "val_pixel_min": float(x_val.min()),
        "val_pixel_max": float(x_val.max()),
    }

    print("Файл сохранен:", output_path)
    print("x_train shape:", summary["x_train_shape"])
    print("x_val shape:", summary["x_val_shape"])
    print("Диапазон пикселей train:", summary["train_pixel_min"], summary["train_pixel_max"])
    print("Диапазон пикселей validation:", summary["val_pixel_min"], summary["val_pixel_max"])

    return summary


# ==============================
# 5. TEST-МЕТАДАННЫЕ
# ==============================

def prepare_test_metadata():
    print_header("ПОДГОТОВКА TEST-МЕТАДАННЫХ")

    sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    test_ids = sample_submission_df["id"].to_numpy(dtype=np.int64)

    missing_test = [
        image_id
        for image_id in test_ids
        if not (TEST_DIR / f"{int(image_id)}.png").exists()
    ]

    test_ids_path = PROCESSED_DIR / "test_ids.csv"
    pd.DataFrame({"id": test_ids}).to_csv(test_ids_path, index=False, encoding="utf-8")

    print("Количество test-изображений:", len(test_ids))
    print("Отсутствующих test-файлов:", len(missing_test))
    print("ID test-набора сохранены:", test_ids_path)
    print("Test-набор лучше нормализовать пакетами перед предсказанием, а не хранить одним огромным массивом.")

    return {
        "test_count": len(test_ids),
        "missing_test": len(missing_test),
    }


# ==============================
# 6. ОТЧЕТ
# ==============================

def write_report(train_df, val_df, preprocessing_summary, test_summary):
    report_path = PROCESSED_DIR / "preprocessing_report.md"

    report = f"""# Предобработка CIFAR-10

## Использованные библиотеки

- `pandas` для чтения таблиц и сохранения метаданных.
- `sklearn.preprocessing.LabelEncoder` для кодирования текстовых меток.
- `sklearn.model_selection.train_test_split` для стратифицированного разбиения.
- `Pillow` и `numpy` для чтения, resize и нормализации изображений.

## Что сделано

- Метки классов закодированы числами от 0 до 9.
- Датасет разбит на train и validation со стратификацией по классам.
- Изображения переведены в RGB.
- Размер изображений приведен к {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}.
- Пиксели нормализованы в диапазон 0..1.
- Test-набор подготовлен как список ID для пакетной обработки.

## Размеры выборок

- Train: {len(train_df)}
- Validation: {len(val_df)}
- Test: {test_summary["test_count"]}
- Отсутствующих test-файлов: {test_summary["missing_test"]}

## Проверка массивов

- `x_train`: {preprocessing_summary["x_train_shape"]}
- `x_val`: {preprocessing_summary["x_val_shape"]}
- Диапазон пикселей train: {preprocessing_summary["train_pixel_min"]}..{preprocessing_summary["train_pixel_max"]}
- Диапазон пикселей validation: {preprocessing_summary["val_pixel_min"]}..{preprocessing_summary["val_pixel_max"]}

## Сохраненные файлы

- `cifar10_train_val_preprocessed.npz`
- `label_mapping.csv`
- `train_split.csv`
- `validation_split.csv`
- `test_ids.csv`
"""

    report_path.write_text(report, encoding="utf-8")
    print("Отчет сохранен:", report_path)


# ==============================
# 7. ОСНОВНОЙ ЗАПУСК
# ==============================

def main():
    labels_df, class_names = load_and_encode_labels()
    train_df, val_df = make_train_validation_split(labels_df)
    preprocessing_summary = preprocess_train_validation(train_df, val_df, class_names)
    test_summary = prepare_test_metadata()
    write_report(train_df, val_df, preprocessing_summary, test_summary)

    print_header("ПРЕДОБРАБОТКА ЗАВЕРШЕНА")
    print("Результаты сохранены в папку:", PROCESSED_DIR)


if __name__ == "__main__":
    main()
