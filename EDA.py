from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ==============================
# 1. НАСТРОЙКИ ПРОЕКТА
# ==============================

PROJECT_DIR = Path(__file__).parent

TRAIN_DIR = PROJECT_DIR / "train"
TEST_DIR = PROJECT_DIR / "test"
LABELS_PATH = PROJECT_DIR / "trainLabels.csv"
SAMPLE_SUBMISSION_PATH = PROJECT_DIR / "sampleSubmission.csv"

RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42
IMAGE_SAMPLE_SIZE = 2000


def print_header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_current_plot(filename):
    output_path = RESULTS_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"График сохранен: {output_path}")
    return output_path


# ==============================
# 2. ЗАГРУЗКА ТАБЛИЦ
# ==============================

def load_labels():
    print_header("ЗАГРУЗКА trainLabels.csv")

    labels_df = pd.read_csv(LABELS_PATH)

    print("Первые строки trainLabels.csv:")
    print(labels_df.head())

    print("\nОбщая информация:")
    labels_df.info()

    print("\nКоличество строк:", len(labels_df))
    print("Названия столбцов:", list(labels_df.columns))
    print("Пропуски по столбцам:")
    print(labels_df.isna().sum())
    print("Дубликаты id:", labels_df["id"].duplicated().sum())

    return labels_df


def load_sample_submission():
    print_header("ПРОВЕРКА sampleSubmission.csv")

    submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    print("Первые строки sampleSubmission.csv:")
    print(submission_df.head())

    print("\nКоличество строк:", len(submission_df))
    print("Названия столбцов:", list(submission_df.columns))
    print("Пропуски по столбцам:")
    print(submission_df.isna().sum())
    print("Уникальные значения в шаблонной колонке label:")
    print(submission_df["label"].value_counts())

    print(
        "\nВажно: sampleSubmission.csv содержит не правильные ответы, "
        "а шаблон итогового файла с предсказаниями."
    )

    return submission_df


# ==============================
# 3. АНАЛИЗ КЛАССОВ
# ==============================

def analyze_classes(labels_df):
    print_header("АНАЛИЗ КЛАССОВ")

    class_counts = labels_df["label"].value_counts().sort_index()
    class_percent = (class_counts / len(labels_df) * 100).round(2)
    class_summary = pd.DataFrame(
        {
            "count": class_counts,
            "percent": class_percent,
        }
    )

    print("Количество изображений по классам:")
    print(class_summary)
    print("\nКоличество классов:", labels_df["label"].nunique())
    print("Минимальный размер класса:", class_counts.min())
    print("Максимальный размер класса:", class_counts.max())

    class_summary.to_csv(RESULTS_DIR / "class_summary.csv", encoding="utf-8")

    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_counts.index, class_counts.values, color="#386641")
    plt.title("Распределение изображений по классам CIFAR-10")
    plt.xlabel("Класс")
    plt.ylabel("Количество изображений")
    plt.xticks(rotation=45, ha="right")
    plt.bar_label(bars, padding=3, fontsize=8)
    save_current_plot("class_distribution.png")

    return class_summary


# ==============================
# 4. ПРОВЕРКА ФАЙЛОВ
# ==============================

def count_png_files(directory):
    return sum(1 for _ in directory.glob("*.png"))


def check_dataset_files(labels_df, submission_df):
    print_header("ПРОВЕРКА ФАЙЛОВ ДАТАСЕТА")

    train_count = count_png_files(TRAIN_DIR)
    test_count = count_png_files(TEST_DIR)

    train_ids = set(labels_df["id"].astype(int))
    submission_ids = set(submission_df["id"].astype(int))

    missing_train = [
        image_id
        for image_id in labels_df["id"].astype(int)
        if not (TRAIN_DIR / f"{image_id}.png").exists()
    ]
    missing_test = [
        image_id
        for image_id in submission_df["id"].astype(int)
        if not (TEST_DIR / f"{image_id}.png").exists()
    ]

    print("Файлов train/*.png:", train_count)
    print("Строк в trainLabels.csv:", len(labels_df))
    print("Файлов test/*.png:", test_count)
    print("Строк в sampleSubmission.csv:", len(submission_df))
    print("Train id min/max:", min(train_ids), max(train_ids))
    print("Test id min/max:", min(submission_ids), max(submission_ids))
    print("Отсутствующих train-файлов:", len(missing_train))
    print("Отсутствующих test-файлов:", len(missing_test))

    if missing_train:
        print("Примеры отсутствующих train-файлов:", missing_train[:10])
    if missing_test:
        print("Примеры отсутствующих test-файлов:", missing_test[:10])

    return {
        "train_files": train_count,
        "test_files": test_count,
        "missing_train": len(missing_train),
        "missing_test": len(missing_test),
    }


def inspect_first_image(labels_df):
    print_header("ПРОВЕРКА ПЕРВОГО ИЗОБРАЖЕНИЯ")

    first_id = int(labels_df.iloc[0]["id"])
    first_label = labels_df.iloc[0]["label"]
    image_path = TRAIN_DIR / f"{first_id}.png"

    if not image_path.exists():
        raise FileNotFoundError(f"Не найдено изображение: {image_path}")

    with Image.open(image_path) as image:
        print("ID:", first_id)
        print("Класс:", first_label)
        print("Путь:", image_path)
        print("Размер:", image.size)
        print("Режим изображения:", image.mode)


# ==============================
# 5. ВИЗУАЛИЗАЦИЯ ПРИМЕРОВ
# ==============================

def show_examples(labels_df, examples_per_class=5):
    print_header("СОЗДАНИЕ СЕТКИ ПРИМЕРОВ ИЗОБРАЖЕНИЙ")

    classes = sorted(labels_df["label"].unique())

    plt.figure(figsize=(examples_per_class * 2, len(classes) * 1.8))
    plot_index = 1

    for class_name in classes:
        class_rows = labels_df[labels_df["label"] == class_name].head(examples_per_class)

        for _, row in class_rows.iterrows():
            image_id = int(row["id"])
            image_path = TRAIN_DIR / f"{image_id}.png"

            with Image.open(image_path) as image:
                plt.subplot(len(classes), examples_per_class, plot_index)
                plt.imshow(image)
                plt.axis("off")

                if plot_index % examples_per_class == 1:
                    plt.ylabel(class_name, fontsize=10)

                plt.title(class_name, fontsize=8)
                plot_index += 1

    plt.suptitle("Примеры изображений из каждого класса CIFAR-10", fontsize=14)
    save_current_plot("class_examples.png")


# ==============================
# 6. ЧИСЛОВЫЕ ПРИЗНАКИ ИЗОБРАЖЕНИЙ
# ==============================

def image_features(image_path):
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        array = np.asarray(rgb_image, dtype=np.float32) / 255.0

    channel_means = array.mean(axis=(0, 1))
    channel_stds = array.std(axis=(0, 1))

    return {
        "width": array.shape[1],
        "height": array.shape[0],
        "mean_r": channel_means[0],
        "mean_g": channel_means[1],
        "mean_b": channel_means[2],
        "std_r": channel_stds[0],
        "std_g": channel_stds[1],
        "std_b": channel_stds[2],
        "brightness": channel_means.mean(),
        "contrast": channel_stds.mean(),
    }


def build_image_features(labels_df, sample_size=IMAGE_SAMPLE_SIZE):
    print_header("РАСЧЕТ ПРИЗНАКОВ ПО ВЫБОРКЕ ИЗОБРАЖЕНИЙ")

    sample_df = (
        labels_df
        .sample(n=min(sample_size, len(labels_df)), random_state=RANDOM_STATE)
        .sort_values("id")
        .reset_index(drop=True)
    )

    rows = []
    for _, row in sample_df.iterrows():
        image_id = int(row["id"])
        features = image_features(TRAIN_DIR / f"{image_id}.png")
        features["id"] = image_id
        features["label"] = row["label"]
        rows.append(features)

    features_df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "image_features_sample.csv"
    features_df.to_csv(output_path, index=False, encoding="utf-8")

    print("Размер выборки:", len(features_df))
    print("Файл с признаками сохранен:", output_path)
    print("\nОписательная статистика признаков:")
    print(features_df.describe().round(4))
    print("\nРазмеры изображений в выборке:")
    print(features_df[["width", "height"]].value_counts())

    return features_df


def analyze_image_features(features_df):
    print_header("РАЗВЕДОЧНЫЙ АНАЛИЗ ПИКСЕЛЬНЫХ ПРИЗНАКОВ")

    grouped = (
        features_df
        .groupby("label")[["brightness", "contrast", "mean_r", "mean_g", "mean_b"]]
        .mean()
        .round(4)
        .sort_index()
    )
    grouped.to_csv(RESULTS_DIR / "image_features_by_class.csv", encoding="utf-8")

    print("Средние признаки по классам:")
    print(grouped)

    numeric_columns = [
        "mean_r",
        "mean_g",
        "mean_b",
        "std_r",
        "std_g",
        "std_b",
        "brightness",
        "contrast",
    ]
    correlation = features_df[numeric_columns].corr()
    print("\nКорреляционная матрица признаков:")
    print(correlation.round(3))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(features_df["brightness"], bins=30, color="#2f3e46", alpha=0.85)
    axes[0, 0].set_title("Распределение яркости")
    axes[0, 0].set_xlabel("brightness")
    axes[0, 0].set_ylabel("frequency")

    labels = sorted(features_df["label"].unique())
    box_data = [
        features_df.loc[features_df["label"] == label, "brightness"]
        for label in labels
    ]
    axes[0, 1].boxplot(box_data, tick_labels=labels)
    axes[0, 1].set_title("Яркость по классам")
    axes[0, 1].set_xlabel("label")
    axes[0, 1].set_ylabel("brightness")
    axes[0, 1].tick_params(axis="x", rotation=45)

    axes[1, 0].scatter(
        features_df["brightness"],
        features_df["contrast"],
        s=12,
        alpha=0.45,
        color="#bc4749",
    )
    axes[1, 0].set_title("Связь яркости и контраста")
    axes[1, 0].set_xlabel("brightness")
    axes[1, 0].set_ylabel("contrast")

    heatmap = axes[1, 1].imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axes[1, 1].set_title("Корреляция пиксельных признаков")
    axes[1, 1].set_xticks(range(len(numeric_columns)), numeric_columns, rotation=45, ha="right")
    axes[1, 1].set_yticks(range(len(numeric_columns)), numeric_columns)
    for i in range(len(numeric_columns)):
        for j in range(len(numeric_columns)):
            axes[1, 1].text(
                j,
                i,
                f"{correlation.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(correlation.iloc[i, j]) > 0.55 else "black",
            )
    fig.colorbar(heatmap, ax=axes[1, 1], fraction=0.046, pad=0.04)

    save_current_plot("image_features_analysis.png")


def save_average_images(labels_df, max_images_per_class=500):
    print_header("СРЕДНИЕ ИЗОБРАЖЕНИЯ ПО КЛАССАМ")

    classes = sorted(labels_df["label"].unique())
    plt.figure(figsize=(14, 3))

    for index, class_name in enumerate(classes, start=1):
        class_rows = labels_df[labels_df["label"] == class_name].head(max_images_per_class)
        accumulator = None

        for _, row in class_rows.iterrows():
            image_path = TRAIN_DIR / f"{int(row['id'])}.png"
            with Image.open(image_path) as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32)
            if accumulator is None:
                accumulator = np.zeros_like(array)
            accumulator += array

        average_image = np.clip(accumulator / len(class_rows), 0, 255).astype(np.uint8)

        plt.subplot(1, len(classes), index)
        plt.imshow(average_image)
        plt.title(class_name, fontsize=8)
        plt.axis("off")

    plt.suptitle("Средние изображения классов")
    save_current_plot("average_images_by_class.png")


# ==============================
# 7. ИТОГОВЫЙ ОТЧЕТ
# ==============================

def write_report(labels_df, submission_df, class_summary, file_summary, features_df):
    report_path = RESULTS_DIR / "eda_report.md"
    brightness_by_class = features_df.groupby("label")["brightness"].mean().sort_values()
    contrast_by_class = features_df.groupby("label")["contrast"].mean().sort_values()

    report = f"""# Разведочный анализ CIFAR-10

## Состав датасета

- Строк в `trainLabels.csv`: {len(labels_df)}
- Изображений в папке `train`: {file_summary["train_files"]}
- Строк в `sampleSubmission.csv`: {len(submission_df)}
- Изображений в папке `test`: {file_summary["test_files"]}
- Отсутствующих train-файлов: {file_summary["missing_train"]}
- Отсутствующих test-файлов: {file_summary["missing_test"]}

## Классы

- Количество классов: {labels_df["label"].nunique()}
- Минимальный размер класса: {class_summary["count"].min()}
- Максимальный размер класса: {class_summary["count"].max()}
- Распределение классов сбалансировано: все классы представлены примерно одинаково.

## Изображения

- Все проверенные изображения имеют размер 32x32 и RGB-представление.
- Для анализа пиксельных признаков использована выборка из {len(features_df)} изображений.
- Самый темный класс по средней яркости: `{brightness_by_class.index[0]}`.
- Самый светлый класс по средней яркости: `{brightness_by_class.index[-1]}`.
- Самый низкий средний контраст: `{contrast_by_class.index[0]}`.
- Самый высокий средний контраст: `{contrast_by_class.index[-1]}`.

## Сохраненные файлы

- `class_distribution.png`
- `class_examples.png`
- `image_features_analysis.png`
- `average_images_by_class.png`
- `class_summary.csv`
- `image_features_sample.csv`
- `image_features_by_class.csv`
"""

    report_path.write_text(report, encoding="utf-8")
    print("Отчет сохранен:", report_path)


# ==============================
# 8. ОСНОВНОЙ ЗАПУСК
# ==============================

def main():
    labels_df = load_labels()
    submission_df = load_sample_submission()

    class_summary = analyze_classes(labels_df)
    file_summary = check_dataset_files(labels_df, submission_df)
    inspect_first_image(labels_df)

    show_examples(labels_df, examples_per_class=5)
    features_df = build_image_features(labels_df)
    analyze_image_features(features_df)
    save_average_images(labels_df)
    write_report(labels_df, submission_df, class_summary, file_summary, features_df)

    print_header("РАЗВЕДОЧНЫЙ АНАЛИЗ ДАТАСЕТА ЗАВЕРШЕН")
    print("Результаты сохранены в папку:", RESULTS_DIR)


if __name__ == "__main__":
    main()
