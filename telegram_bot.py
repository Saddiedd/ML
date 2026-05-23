import asyncio
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from modeling import build_resnet18


PROJECT_DIR = Path(__file__).parent
MODEL_PATH = PROJECT_DIR / "models" / "ResNet18_best.pth"
PROXY_URL = "http://127.0.0.1:10809"
BOT_TOKEN = "7983657913:AAEAkVpL1ybvwPG-IuKThKdgEVo9MkeLWuE"

CLASS_TRANSLATIONS = {
    "airplane": "самолет",
    "automobile": "автомобиль",
    "bird": "птица",
    "cat": "кот/кошка",
    "deer": "олень",
    "dog": "собака",
    "frog": "лягушка",
    "horse": "лошадь",
    "ship": "корабль",
    "truck": "грузовик",
}


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Модель не найдена: {MODEL_PATH}. Сначала запустите modeling.py."
        )

    device = get_device()
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]

    model = build_resnet18(class_count=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, device


def preprocess_image(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32))

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)

    return torch.tensor(image_array, dtype=torch.float32)


def predict_image(model, class_names, device, image_bytes):
    image_tensor = preprocess_image(image_bytes).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probabilities, top_indices = torch.topk(probabilities, k=min(3, len(class_names)))

    predictions = []
    for probability, class_index in zip(top_probabilities, top_indices):
        class_name = str(class_names[int(class_index)])
        predictions.append(
            {
                "class_name": class_name,
                "translation": CLASS_TRANSLATIONS.get(class_name, class_name),
                "probability": float(probability),
            }
        )

    return predictions


def format_predictions(predictions):
    best = predictions[0]
    lines = [
        f"Предсказание: {best['translation']} ({best['class_name']})",
        f"Уверенность: {best['probability']:.1%}",
        "",
        "Топ-3:",
    ]

    for prediction in predictions:
        lines.append(
            f"- {prediction['translation']} ({prediction['class_name']}): "
            f"{prediction['probability']:.1%}"
        )

    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Пришли фотографию, а я определю, что на ней изображено. "
        "Используется ResNet18, обученная на CIFAR-10."
    )


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Обрабатываю изображение...")

    photo = update.message.photo[-1]
    photo_file = await photo.get_file()
    image_bytes = await photo_file.download_as_bytearray()

    model = context.application.bot_data["model"]
    class_names = context.application.bot_data["class_names"]
    device = context.application.bot_data["device"]

    predictions = predict_image(model, class_names, device, bytes(image_bytes))
    await update.message.reply_text(format_predictions(predictions))


async def handle_non_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Пришли именно фотографию, не текстовый файл.")


def main():
    token = BOT_TOKEN
    if not token:
        raise RuntimeError("Не найден токен Telegram-бота.")

    model, class_names, device = load_model()
    print("Модель загружена:", MODEL_PATH)
    print("Устройство:", device)
    print("Proxy:", PROXY_URL)

    application = (
        ApplicationBuilder()
        .token(token)
        .proxy(PROXY_URL)
        .get_updates_proxy(PROXY_URL)
        .build()
    )

    application.bot_data["model"] = model
    application.bot_data["class_names"] = class_names
    application.bot_data["device"] = device

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", start))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(~filters.PHOTO, handle_non_photo))

    print("Бот запущен. Остановить: Ctrl+C")
    asyncio.set_event_loop(asyncio.new_event_loop())
    application.run_polling()


if __name__ == "__main__":
    main()
