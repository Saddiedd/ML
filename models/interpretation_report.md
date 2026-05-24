# Интерпретация и использование модели

## Выбранная модель

Для использования выбрана модель `ResNet18`, так как она показала лучший результат на validation-наборе.

| Модель | Лучшая эпоха | Validation accuracy | Validation loss |
|---|---:|---:|---:|
| ResNet18 | 5 | 0.6753 | 0.9626 |
| MobileNetV2 | 10 | 0.5023 | 1.5286 |

## Качество выбранной модели

- Accuracy на validation-наборе: 0.6753
- Средняя уверенность на правильных ответах: 83.2%
- Средняя уверенность на ошибочных ответах: 61.1%

Если уверенность на ошибках высокая, это означает, что модель иногда ошибается достаточно уверенно. Такие случаи важны для ручной проверки.

## Лучше всего распознаются

| Класс | Precision | Recall | F1-score |
|---|---:|---:|---:|
| automobile | 0.8921 | 0.8267 | 0.8581 |
| truck | 0.8415 | 0.7967 | 0.8185 |
| ship | 0.8081 | 0.8000 | 0.8040 |

## Хуже всего распознаются

| Класс | Precision | Recall | F1-score |
|---|---:|---:|---:|
| cat | 0.4809 | 0.4200 | 0.4484 |
| bird | 0.4659 | 0.6600 | 0.5462 |
| dog | 0.5744 | 0.5533 | 0.5637 |

## Самые частые ошибки

| Истинный класс | Предсказанный класс | Количество |
|---|---|---:|
| dog | cat | 60 |
| cat | dog | 60 |
| deer | bird | 44 |
| cat | frog | 43 |
| airplane | bird | 41 |
| cat | bird | 37 |
| deer | frog | 35 |
| dog | bird | 32 |
| horse | deer | 27 |
| horse | bird | 26 |

## Практическое использование

Практическое использование модели реализовано в `telegram_bot.py`: пользователь отправляет фотографию, изображение приводится к формату 32x32 RGB, после чего модель возвращает наиболее вероятный класс и уверенность.

## Сохраненные материалы

- `S:\Posolstvo\politex\now_sem\MO\models\interpretation\class_metrics.csv`
- `S:\Posolstvo\politex\now_sem\MO\models\interpretation\top_confusions.csv`
- `S:\Posolstvo\politex\now_sem\MO\models\interpretation\correct_predictions.png`
- `S:\Posolstvo\politex\now_sem\MO\models\interpretation\wrong_predictions.png`
