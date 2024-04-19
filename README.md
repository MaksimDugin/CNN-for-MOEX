# Прогнозирование финансовых рынков с использованием свёрточных нейронных сетей

Данный репозиторий содержит код, разработанный для курсовой работы по прогнозированию финансовых рынков с использованием свёрточных нейронных сетей.

## Описание

В данной работе исследуется возможность применения свёрточных нейронных сетей (CNN) для прогнозирования динамики цен на финансовых рынках. Мы исследуем архитектуру CNN и её эффективность в предсказании направления движения цен на акции. 

## Структура репозитория

1. **data/**: Папка для хранения данных, используемых в процессе обучения и тестирования модели.
2. **src/**: Исходный код, включая скрипты для обработки данных, обучения модели и тестирования.

## Требования к окружению

Для успешного запуска кода требуется установить следующие зависимости:

Для обработки данных
- Python 3.12.2
- библиотеки: NumPy 1.26.4, pandas 2.2.1, requests 2.31.0, plotly 5.20.0

Для нейронной сети
- Python 3.6.13
- библиотеки: NumPy, tensorflow 1.2.0

## Использование

1. Запустите программу `Parser_price.ipynb`, либо сами скачайте данные из `data\`.
2. Обучите модель: выполните скрипт `CNN.py`, указав путь к данным и гиперпараметры.
3. Протестируйте модель: выполните скрипт `CNN.py`, заменив необходимое.

## Ссылки

1. Статья: "Нейронные сети в мире финансов: Прогнозирование движения цен на российском рынке акций" Автор: Дугин М.Д. 2024

## Авторы

- Дугин Максим Денисович ([@MaksimDugin](https://github.com/MaksimDugin))
tg: @makdugin
email: maxdugin03@rambler.ru
