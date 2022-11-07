# Машинное обучение для задач информационной безопасности.
# Лабораторная работа №1.

## Выполнил - Мухин Артем, группа 6233.

 * Базовый контест :heavy_check_mark:
 * Вторичные контесты :heavy_check_mark:
 * Углубленное аналитическое исследование (частично сделано в рамках вторичных контестов)

## Зависимости
В этот раз без докерфайла （︶^︶）

```bash
pip install -r requirements.txt
```

## Запуск скрипта 

```bash
Usage: run.py [OPTIONS]

Options:
  -m, --model-path TEXT  Path to the model
  -d, --data-path TEXT   Path to the data to evaluate on
  --help                 Show this message and exit.
```

Пример: 

```bash
python run.py -m mymodel.pkl -d train.csv
```