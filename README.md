# Машинное обучение для задач информационной безопасности.
# Лабораторная работа №1.

## Выполнил - Мухин Артем, группа 6233.

 * Базовый контест :heavy_check_mark:
 * Вторичные контесты :heavy_check_mark:
 * Углубленное аналитическое исследование (частично сделано в рамках вторичных контестов)

## Модели

`1.1.pkl` - соответствует выполненному заданию пункта 1.1. Перцептрон работающий с данными размера [1, 20], i.e. 20 признаков.

`2.2.pkl` - соответствует выполненному заданию пункта 2.2. 
Перцептрон работающий с данными размера [1, 10], i.e. 10 наиболее информативных признаков.


## Зависимости
В этот раз без докерфайла （︶^︶）

```bash
pip install -r requirements.txt
```

## Запуск скрипта 

```bash
➜ python run.py --help
Usage: run.py [OPTIONS]

Options:
  -m, --model-path TEXT   Path to the model
  -d, --data-path TEXT    Path to the data to predict which
  -l, --labels-path TEXT  Path to the labels to evaluate on
  --skip-preprocessing    In case you already have preprocessed data
  -v, --verbose           Will print all the predictions
  --help                  Show this message and exit.
```

Пример: 

```bash
python run.py -m mymodel.pkl -d train.csv -l labels.csv

              precision    recall  f1-score   support

  Is not bot       1.00      0.99      1.00      1800
      Is bot       0.97      1.00      0.99       600

    accuracy                           0.99      2400
   macro avg       0.99      0.99      0.99      2400
weighted avg       0.99      0.99      0.99      2400
```