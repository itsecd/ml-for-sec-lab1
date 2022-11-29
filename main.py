import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from loguru import logger
from joblib import load


def classify(model_file_name: str, data_file_name: str) -> np.array:
    """Функция классификации."""
    if not os.path.exists(model_file_name) or not os.path.exists(data_file_name):
        raise FileNotFoundError("Файла с моделью или данными не существует.")

    logger.debug(f"Попытка считывания данных из файла {data_file_path}...")
    data = pd.read_csv(data_file_name, on_bad_lines="skip")
    logger.debug("Данные считаны успешно.")

    logger.debug(f"Загружаем модель из файла {model_file_name}...")
    model = load(model_file_name)
    logger.debug("Модель успешно загружена.")

    logger.debug(f"Делаем предсказания...")
    prediction_array = np.array(model.predict(data))
    logger.debug("Предсказания готовы.")
    return prediction_array


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 3:
        raise Exception("Не заданы все необходимые аргументы."
                        "\nФормат команды: python main.py -data <data_file_path> <-model <model_file_path>>")
    data_file_path = args[2]
    if os.path.splitext(data_file_path)[1] != ".csv":
        raise Exception("Файл с данными должен быть формата csv.")
    model_file_path = "./model.joblib"
    if len(args) == 5:
        model_file_path = args[-1]
    predictions = classify(model_file_path, data_file_path)

    print("Выберите одно из следующих действий:\n1 - Вывести на экран результаты;\n2 - Сохранить в файл")
    choice = int(input("Ваш выбор:"))

    if choice == 1:
        print("is_bot")
        for prediction in predictions:
            print(prediction)
    else:
        if not os.path.exists("results"):
            os.makedirs("results")
        results_file_path = os.path.join("results", f"predictions_{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}.csv")
        pd.DataFrame(data=predictions, columns=["is_bot"]).to_csv(results_file_path, index=False)
        print(f"Данные успешно сохранены в файле: {results_file_path}")