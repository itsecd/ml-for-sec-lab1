import pickle

import numpy as np
import pandas as pd


def classify(model_file_name: str, data_file_name: str) -> np.array:
    """
    :param model_file_name: имя файла, из которого будут загружены параметры обученной ML-модели
    (например, ‘model.txt’).
    :param data_file_name: имя CSV-файла с данными для анализа, которые нужно классифицировать.
    Файл имеет ту же структуру, что и файл обучающей выборки. Каждая строка файла (кроме
    заголовочной) содержит признаки одного пользователя.
    :return: numpy-вектор с результатами классификации размерности (K, ), где K - количество содержательных строк в
    dataFileName.
    """
    with open(model_file_name, 'rb') as f:
        estimator_scaler_dict = pickle.load(f)

    estimator = estimator_scaler_dict['model']
    scaler = estimator_scaler_dict['scaler']

    df = pd.read_csv(data_file_name)

    df = scaler.transform(df)

    return estimator.predict(df)
