import os
import sys
import pandas as pd
import joblib
from joblib import dump
import sklearn

def classify(modelFileName,dataFileName):
        if not os.path.exists(modelFileName):
            raise FileNotFoundError('Модель отсутствует')

        if not os.path.exists(dataFileName):
            raise FileNotFoundError('Данные отсутствуют')

        features=pd.read_csv(dataFileName)
        model = joblib.load(modelFileName)
        print('Модель и данные считаны успешно')

        y_pred = model.predict(features)

        return y_pred
if __name__ == '__main__':

        model = input('Введите имя модели ')
        data = input('Введите имя файла с признаками ')

        result = classify(model, data)
        r=pd.DataFrame(result, columns=['is_bot']).to_csv('result.csv', index=False)
        print('Pезультаты сохранены в файле result.csv')