import os
import sys
import numpy as np
import pandas as pd
import pickle


def classify(model: str, data: str):
    if not os.path.exists(model):
        raise FileNotFoundError('Файл с моделью отсутствует.')
    
    if not os.path.exists(data):
        raise FileNotFoundError('Файл с данными отсутствует.')
        
        
    df = pd.read_csv(data)
    print('Файл с данными считан успешно. Перехожу к загрузке модели.')
    
    model = pickle.load(open(model, 'rb'))
    print('Модель загружена. Перехожу к предсказаниям.')
    
    predict = np.array(model.predict(df))
    print('Предсказания готовы.')
    
    return predict


if __name__ == '__main__':
    model = input('Введите путь до файла с моделью: ')
    data = input('Введите путь до файла с данными: ')
    
    predict = classify(model, data)
    print('Хотите сохранить результат в Excel?\n 1 - Нет\n 2 - Да')
    choice = int(input('Сохраняем?: '))
    
    if choice == 1:
        print('Вывожу predict в консоль: ')
        print('is_bot')
        for pred in predict:
            print(pred)
    else:
        pd.DataFrame(data=predict, columns=['is_bot']).to_csv('предсказания.csv', index=False)
        print(f'Предсказания сохранены в файл предсказания.csv') 
