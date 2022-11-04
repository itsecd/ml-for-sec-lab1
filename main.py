import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger


def classify(model: str, data: str):
    """function to run predictions"""
    if not os.path.exists(model):
        raise FileNotFoundError('No file with the model.')

    if not os.path.exists(data):
        raise FileNotFoundError('No data file is available.')

    logger.debug(f'Reading data from a file: {data}.')
    df = pd.read_csv(data)
    logger.debug(
        'The data file was read successfully. Moving on to loading the model.')

    model = pickle.load(open(model, 'rb'))
    logger.debug(f'The model: {model} is loaded. Moving on to predictions.')

    predict = np.array(model.predict(df))
    logger.debug('The predictions are ready.')

    return predict


if __name__ == '__main__':
    model = input('Enter the path to the file with the model: ')
    data = input('Enter the path to the data file: ')

    predict = classify(model, data)
    print('Do you want to save the result to Excel?\n 1 - No\n 2 - Yes')
    choice = int(input('Saving?: '))

    if choice == 1:
        print('Output predict in the console: ')
        for pred in predict:
            print(f'is_bot: {pred}')
    else:
        now = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
        pd.DataFrame(data=predict, columns=['is_bot']).to_csv(
            f'results/predictions_{now}.csv', index=False)
        print(f'the predictions were saved to the folder "results".')
