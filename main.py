"""module for classification function."""
import os
import pickle
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from loguru import logger


def classify(model, data):
    """function to classify."""
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

    predictions = np.array(model.predict(df))  # type: ignore
    logger.debug('The predictions are ready.')

    return predictions


if __name__ == '__main__':
    """
    Example:
    python main.py -m "models/model_XGB_1_1.pkl" -d "data/features_train.csv" -s "results/predictions.csv"
    """
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-s', '--save', required=True)
    args = parser.parse_args()
    predict = classify(args.model, args.data)
    pd.DataFrame(
        data=predict, columns=['is_bot']).to_csv(args.save, index=False)
    print(f'The predictions were saved to a file: {args.save}.')
