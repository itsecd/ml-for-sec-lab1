import click
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import pickle
from typing import Optional

from preprocessing import preprocessing


@click.command()
@click.option('-m', '--model-path', default='models/2.2.pkl', help='Path to the model')
@click.option('-d', '--data-path', default='features_train.csv', help='Path to the data to predict which')
@click.option('-l', '--labels-path', default=None, help='Path to the labels to evaluate on')
@click.option('--skip-preprocessing', is_flag=True, help='In case you already have preprocessed data')
@click.option('-v', '--verbose', is_flag=True, help='Will print all the predictions')
def main(model_path: str, data_path: str,
         labels_path: Optional[str], skip_preprocessing=False, verbose=False) -> np.ndarray:
    predictions = classify(model_path, data_path, skip_preprocessing)
    if verbose:
        print(list(map(lambda x: 'is not bot' if x == 0 else 'is bot', predictions)))
    if labels_path is not None:
        evaluate(labels_path, predictions)
    return predictions


def classify(model_file_name: str, data_file_name: str,
             skip_preprocessing: bool) -> np.array:
    data = pd.read_csv(data_file_name).values
    if not skip_preprocessing:
        data = preprocessing(data)
    with open(model_file_name, 'rb') as f:
        model = pickle.load(f)
    return model.predict(data)


def evaluate(labels_path: str, predictions: np.ndarray):
    labels = pd.read_csv(labels_path).values
    labels = labels.astype(np.uint8, copy=False)
    print(classification_report(labels, predictions,
          target_names=['Is not bot', 'Is bot']))


if __name__ == '__main__':
    main()
