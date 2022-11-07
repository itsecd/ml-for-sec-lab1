import click
import numpy as np
import pandas as pd

import pickle

global important_features


@click.command()
@click.option('-m', '--model-path', default='models/best_model.pkl', help='Path to the model')
@click.option('-d', '--data-path', default='features_train.csv', help='Path to the data to evaluate on')
def main(model_path: str, data_path: str) -> np.ndarray:
    res = classify(model_path, data_path)
    # print(res[:10])
    

def classify(model_file_name: str, data_file_name: str) -> np.array:
    data = pd.read_csv(data_file_name).values
    data = data[..., important_features]
    with open(model_file_name, 'rb') as f:
        model = pickle.load(f)

    return model.predict(data)
    

if __name__ == '__main__':
    important_features = np.load('important_features_indices.npy')
    main()