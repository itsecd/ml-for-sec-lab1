import pandas as pd
from argparse import ArgumentParser
from pickle import loads
import numpy as np

from sklearn.svm import SVC
from normalizer import DataNormalizer


def classify(model_path: str, data_path: str) -> np.ndarray:
    with open(model_path, 'rb') as f:
        components = loads(f.read())
        
    norm: DataNormalizer = components['norm']
    model: SVC = components['model']
    
    data = pd.read_csv(data_path)
    
    data = norm.normalize(data)
    return model.predict(data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-d', '--data', required=True)
    parser.add_argument('-s', '--save', required=True)
    
    args = parser.parse_args()
    
    preds = classify(args.model, args.data)
    
    pd.DataFrame(data=preds.tolist(), columns=['is_bot']).to_csv(args.save, index=False)
    print(f'Predictions were saved to {args.save}.')
