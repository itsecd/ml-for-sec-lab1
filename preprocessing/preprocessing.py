import pickle

import numpy as np


with open('preprocessing/feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('preprocessing/outliers_detector.pkl', 'rb') as f:
    outlier_detector = pickle.load(f)

informative_features = np.load('preprocessing/important_features_indices.npy')


def preprocessing(features: np.ndarray) -> np.ndarray:
    # Step 0: scale features
    features = scaler.transform(features)
    # Step 1: Only important features
    features = features[..., informative_features]
    return features