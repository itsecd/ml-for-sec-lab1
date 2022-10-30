from pickle import dumps, loads

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


class DataNormalizer:
    BINARY_FEATURES = [
        'is_default_profile',
        'is_profile_use_background_image',
        'is_verified'
    ]
    NON_EXP_FEATURES = [
        'num_digits_in_screen_name',
        'num_digits_in_name',
        'followers_friends_ratio',
        'length_of_name',
        'screen_name_length',
        'user_age',
        'description_length'
    ]
    EXP_FEATURES = [
        'listed_growth_rate', 
        'favourites_growth_rate',
        'friends_growth_rate',
        'followers_growth_rate',
        'tweets_freq',
        'listed_count',
        'favourites_count',
        'friends_count',
        'followers_count',
        'statuses_count'
    ]
    
    # Parameters' names for saving
    P_PCA = 'PCA'
    P_MEAN = 'MEAN'
    P_STD = 'STD'
    P_EXP_QUANTILE = 'QUANTILE'
    
    @staticmethod
    def load(path: str):
        with open(path, 'rb') as f:
            return loads(f.read())
    
    def __init__(self, exp_quantile = 0.95, n_components=7) -> None:
        self.pca = None
        self.mean = None
        self.std = None
        self.exp_quantile_val = None
        self.exp_quantile = exp_quantile
        self.n_components = n_components
        
    def fit(self, data: pd.DataFrame):
        exp_features, _, non_exp_features = self.extract_feature_groups(data)
        
        # --- Process exponentially distributed features
        self.exp_quantile_val = exp_features.quantile(self.exp_quantile)
        qt = self.exp_quantile_val
        exp_features = exp_features.where(exp_features < qt, qt, axis=1)
        
        nonbin_features = np.concatenate([
            non_exp_features.to_numpy(),
            exp_features.to_numpy()
        ], axis=1)
        self.mean, self.std = nonbin_features.mean(axis=0), nonbin_features.std(axis=0)
        
        # --- Train PCA
        nonbin_features = (nonbin_features - self.mean) / (self.std + 1e-3)
        self.pca = PCA(n_components=self.n_components).fit(nonbin_features)
    
    def extract_feature_groups(self, data: pd.DataFrame):
        exp_features = data[DataNormalizer.EXP_FEATURES]
        binary_features = data[DataNormalizer.BINARY_FEATURES]
        non_exp_features = data[DataNormalizer.NON_EXP_FEATURES]
        return exp_features, binary_features, non_exp_features
        
    def normalize(self, data: pd.DataFrame):
        exp_features, binary_features, non_exp_features = self.extract_feature_groups(data)
        
        assert self.exp_quantile is not None
        exp_features = exp_features.where(
            exp_features < self.exp_quantile_val, self.exp_quantile_val, axis=1)
        
        nonbin_features = np.concatenate([
            non_exp_features.to_numpy(),
            exp_features.to_numpy()
        ], axis=1)
        
        assert self.mean is not None and self.std is not None
        nonbin_features = (nonbin_features - self.mean) / (self.std + 1e-3)

        assert self.pca is not None
        nonbin_features = self.pca.transform(nonbin_features)
        return np.concatenate([
            nonbin_features,
            binary_features.to_numpy()
        ], axis=1)
    
    def save(self, path: str):
        params = {
            DataNormalizer.P_PCA: self.pca,
            DataNormalizer.P_MEAN: self.mean,
            DataNormalizer.P_STD: self.std,
            DataNormalizer.P_EXP_QUANTILE: self.exp_quantile_val
        }
        
        with open(path, 'wb') as f:
            f.write(dumps(self))
    
        
    