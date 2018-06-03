import pandas as pd
import pickle
import numpy as np
import sys
sys.path.append('../utils')
import ourPreprocessor
from ourPreprocessor import stopWords


def features(model):
    model = pickle.load(open(model, 'rb'))
    tfidf = model.named_steps['tfidf']
    svd = model.named_steps['svd']
    feature_names = tfidf.get_feature_names()
    number_of_components = svd.n_components
    component_features = pd.DataFrame(columns=['1', '2', '3', '4', '5','6','7','8','9','10'])
    for row in range(number_of_components):
        best_features =  [feature_names[i] for i in svd.components_[row].argsort()[::-1]]
        score = [svd.components_[row][i] for i in svd.components_[row].argsort()[::-1]]
        for i in range(10):
            component_features.loc[row, str(i+1)] = (best_features[i],round(score[i], 4)  )
    return component_features