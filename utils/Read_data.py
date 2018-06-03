import sys
sys.path.append('/mnt/4_TB_HD/ramona/utils')
sys.path.append('/mnt/4_TB_HD/ramona/utils/features')
import pandas as pd
from sklearn.preprocessing import LabelEncoder 
from keras.utils import to_categorical
from pathlib import Path
import numpy as np



def read_data(date, stack = False, stage2=True, indicator_matrix = True):
    '''
    the data has to be passed through in the form: '_18-04-24-21-46'
    
    '''
    Y_test_df = pd.read_pickle('/mnt/4_TB_HD/ramona/utils/features/Ytest_'+date)
    Y_train_df = pd.read_pickle('/mnt/4_TB_HD/ramona/utils/features/Ytrain_'+date)
    X_test_df = pd.read_pickle('/mnt/4_TB_HD/ramona/utils/features/Xtest_'+date)
    X_train_df = pd.read_pickle("/mnt/4_TB_HD/ramona/utils/features/Xtrain_"+date)
    if stage2 is True:
        stage2path = Path("/mnt/4_TB_HD/ramona/utils/features/stage2_"+date)
        if stage2path.is_file():
            stage2_transf = pd.read_pickle(stage2path)
            stage2filtered = Path("/mnt/4_TB_HD/ramona/utils/features/stage2filtered_"+date)
            if stage2filtered.is_file():
                stage2_transf_filtered= pd.read_pickle(stage2filtered)
            else:
                stage2_transf_filtered =  'no stage2 calculated'
        else:
            stage2_transf ='no stage2 calculated'
            stage2_transf_filtered =  'no stage2 calculated'
    
    
    
    
    X_train = X_train_df
    Y_train = Y_train_df.astype(int)
    X_test = X_test_df
    Y_test = Y_test_df.astype(int)
    if indicator_matrix is True: 
        encoder = LabelEncoder()
        encoder.fit(Y_train.astype(str))
        encoded_y = encoder.transform(Y_train)
        Y_train_cateogorial = to_categorical(encoded_y)
        encoded_y_test = encoder.transform(Y_test)
        Y_test_cateogorial = to_categorical(encoded_y_test)
        
    if stage2 is True:
        if stage2_transf_filtered.ndim == 1:
             stage2_transf_filtered= np.vstack(( stage2_transf_filtered))

        stack_X = np.vstack((X_train, X_test))
        stack_Y = np.hstack((Y_train,Y_test))
        stack_Y_categorial = np.vstack((Y_train_cateogorial, Y_test_cateogorial))

        if stack is True: 
            return stack_X,stack_Y  , stack_Y_categorial,  X_train, Y_train, X_test, Y_test, Y_train_cateogorial, Y_test_cateogorial, stage2_transf, stage2_transf_filtered
    
        return X_train, Y_train, X_test, Y_test, Y_train_cateogorial, Y_test_cateogorial, stage2_transf, stage2_transf_filtered
    
    if indicator_matrix is True: 
        return X_train, Y_train, X_test, Y_test, Y_train_cateogorial, Y_test_cateogorial
    else: 
         return X_train, Y_train, X_test, Y_test