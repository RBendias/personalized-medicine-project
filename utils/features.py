
# coding: utf-8

# In[ ]:


class features:

    def __init__(self, chars='1000'):
        import pandas as pd
        import numpy as np
        import sys
        sys.path.append('../')
        import ourPreprocessor
        from sklearn.utils import shuffle
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as esw
        
        # read text sections
        Y_test = pd.read_pickle('../develop/Y_test_'+chars)
        Y_train = pd.read_pickle("../develop/Y_train_"+chars)
        X_test = pd.read_pickle("../develop/X_test_"+chars)
        X_train = pd.read_pickle("../develop/X_train_"+chars)
        training_variants = pd.read_csv("../utils/data/training_variants")
        training_variants = training_variants[0:3316]
        test_variants = pd.read_csv("../utils/data/test_variants")
        self.training_merge = training_variants.join(X_train.to_frame())
        self.test_merge = test_variants.join(X_test.to_frame())
        
        Y_test = Y_test.values
        Y_train = Y_train.values
        X_test = X_test.values
        X_train = X_train.values
        
        self.chars = chars
        
        # stack X and Y
        trainset =np.stack((self.training_merge,Y_train), axis =1)
        testset =np.stack((self.test_merge,Y_test), axis =1)
        train_df=pd.DataFrame(trainset)
        test_df=pd.DataFrame(testset)
        
        # shuffle to avoid modeling trends
        train_df = shuffle(train_df)
        test_df = shuffle(test_df)
        
        #split into X and Y again
        self.Y_test = test_df.loc[:,1].values
        self.Y_train = train_df.loc[:,1].values
        X_test = test_df.loc[:,0].values
        X_train = train_df.loc[:,0].values
        self.X_test = test_df.loc[:,0].values
        self.X_train = train_df.loc[:,0].values
        self.var_train = X_train['Variation']
        self.var_test = X_test['Variation']

        
    def processText(self, preprocessor, tokenizer, ngram, max_features, max_df, min_df, n_iter, n_comp, normalizer, balance = None):
        import datetime
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import Normalizer
        from sklearn.pipeline import Pipeline
        import pickle
        import sys
        sys.path.append('../utils/features')
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as esw
        
        stopWords = ["fig", "figure", "et", "al", "table",  
        "data", "analysis", "analyze", "study",  
        "method", "result", "conclusion", "author",  
        "find", "found", "show", "perform",  
        "demonstrate", "evaluate", "discuss", "google", "scholar",   
        "pubmed",  "web", "science", "crossref", "supplementary", '(fig.)', '(figure', 'fig.', 'al.', 'did',
        'thus,', '...', 'interestingly,', 'and/or', 'author'] + list(esw)
        
        pipeline = Pipeline([('tfidf', TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer,            
                                                      max_features= max_features , max_df= max_df, 
                                                      min_df=min_df, ngram_range=(1,ngram), stop_words=stopWords)), 
                        ('svd', TruncatedSVD(n_components = n_comp, n_iter = n_iter)),
                        ('norm', normalizer)])
        train, y_resampled, test = self.prep(pipeline, self.X_train, self.Y_train, self.X_test, self.Y_test, balance)
        
        path = '/mnt/4_TB_HD/ramona/utils/features/'
        time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        filename1 = 'Xtrain_' + time
        filename2 = 'Xtest_' + time
        filename3 = 'Ytrain_' + time
        filename4 = 'Ytest_' + time
        with open(path + filename1, 'wb') as f1:
            pickle.dump(train, f1)
        with open(path + filename2, 'wb') as f2:
            pickle.dump(test, f2)
        with open(path + filename3, 'wb') as f3:
            pickle.dump(y_resampled, f3)
        with open(path + filename4, 'wb') as f4:
            pickle.dump(self.Y_test, f4)
            
        #pickle.dump(train, open( filename1, "wb" ))
        #pickle.dump(test, open( filename2, "wb" ))
        #pickle.dump(self.Y_train, open( filename3, "wb" ))
        #pickle.dump(self.Y_test, open( filename4, "wb" ))
        self.save_pipelineinfo(pipeline.named_steps, filename1, filename2, filename3, filename4, balance)
        self.save_pipeline(model=pipeline, time=time)

# In[ ]:


    def prep(self, pipeline, X_train, Y_train, X_test, Y_test, balance):
        x_features_tr = pipeline.fit_transform(X_train, Y_train)
        x_features_test = pipeline.transform(X_test)
        if balance is not None:
            X_resampled, Y_resampled = balance.fit_sample(x_features_tr, Y_train.astype(int))
            return X_resampled, Y_resampled, x_features_test
        else:
            return x_features_tr, Y_train, x_features_test


# In[ ]:
    def save_pipeline(self, model=None, time=None):
        import pickle
        import sys
        sys.path.append('../utils/features')
        f = '/mnt/4_TB_HD/ramona/utils/textPipelines/' + str(time)
        with open(f, 'wb') as pl:
            pickle.dump(model, pl)

    def save_pipelineinfo(self, model = None , trainpickle = None, testpickle= None, Y_train=None, Y_test=None, balance = None):
        import csv
        from pathlib import Path
        '''
        Appends the input parameter to a csv data file which is specified through the filename (default: deliver/results.csv)
        '''
        filename = '/mnt/4_TB_HD/ramona/deliver/pipelineinfo2.csv'
        my_file = Path(filename)
        if my_file.is_file():   
            with open(filename,'a') as f:
                f = csv.writer(f)
                f.writerow([ model, trainpickle, testpickle, Y_train, Y_test, self.chars, balance.get_params])  
        else: 
            with open(filename,'w') as f:
                f = csv.writer(f)
                f.writerow([ 'Pipeline info', 'Train pickle filename', 'Test pickle filename', 'Y train', 'Y test', 'Text', 'Balancing method'])
                f.writerow([ model, trainpickle, testpickle, Y_train, Y_test, self.chars, balance.get_params])

