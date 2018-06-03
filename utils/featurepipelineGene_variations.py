import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import normalize
from Read_data import read_data
import sys
from scipy.sparse.linalg import svds
sys.path.append('../')
import ourPreprocessor
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS as esw
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import vstack
import pickle
import csv
from pathlib import Path
import math
from sklearn.externals import joblib





class featurepipeline:

    def __init__(self,  shuffling = False, gene_variation=True):
        
        self.shuffled=shuffling
        self.gene_variation = gene_variation
       
            
    def gene_variation_transform(self, n_comp = 25,n_iter=20 , columns = None):
        '''
        Based on https://www.kaggle.com/danofer/genetic-variants-to-protein-features
        '''
        df_joint = pd.concat([self.train_gene_variations_df,self.test_gene_variations_df,self.stage2test_gene ])
        df_joint["simple_variation_pattern"] = df_joint.Variation.str.contains(r'^[A-Z]\d{1,7}[A-Z]$',case=False)
        df_joint['location_number'] = df_joint.Variation.str.extract('(\d+)', expand=True)
        AA_VALID = 'ACDEFGHIKLMNPQRSTVWY'
        df_joint['variant_letter_first'] = df_joint.apply(lambda row: row.Variation[0] if row.Variation[0] in (AA_VALID) else np.NaN,axis=1)
        df_joint['variant_letter_last'] = df_joint.apply(lambda row: row.Variation.split()[0][-1] if (row.Variation.split()[0][-1] in (AA_VALID)) else np.NaN,axis=1)
        df_joint.loc[df_joint.simple_variation_pattern==False,['variant_letter_last',"variant_letter_first"]] = np.NaN
        ofer8=self.TransDict_from_list(["C", "G", "P", "FYW", "AVILM", "RKH", "DE", "STNQ"])
        sdm12 =self.TransDict_from_list(["A", "D", "KER", "N",  "TSQ", "YF", "LIVM", "C", "W", "H", "G", "P"] )
        self.pc5 = {"I": "A", # Aliphatic
             "V": "A",         "L": "A",
             "F": "R", # Aromatic
             "Y": "R",         "W": "R",         "H": "R",
             "K": "C", # Charged
             "R": "C",         "D": "C",         "E": "C",
             "G": "T", # Tiny
             "A": "T",         "C": "T",         "S": "T",
             "T": "D", # Diverse
             "M": "D",         "Q": "D",         "N": "D",
             "P": "D"}
        df_joint['AAGroup_ofer8_letter_first'] = df_joint["variant_letter_first"].map(ofer8)
        df_joint['AAGroup_ofer8_letter_last'] = df_joint["variant_letter_last"].map(ofer8)
        df_joint['AAGroup_ofer8_equiv'] = df_joint['AAGroup_ofer8_letter_first'] == df_joint['AAGroup_ofer8_letter_last']
        df_joint['AAGroup_m12_equiv'] = df_joint['variant_letter_last'].map(sdm12) == df_joint['variant_letter_first'].map(sdm12)
        df_joint['AAGroup_p5_equiv'] = df_joint['variant_letter_last'].map(self.pc5) == df_joint['variant_letter_first'].map(self.pc5)
        df_joint = df_joint.astype(str)
        
        
        if columns is None or 'PC_distance' in columns: 
            ph_distances = pd.read_csv("../utils/physiochem.csv", sep=';')
            self.ph_distances = ph_distances.set_index('Unnamed: 0')
            PC_distance= df_joint[['variant_letter_first', 'variant_letter_last']].apply(lambda x: self.finddistance(x[0],x[1]), axis=1)
            meanv=np.mean(PC_distance)
            PC_distance[np.isnan(PC_distance)]=meanv
            PC_distance = normalize(PC_distance.values.reshape(-1, 1), axis=0)
            PC_distance_train = PC_distance[:self.shape_train[0]]
            PC_distance_test = PC_distance[self.shape_train[0]: self.shape_train[0]+self.shape_test[0]]   
            PC_distance_stage2test = PC_distance[self.shape_train[0]+self.shape_test[0]:self.shape_train[0]+self.shape_test[0]+self.shape_stage2test[0]]
       
        
        if columns is None: 
            self.df_joint_columns = df_joint.columns
            self.df_joint_columns = np.append(self.df_joint_columns, ['PC_distance'])
        else: 
            self.df_joint_columns = columns[:]
            if 'PC_distance' in columns:
                columns.remove('PC_distance')
            if columns == []:

                df_joint = []
            else: 
                df_joint = df_joint[columns]
            
      
        if len(df_joint)>0:    
            onehotlabels = self.Label_encoder(df_joint)
            train_gene = onehotlabels[:self.shape_train[0]]
            test_gene = onehotlabels[self.shape_train[0]: self.shape_train[0]+self.shape_test[0]]
            stage2test_gene = onehotlabels[self.shape_train[0]+self.shape_test[0]:self.shape_train[0]+self.shape_test[0]+self.shape_stage2test[0]]
        

        
       
            
            pipeline = Pipeline([
                        ('svd', TruncatedSVD(n_components = n_comp, n_iter = n_iter)),
                        ('norm', Normalizer())
                              ])
    

            feature_train = pipeline.fit_transform(train_gene)
            feature_test = pipeline.transform(test_gene)
            feature_stage2test = pipeline.transform(stage2test_gene)
            if columns is None or 'PC_distance' in self.df_joint_columns:
                feature_train = np.hstack((feature_train ,PC_distance_train ))
                feature_test = np.hstack((feature_test ,PC_distance_test ))
                feature_stage2test = np.hstack((feature_stage2test ,PC_distance_stage2test))
            
        elif columns is None or 'PC_distance' in self.df_joint_columns: 
          
            feature_train = PC_distance_train
            feature_test = PC_distance_test
            feature_stage2test = PC_distance_stage2test
       

        return feature_train, feature_test, feature_stage2test

        
    def Label_encoder(self, df_joint):
    
        le = LabelEncoder()
        df_joint_encoded = df_joint.apply(le.fit_transform)
        enc = OneHotEncoder()
        enc.fit(df_joint_encoded)


        onehotlabels = enc.transform(df_joint_encoded).toarray()

        return onehotlabels 
        
            
    def finddistance(self,AA1 = None, AA2=None):
        if AA1 in self.pc5 and AA2 in self.pc5 and AA1 != 'W':
            AAlist = self.ph_distances.loc[AA1] #Finds row for AA1
        else:
            return float('nan')
        if AA2 == 'S' or AA1=='W':
            dist=float('nan')
        else:
            dist=AAlist.get(AA2) #Search for AA2
        if math.isnan(dist): #If not found, switch order and search again
            dist = self.finddistance(AA1=AA2, AA2=AA1)
        return dist
    
    def TransDict_from_list(self, groups):
        '''
        Given a list of letter groups, returns a dict mapping each group to a
        single letter from the group - for use in translation.
        >>> alex6=["C", "G", "P", "FYW", "AVILM", "STNQRHKDE"]
        >>> trans_a6 = TransDict_from_list(alex6)
        >>> print(trans_a6)
        {'V': 'A', 'W': 'F', 'T': 'D', 'R': 'D', 'S': 'D', 'P': 'P',
         'Q': 'D', 'Y': 'F', 'F': 'F',
         'G': 'G', 'D': 'D', 'E': 'D', 'C': 'C', 'A': 'A',
          'N': 'D', 'L': 'A', 'M': 'A', 'K': 'D', 'H': 'D', 'I': 'A'}
        '''
        transDict = dict()

        result = {}
        for group in groups:
            g_members = sorted(group) #Alphabetically sorted list
            for c in g_members:
                result[c] = str(g_members[0]) #K:V map, use group's first letter as represent.
        return result

    
    def processGeneVariation(self,pipeline_date='', balance = None,   gene_n_comp = 25,gene_n_iter = 50, gene_columns= None, train = None, test = None, stage2_transf = None, pipeline = None, svd_name= None ):
        if pipeline_date!='': 
            train, test, stage2_transf, pipeline= self.load_data_gene_processing(pipeline_date)
        
        feature_train, feature_test, feature_stage2test = self.gene_variation_transform(gene_n_comp,gene_n_iter, gene_columns )
        train = np.hstack((train, feature_train))
        test = np.hstack((test, feature_test))
        stage2test = np.hstack(( stage2_transf, feature_stage2test))
        if svd_name is None: 
            svd_name = str('as in pipeline'+pipeline_date)
        self.merge_and_save(train, test, stage2test, pipeline, svd_name = svd_name, balance = balance, pipeline_date=pipeline_date)
                
    
    
    def processText(self,chars='1000',preprocessor = ourPreprocessor.myPreprocessor , tokenizer = ourPreprocessor.tokenizeronlyletters, ngram = 1, max_features = 50000, max_df = 0.9, min_df = 0, n_iter = 50, n_comp = 300, normalizer = Normalizer(), balance = None,  gene_n_comp = 25,gene_n_iter=20 , gene_columns = None, svd = None ):
    
  
        self.chars = chars
        self.load_data_for_textprocessing_and_gene_processing()
            
        if svd == 'scipy_svd':
            train, test, stage2_transf = self.process_via_scipy_svd(preprocessor, tokenizer,max_features , max_df,  n_iter,n_comp)
            svd_name = svd
        
        else:
            pipeline = Pipeline([('tfidf', TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer,            
                                             max_features= max_features , max_df= max_df, 
                                            min_df=min_df, ngram_range=(1,ngram), stop_words=ourPreprocessor.stopWords)), 
                            ('svd', TruncatedSVD(n_components = n_comp, n_iter = n_iter)),
                            ('norm', normalizer)])
            
            train = pipeline.fit_transform(self.X_train)
            test = pipeline.transform(self.X_test)
            stage2_transf = pipeline.transform(self.stage2)
            svd_name = 'TruncatedSVD'
        if self.gene_variation is True: 
            self.processGeneVariation(balance = balance,   gene_n_comp =gene_n_comp ,gene_n_iter = gene_n_iter, gene_columns= gene_columns, train = train, test= test, stage2_transf=stage2_transf, pipeline = pipeline , svd_name = svd_name )
             
        else: 
            self.merge_and_save(train, test, stage2_transf, pipeline, svd_name, balance)
        
    
    
    def merge_and_save(self,train, test, stage2test, pipeline, svd_name = '', balance = None, pipeline_date=''):
        time = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        self.save_pipeline(model=pipeline, time=time )
        path = '/mnt/4_TB_HD/ramona/utils/features/'
        
        if balance is not None:
            train, y_resampled = balance.fit_sample(train, self.Y_train.astype(int))
        else: 
            y_resampled = self.Y_train
            
            
            
        stage2filtered = self.stage2_solution['ID'].apply(lambda x:stage2test[x-1] ).values

        
        filename1 = 'Xtrain_' + time
        filename2 = 'Xtest_' + time
        filename3 = 'Ytrain_' + time
        filename4 = 'Ytest_' + time
        filename5 = 'stage2_' + time
        filename6 = 'stage2filtered_' + time
        with open(path + filename1, 'wb') as f1:
            pickle.dump(train, f1)
        with open(path + filename2, 'wb') as f2:
            pickle.dump(test, f2)
        with open(path + filename3, 'wb') as f3:
            pickle.dump(y_resampled, f3)
        with open(path + filename4, 'wb') as f4:
            pickle.dump(self.Y_test, f4)
        with open(path + filename5, 'wb') as f5:
            pickle.dump(stage2test, f5)
        with open(path + filename6, 'wb') as f5:
            pickle.dump(stage2filtered, f5)

        if self.gene_variation is False:
            columns=''
        else:
            columns = self.df_joint_columns
        self.save_pipelineinfo(pipeline.named_steps, filename1, filename2, filename3, filename4, balance, columns, svd_name,pipeline_date )
        


    def save_pipeline(self, model=None, time=None):
        
        f = '/mnt/4_TB_HD/ramona/utils/textPipelines/' + str(time)
        #joblib.dump(f, model+time+'.pkl', compress = 1)
        with open(f, 'wb') as pl:
            pickle.dump(model, pl)
    

    def save_pipelineinfo(self, model = '' , trainpickle = '', testpickle= '', Y_train='', Y_test='', balance = '', genecolumns='',svd_name = '', pipeline_date =''):
    
        '''
        Appends the input parameter to a csv data file which is specified through the filename (default: deliver/results.csv)
        '''
        filename = '/mnt/4_TB_HD/ramona/deliver/pipelineinfo2.csv'
        if balance is not None: 
            balancing_info = balance.get_params
        else:
            balancing_info = 'No balancing'
        my_file = Path(filename)
        if my_file.is_file():   
            with open(filename,'a') as f:
                f = csv.writer(f)
                f.writerow([ model, trainpickle, testpickle, Y_train, Y_test, self.chars,balancing_info, self.shuffled, genecolumns, svd_name, pipeline_date ])   
        else: 
            with open(filename,'w') as f:
                f = csv.writer(f)
                f.writerow([ 'Pipeline info', 'Train pickle filename', 'Test pickle filename', 'Y train', 'Y test', 'Text', 'Balancing method'])
                f.writerow([ model, trainpickle, testpickle, Y_train, Y_test, self.chars, balancing_info])
                
                
    def process_via_scipy_svd(self, preprocessor, tokenizer,max_features , max_df, maxiter,k ):
        tfidf = TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer,            
                                                      max_features= max_features , max_df= max_df)
           
        train = tfidf.fit_transform(self.X_train)
        test = tfidf.transform(self.X_test)
        stage2_transf = tfidf.transform(self.stage2)
        (U, s, Va) = svds(train, k= k , maxiter= maxiter)
        train = np.dot(U,np.diag(s))
        test = np.dot(test,Va.T)
        stage2_transf = np.dot(stage2_transf , Va.T )
     
        return train, test, stage2,transf
    
    
    def load_data_for_textprocessing_and_gene_processing(self ):
        Y_test = pd.read_pickle('../develop/Y_test_'+self.chars).values
        Y_train = pd.read_pickle("../develop/Y_train_"+self.chars).values
        X_test = pd.read_pickle("../develop/X_test_"+self.chars).values
        X_train = pd.read_pickle("../develop/X_train_"+self.chars).values
        self.stage2 = pd.read_pickle("/mnt/4_TB_HD/ramona/utils/stage2_data/stage2test_" + self.chars +".sav").values
        self.stage2_solution = pd.read_csv("/mnt/4_TB_HD/ramona/utils/stage2_data/stage_2_private_solution.csv")
       
               
        
        if self.gene_variation is True:
            train_gene = pd.read_pickle("../utils/data/train_variants_filtered")
            test_gene = pd.read_pickle("../utils/data/test_variants_filtered")
            self.stage2test_gene = pd.read_csv("../utils/stage2_data/stage2_test_variants.csv")[['Gene', 'Variation']]
            trainset =np.stack((X_train,Y_train,train_gene['Gene'].values, train_gene['Variation'].values ), axis =1)
            testset =np.stack((X_test,Y_test,test_gene['Gene'].values, test_gene['Variation'].values ), axis =1)
            
        
        # stack X and Y
        else: 
            trainset =np.stack((X_train,Y_train), axis =1)
            testset =np.stack((X_test,Y_test), axis =1)
        train_df=pd.DataFrame(trainset)
        test_df=pd.DataFrame(testset)
        
        
        
        # shuffle to avoid modeling trends
        if self.shuffled is True: 
            train_df = shuffle(train_df)
        

        #split into X and Y again
        self.Y_test = test_df.loc[:,1].values
        self.Y_train = train_df.loc[:,1].values
        self.X_test = test_df.loc[:,0].values
        self.X_train = train_df.loc[:,0].values
    
        if self.gene_variation is True:
            self.test_gene_variations_df = test_df.loc[:,2:]
            self.test_gene_variations_df = self.test_gene_variations_df.rename(index=str, columns={2: 'Gene', 3: 'Variation'})
            self.train_gene_variations_df = train_df.loc[:,2:]
            self.train_gene_variations_df = self.train_gene_variations_df.rename(index=str, columns={2: 'Gene', 3: 'Variation'})
        
        self.shape_train = self.Y_train.shape
        self.shape_test = self.Y_test.shape
        self.shape_stage2test = self.stage2.shape        
        
    
    def load_data_gene_processing(self,date):
            
        if self.shuffled is True:
            raise AttributeError('Shuffeling is not possible when using an already prepoccessed X_train')
        pipeline = pickle.load( open( '/mnt/4_TB_HD/ramona/utils/textPipelines/' + date, "rb" ) )
        n_comp_svd = pipeline.named_steps['svd'].n_components
        train, self.Y_train, test, self.Y_test, Y_train_cateogorial, Y_test_cateogorial, stage2_transf, stage2_transf_filtered = read_data(date)
        train = train[:,:n_comp_svd]
        test = test[:,:n_comp_svd]
        
        X_train = 'Xtrain_'+date
        pipelineinfo = pd.read_csv("/mnt/4_TB_HD/ramona/deliver/pipelineinfo2.csv")
        self.chars = pipelineinfo[pipelineinfo['Train pickle filename'] == X_train]['Text'].values[0]
        if type(stage2_transf) == str:
            stage2_transf = pipeline.transform(pd.read_pickle("/mnt/4_TB_HD/ramona/utils/stage2_data/stage2test_" + self.chars +".sav").values)
            
            
        self.stage2_solution = pd.read_csv("/mnt/4_TB_HD/ramona/utils/stage2_data/stage_2_private_solution.csv")
        stage2_transf = stage2_transf[:,:n_comp_svd]
        
    
        self.test_gene_variations_df = pd.read_pickle("../utils/data/train_variants_filtered")[['Gene', 'Variation']]
        self.train_gene_variations_df = pd.read_pickle("../utils/data/test_variants_filtered")[['Gene', 'Variation']]
        self.stage2test_gene = pd.read_csv("../utils/stage2_data/stage2_test_variants.csv")[['Gene', 'Variation']]
        
        self.shape_train = self.Y_train.shape
        self.shape_test = self.Y_test.shape
        self.shape_stage2test = stage2_transf.shape  
            
        return train, test, stage2_transf, pipeline