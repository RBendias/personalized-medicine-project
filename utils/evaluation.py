from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import csv
from pathlib import Path
from sklearn.preprocessing import LabelBinarizer
import datetime

class evaluation_class():
    '''
    This class should provide all important evaluation tools and save if required the evaulation data in a csv file
    
    param number_of_characters: number of characters which were taken for creating the data set
    param y_pred: by default None, then it will be calculated with the model, evaluation of an already existing y_pred is also possible
    param Y_test: the data file which is used with the correct results in it
    param model: This could be either a pipeline or training model, such that the y_pred can be calculated if it isn't given
    
    
    '''
    def __init__(self ,number_of_characters = None,Y_test = None, y_pred = None, X_test = None , model = None, pipelinedate = None, modeltime = None, weighting = None, balance = None):
        
       
        if isinstance(X_test, str):
            self.X_test_file = Y_test
            self.X_test = pd.read_pickle(X_test).astype(int)
            if not isinstance(self.X_test, (np.ndarray, np.generic)):
                self.X_test = self.X_test.values 
        else: 
            self.X_test = X_test

        if isinstance(Y_test, str):
            self.Y_test_file = Y_test
            self.Y_test = pd.read_pickle(Y_test).astype(int)
            if not isinstance(self.Y_test, (np.ndarray, np.generic)):
                self.Y_test = self.Y_test.values
        else: 
            self.Y_test = Y_test
            if pipelinedate is not None:
                self.Y_test_file = pipelinedate
            else: 
                self.Y_test_file = input('X_test or Y_test file information')
        self.model = model
        if number_of_characters is not None: 
            self.number_of_characters = number_of_characters
        self.lb = LabelBinarizer()
        self.lb = self.lb.fit([1,2,3,4,5,6,7,8,9])
            
        if not (model is  None and y_pred is None):
            if y_pred is not None: 
                self.Y_pred = y_pred
            else:
                try: 
                    self.Y_pred = self.model.predict_proba(self.X_test) 
                    print(self.Y_pred)
                except AttributeError:
                    self.Y_pred = self.model.predict(self.X_test)
                print(self.Y_pred.shape)
        
            if self.Y_pred.ndim == 1: # No probabilities
                self.Y_pred = self.Y_pred.astype(int)
                self.Y_pred_indicator_matrix = self.lb.transform(self.Y_pred) 
            else:  
                self.Y_pred_indicator_matrix = self.Y_pred
                self.Y_pred = self.lb.inverse_transform(self.Y_pred,  threshold = 0.5)
            
            if self.Y_test.ndim == 1:
                self.Y_test_indicator_matrix = self.lb.transform(self.Y_test.astype(int))
                self.Y_test = self.Y_test.astype(int)
            else:
                self.Y_test_indicator_matrix = self.Y_test
                self.Y_test = self.lb.inverse_transform(self.Y_test, threshold = 0.5)
        
        if self.model is None:
             self.model = 'modeltime = '+str(modeltime)+','+'weighting = '+str(weighting)+','+'oversampling = '+str(balance)
    
    def entire_evaluation(self, filename = '/mnt/4_TB_HD/ramona/develop/results.csv', filename_trainingmodel=None, params = None):
        '''
        This returns the evaulation values, the confusion matrix and saves the model in results.csv
        '''
        self.evaluation_values()
        print('Accuracy:',self.accuracy,'Log loss:', self.logloss, 'F1 micro:', self.f1_micro,'F1 macro:' ,self.f1_macro)
        self.confusionmatrix()
        self.save_evaluation(filename = filename, filename_trainingmodel = filename_trainingmodel, params = params)
    
    
    def predict(self):
        self.Y_pred = self.model.predict(self.X_test)
        return self.Y_pred
    

    def evaluation_values(self):
        '''
        the function returns the main evaluation values: accuracy, logloss,f1_micro and macro

        Y_pred: 1d array-like, or label indicator array / sparse matrix

        ''' 
        
        #Y_test_indicator_matrix = self.lb.transform(self.Y_test)

        self.logloss = log_loss(self.Y_test_indicator_matrix, self.Y_pred_indicator_matrix)
        self.accuracy = np.mean(self.Y_pred.astype(int) == self.Y_test.astype(int))

        self.f1_micro = metrics.f1_score(self.Y_test, self.Y_pred, average='micro')
        self.f1_macro = metrics.f1_score(self.Y_test,self.Y_pred, average='macro')
        return self.accuracy, self.logloss, self.f1_micro, self.f1_macro

    
    def confusionmatrix(self, calculated_confusion_matrix = None):
        if  calculated_confusion_matrix is None: 
            self.cnf_matrix = confusion_matrix(self.Y_test,self.Y_pred.astype(int))
        else: 
            self.confusion_matrix =  calculated_confusion_matrix
        class_names = [1,2,3,4,5,6,7,8,9]
        #path = '/mnt/4_TB_HD/ramona/deliver/confusion_matrix_ '+name_of_trainingmodel+'.pdf'
        #if self.model != None: 
         #   title = str(self.model.named_steps)
    
        plt.figure()
        self.plot_confusion_matrix(self.cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')
        plt.figure()
        self.plot_confusion_matrix(self.cnf_matrix, classes=class_names, normalize=True,title='Normalized confusion matrix')
        plt.show()

        
    def plot_confusion_matrix(self, cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()


    def save_evaluation(self, filename = '/mnt/4_TB_HD/ramona/develop/results.csv', filename_trainingmodel=None, params = None):
        '''
        Appends the input parameter to a csv data file which is specified through the filename (default: develop/results.csv)
        '''
    
        if self.model is None: 
            self.model = input('input modelname like LogisticRegression() or the model file')
        
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d_%H:%M")
        if filename_trainingmodel is None:
            filename_trainingmodel = now + str(self.model)
        
        if params is None: 
            my_file = Path(filename)
            if my_file.is_file():   
                with open(filename,'a') as f:
                    f = csv.writer(f)
                    f.writerow([str(self.Y_test_file),  filename_trainingmodel,self.accuracy, self.logloss, self.f1_micro, self.f1_macro, self.cnf_matrix])  
            else: 
                with open(filename,'w') as f:
                    f = csv.writer(f)
                    f.writerow([ 'Y test', 'modelname', 'Accuracy', 'Log loss',  'F1 micro', 'F1 macro', 'Confusion Matrix'])
                    f.writerow([str(self.Y_test_file) , filename_trainingmodel, self.accuracy, self.logloss, self.f1_micro, self.f1_macro, self.cnf_matrix])
            print('one row with the evaluation data is appended at {}'.format(filename))
        else: 
            my_file = Path(filename)
            if my_file.is_file():   
                with open(filename,'a') as f:
                    f = csv.writer(f)
                    f.writerow([params[0],params[1],params[2],params[3],params[4],params[5],params[6], params[7], params[8], params[9], params[10], str(self.Y_test_file) , filename_trainingmodel, self.accuracy, self.logloss, self.f1_micro, self.f1_macro, self.cnf_matrix])
            else: 
                with open(filename,'w') as f:
                    f = csv.writer(f)
                    f.writerow(['Iterations','learning_rate', 'depth', 'l2_leaf_reg', 'model_size_reg', 'rsm','random_seed','boosting_type',  'bagging_temperature', 'max_bin', 'approx_on_full_history',  'Y test', 'modelname', 'Accuracy', 'Log loss',  'F1 micro', 'F1 macro', 'Confusion Matrix'])
                    f.writerow([params[0],params[1],params[2],params[3],params[4],params[5],params[6], params[7], params[8], params[9], params[10], str(self.Y_test_file) , filename_trainingmodel, self.accuracy, self.logloss, self.f1_micro, self.f1_macro, self.cnf_matrix])
            print('one row with the evaluation data is appended at {}'.format(filename))
            

