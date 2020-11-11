from sklearn import metrics
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.preprocessing import LabelBinarizer


class Evaluation:
    def __init__(
        self, y_test, y_pred,
    ):
        self.y_test = y_test
        self.y_pred = y_pred
        
        self.lb = LabelBinarizer()
        self.lb = self.lb.fit([1,2,3,4,5,6,7,8,9])
        
        if self.y_pred.ndim == 1: # No probabilities
            self.y_pred = self.y_pred.astype(int)
            self.y_pred_indicator_matrix = self.lb.transform(self.y_pred) 
        else:  
            self.y_pred_indicator_matrix = self.y_pred
            self.y_pred = self.lb.inverse_transform(self.y_pred,  threshold = 0.5)
            
        if self.y_test.ndim == 1:
            self.y_test_indicator_matrix = self.lb.transform(self.y_test.astype(int))
            self.y_test = self.y_test.astype(int)
        else:
            self.y_test_indicator_matrix = self.y_test
            self.y_test = self.lb.inverse_transform(self.y_test, threshold = 0.5)
         
        

    def entire_evaluation(
        self, filename="results.csv", filename_trainingmodel=None, params=None,
    ):
        """
        This returns the evaulation values, the confusion matrix and saves the model in results.csv
        """
        self.calculate_evaluation_values()
        print(
            "Accuracy:",
            self.accuracy,
            "Log loss:",
            self.logloss,
            "F1 micro:",
            self.f1_micro,
            "F1 macro:",
            self.f1_macro,
        )
        self.plot_confusionmatrices()

    def calculate_evaluation_values(self):
        """
        The function returns the main evaluation values: accuracy, logloss,f1_micro and macro
        Y_pred: 1d array-like, or label indicator array / sparse matrix
        """
        self.logloss = log_loss(
            self.y_test_indicator_matrix, self.y_pred_indicator_matrix
        )
        self.accuracy = np.mean(self.y_pred.astype(int) == self.y_test.astype(int))
        self.f1_micro = metrics.f1_score(self.y_test, self.y_pred, average="micro")
        self.f1_macro = metrics.f1_score(self.y_test, self.y_pred, average="macro")
        return self.accuracy, self.logloss, self.f1_micro, self.f1_macro

    def plot_confusionmatrices(self):
        """
        This function plots normalized and not normalized confusion matrix.
        """
        self.cnf_matrix = confusion_matrix(self.y_test, self.y_pred.astype(int))
        class_names = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        plt.figure()
        self.plot_confusion_matrix(
            self.cnf_matrix,
            classes=class_names,
            normalize=False,
            title="Confusion matrix, without normalization",
        )

        plt.figure()
        self.plot_confusion_matrix(
            self.cnf_matrix,
            classes=class_names,
            normalize=True,
            title="Normalized confusion matrix",
        )
        plt.show()

    # sklearn function: https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    def plot_confusion_matrix(
        self, cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
    ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            with np.errstate(all="ignore"):
                cm = cm / cm.sum(axis=1, keepdims=True)
            # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
