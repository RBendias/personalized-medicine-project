# Import statements
import numpy as np
import pandas as pd

# Read data
train_df = pd.read_csv('..\\training_variants')
text_df = pd.read_csv('..\\training_text', sep = '\|\|', header = None, skiprows = 1, names = ['ID', 'Text'], engine = 'python', encoding = 'utf-8')

#Check out the data frames
print(train_df.shape)
print(train_df.head(5))
print(text_df.shape)
print(text_df.head(5))
