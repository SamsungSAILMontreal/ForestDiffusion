"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from CTABGANPlus.pipeline.data_preparation import DataPrep
from CTABGANPlus.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

import warnings

warnings.filterwarnings("ignore")

class CTABGAN():

    def __init__(self,
                 pd_data,
                 categorical_columns = [], 
                 log_columns = [],
                 mixed_columns= {},
                 general_columns = [],
                 non_categorical_columns = [],
                 integer_columns = []):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer()
        self.raw_df = pd_data
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.general_columns = general_columns
        self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
                
    def fit(self):
        
        start_time = time.time()
        self.data_prep = DataPrep(self.raw_df,self.categorical_columns,self.log_columns,self.mixed_columns,self.general_columns,self.non_categorical_columns,self.integer_columns)
        self.synthesizer.fit(train_data=self.data_prep.df, categorical = self.data_prep.column_types["categorical"], mixed = self.data_prep.column_types["mixed"],
        general = self.data_prep.column_types["general"], non_categorical = self.data_prep.column_types["non_categorical"])
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self):
        
        sample = self.synthesizer.sample(len(self.raw_df)) 
        sample_df = self.data_prep.inverse_prep(sample)
        
        return sample_df
