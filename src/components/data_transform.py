import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler


from src.logger import logging
from src.exceptions import CustomException
from src.components.data_ingestion import DataIngestion

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical = ["writing_score","reading_score"]
            categorical = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                ]
            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('standard_scaler',StandardScaler(with_mean=False))
                ]
            )
            pre_processor = ColumnTransformer(
                [
                    ("num_transformer",num_pipeline,numerical),
                    ("cat_transformer",cat_pipeline,categorical)
                ]
            )

            return pre_processor
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_transformation(self,train_path,test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logging.info("Read datasets after ingestion")
        preprocessor = self.get_data_transformer_obj()
        target = "math_score"

        target_train_df = train_df[target]
        input_df = train_df.drop(columns = [target],axis = 1)

        target_test_df = test_df[target]
        input_test_df = test_df.drop(columns = [target],axis =1)
        logging.info("Transformation in progress")
        input_arr = preprocessor.fit_transform(input_df)
        input_test_arr = preprocessor.fit_transform(input_test_df)
        
        target_arr = preprocessor.fit_transform(target_train_df)
        target_test_arr = preprocessor.fit_transform(target_test_df)
        logging.info("Transformation Done")
        return [ self.data_transformation_config,input_arr,input_test_arr,target_arr,target_test_arr]
    
if __name__ == "__main__":
    ingestion_obj = DataIngestion()
    train, test = ingestion_obj.initiate_data_ingestion()
    transformer_obj = DataTransformation()
    [_,input_train_arr,input_test_arr,output_train,output_test] = transformer_obj.start_transformation(train,test)

