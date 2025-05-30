import os
import sys

from src.logger import logging
from src.exceptions import CustomException
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts',"train.csv")
    test_data_path : str = os.path.join('artifacts',"test.csv")
    raw_data_path :str = os.path.join('artifacts',"raw_data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")
        try:
            df=pd.read_csv('notebook/stud.csv')
            logging.info('Reading dataset')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok= True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header= True)

            train_set,test_set = train_test_split(df,test_size=0.25,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path)
            test_set.to_csv(self.ingestion_config.test_data_path)
            logging.info("Train test split done,Data Ingestion Done")
            return (self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path)
            
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()