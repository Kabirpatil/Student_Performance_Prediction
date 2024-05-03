# To predict the model.
import os        
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path= "artifacts\model.pkl"
            preprocessor_path = 'artifacts\preprocessor.pkl'
            print("Before Loading")
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            print("After Loading")
            data_scaled  = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__( self,
        Student_Name: str,
        Gender: str,
        Sleep_type: str,
        screen_time:str,
        Lunch_type: str,
        study_preparation: str,
        Cat_1: int,
        Cat_2: int):
        
        self.Student_Name = Student_Name

        self.Gender = Gender

        self.Sleep_type = Sleep_type

        self.screen_time = screen_time

        self.Lunch_type = Lunch_type

        self.study_preparation = study_preparation

        self.Cat_1 = Cat_1

        self. Cat_2 =  Cat_2

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                
                "Gender": [self.Gender],
                "Sleep_type": [self.Sleep_type],
                "screen_time": [self.screen_time],
                "Lunch_type": [self.Lunch_type],
                "study_preparation": [self.study_preparation],
                "Cat_1": [self.Cat_1],
                "Cat_2": [self. Cat_2],
                "Student_Name" : [self.Student_Name],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)



