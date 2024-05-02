from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application = Flask(__name__)

app = application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            
            Gender = request.form.get('Gender'),
            Sleep_type = request.form.get('Sleep_type'),
            screen_time = request.form.get('screen_time'),
            Lunch_type = request.form.get('Lunch_type'),
            study_preparation = request.form.get('study_preparation'),
            Cat_1 = request.form.get('Cat_1'),
            Cat_2 = request.form.get('Cat_2'),
            Student_Name = request.form.get('Student_Name'),

        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results = results[0::1])
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug = True)        


