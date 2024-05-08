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

@app.route('/login')
def login():
    return render_template('login.html') 

@app.route('/home.html')
def home():
    return render_template('home.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    savingAllowed = True #remove and add as function argument

    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            
            Student_Name = request.form.get('Student_Name'),
            Gender = request.form.get('Gender'),
            Sleep_type = request.form.get('Sleep_type'),
            screen_time = request.form.get('screen_time'),
            Lunch_type = request.form.get('Lunch_type'),
            study_preparation = request.form.get('study_preparation'),
            Cat_1 = request.form.get('Cat_1'),
            Cat_2 = request.form.get('Cat_2'),
            

        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        

        predict_pipeline = PredictPipeline()
        
        results = predict_pipeline.predict(pred_df)
        
        if savingAllowed:
            with open("data.csv", "a") as mFile:
                
                mFile.writelines(f"{data.Student_Name},{data.Gender},{data.Sleep_type},{data.Lunch_type},{data.study_preparation},{data.screen_time},{data.Cat_1},{data.Cat_2},{results[0]} \n")
        
        print("after Prediction")
        ranking()
        return render_template('home.html',results = results[0])
    
def ranking():
    import pandas as pd

    # Load the data from the CSV file
    df = pd.read_csv('data.csv')

    # Sort the data in descending order based on the 'Predicted CGPA' column
    df = df.sort_values(by='Predicted CGPA', ascending=False)

    # Assign ranks based on predicted CGPA, handling ties by assigning the same rank
    df['Rank'] = df['Predicted CGPA'].rank(method='dense', ascending=False).astype(int)

    # Reset the index starting from 0
    df.reset_index(drop=True, inplace=True)

    # Save the ranked data to a new CSV file
    df.to_csv('ranking.csv', index=False)



if __name__=="__main__":
    app.run(host="0.0.0.0",debug = True)        

