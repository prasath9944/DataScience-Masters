from flask import Flask, request, app,render_template,jsonify
from flask import Response
import pickle
import numpy as np
import pandas as pd
from flask_cors import CORS

application = Flask(__name__)
app=application
app = Flask(__name__)
CORS(app)

scaler=pickle.load(open("Model/standardScalar.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))


## Route for Single data point prediction
@app.route('/api/predictdata',methods=['POST'])
def predict_datapoint():
    data = request.get_json()
    name=data.get('name')
    email=data.get('email')
    
    Pregnancies=int(data.get("Pregnancies"))
    Glucose = float(data.get('Glucose'))
    BloodPressure = float(data.get('BloodPressure'))
    SkinThickness = float(data.get('SkinThickness'))
    Insulin = float(data.get('Insulin'))
    BMI = float(data.get('BMI'))
    DiabetesPedigreeFunction = float(data.get('DiabetesPedigreeFunction'))
    Age = float(data.get('Age'))

    # Perform your operations on the data
    
    new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    predict=model.predict(new_data)
       
    if predict[0] ==1 :
        result = 'Diabetic'
    else:
        result ='Non-Diabetic'
    response = {
        "message": "Form data received successfully",
        "receivedData": result
    }
    return jsonify(response)


if __name__=="__main__":
    app.run(host="0.0.0.0")