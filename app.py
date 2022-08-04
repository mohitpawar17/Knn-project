import numpy as np
from flask import Flask, request, jsonify, render_template
#from flask_ngrok import run_with_ngrok
import pickle
import pandas as pd


app = Flask(__name__)
model = pickle.load(open('Project_4.pkl','rb')) 
dataset= pd.read_csv('train.csv')
X=dataset.iloc[:,[5,6,7,9,4,2]].values
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 4] = labelencoder_X.fit_transform(X[:, 4])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
#run_with_ngrok(app)

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    Age = float(request.args.get('age'))
    SibSp=float(request.args.get('sibsp'))
    Parch=float(request.args.get('parch'))
    Fare=float(request.args.get('fare'))
    Gender=float(request.args.get('gender'))
    Pclass=float(request.args.get('pclass'))

    
    prediction = model.predict(sc.transform([[Age,SibSp,Parch,Fare,Gender,Pclass]]))
    if prediction==0:

      message1="Passenger will not survive."
    
    else:
      message1="Passenger will survive."

    
    
        
    return render_template('index.html', prediction_text='KNN Model predicted : {}'.format(message1))


if __name__ == "__main__":
    app.run(debug=True)
