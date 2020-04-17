# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 16:53:54 2020

@author: Mainak Kundu
"""

import os
from flask import Flask, jsonify, request
from flask_restful import Api, Resource
#from model.Train import train_model
from sklearn.externals import joblib
import pickle

app = Flask(__name__)
api = Api(app)

os.chdir(r'C:\Users\Mainak Kundu\Desktop\DEPLOYMENT-GCP')


with open('iris_gb.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

class MakePrediction(Resource):
    @staticmethod
    def post():
        posted_data = request.get_json()
        sepal_length = posted_data['sepal_length']
        sepal_width = posted_data['sepal_width']
        petal_length = posted_data['petal_length']
        petal_width = posted_data['petal_width']

        prediction = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])[0]
        if prediction == 0:
            predicted_class = 'Iris-setosa'
        elif prediction == 1:
            predicted_class = 'Iris-versicolor'
        else:
            predicted_class = 'Iris-virginica'

        return jsonify({
            'Prediction': predicted_class
        })


api.add_resource(MakePrediction, '/predict')


if __name__ == '__main__':
    app.run(debug=True)