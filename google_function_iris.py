import requests 
import pickle 
from google.cloud import storage
from flask import Flask, jsonify, request

def classifier(request):

    if request.method=='GET':
        return "Welcome to 1st deployment"
    if request.method=='POST':
        posted_data = request.get_json()
        sepal_length = posted_data['sepal_length']
        sepal_width = posted_data['sepal_width']
        petal_length = posted_data['petal_length']
        petal_width = posted_data['petal_width']
        storage_client = storage.Client()
        bucket = storage_client.get_bucket('iris-model-dp')

        blob = bucket.blob('models/iris_gb.pickle')
        blob.download_to_filename('/tmp/iris_gb.pkl')
        model = pickle.load(open('/tmp/iris_gb.pkl','rb'))

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
        return output 

