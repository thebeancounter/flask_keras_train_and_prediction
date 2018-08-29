from utils import get_data
#from flask import requests
import requests
import json

(x_train, y_train), (x_test, y_test) = get_data()
data = {'data': x_test[0:200].tolist(), 'model_name': 'model_1'}
r = requests.post("http://127.0.0.1:5000/predict", data=json.dumps(data))
print(r.text)