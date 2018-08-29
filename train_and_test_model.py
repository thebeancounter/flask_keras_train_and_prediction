from utils import get_data
import requests
import json

train_data = {
    "model_name": "model_1",
    "layers": 3,
    "size": 50,
    "dropout": "0.2"
}
requests.post("http://127.0.0.1:5000/train_model", data = json.dumps(train_data))

(x_train, y_train), (x_test, y_test) = get_data()

data = {'data': x_test[0:200].tolist(), 'model_name': 'model_1'}
r = requests.post("http://127.0.0.1:5000/predict", data=json.dumps(data))
print(r.text)