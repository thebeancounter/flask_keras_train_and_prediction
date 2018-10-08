from obs.utils import get_data
import requests
import json
from obs.utils import check_if_on_heroku, heroku_path, local_path
import os

train_data = {
    "model_name": "model_1",
    "layers": 3,
    "size": 50,
    "dropout": "0.2"
}

path = heroku_path if check_if_on_heroku() else local_path

# "http://127.0.0.1:5000/train_model"
requests.post(os.path.join(path, "train_model"), data = json.dumps(train_data))

(x_train, y_train), (x_test, y_test) = get_data()

data = {'data': x_test[0:200].tolist(), 'model_name': 'model_1'}

# "http://127.0.0.1:5000/predict"
r = requests.post(os.path.join(path, "predict"), data=json.dumps(data))

print(r.text)