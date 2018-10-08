from flask import Flask
from flask_cors import CORS
import os
from obs.utils import check_if_on_heroku

app = Flask(__name__)
CORS(app)
models = {}
os.environ['KERAS_BACKEND'] = "theano"

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route("/train_model", methods=['POST'])
def train_model():
    from obs.utils import build_model, get_data
    import json
    from flask import request, jsonify
    data = json.loads(request.data)
    model = build_model(**data)
    (x_train, y_train), (x_test, y_test) = get_data()
    hist = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(x_test, y_test)).history
    models[data["model_name"]] = model
    return jsonify(hist)


@app.route("/predict", methods=['POST'])
def predict():
    import json
    from flask import request, jsonify
    import numpy as np
    import tensorflow as tf

    tf.keras.backend.clear_session()

    data = json.loads(request.data)

    model_name = data["model_name"]

    if not isinstance(data["data"], np.ndarray):
        data["data"] = np.array(data["data"])

    if model_name in models.keys():
        prediction = np.argmax(models[model_name].predict(data["data"]), axis=1).tolist()
    else:
        prediction = None

    return jsonify(prediction)


if not check_if_on_heroku():
    print("not on heroku!! ")
    # this is not needed on heroku!
    app.run(debug=True)