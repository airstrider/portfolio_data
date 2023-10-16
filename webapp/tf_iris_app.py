from flask import Flask, render_template, request
import tensorflow as tf
from pathlib import Path

tensorflow_object = Path(__file__).parent / \
    '../model/tensorflow-iris.h5'
model = tf.keras.models.load_model(tensorflow_object)

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('tf_iris_req.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.is_json:
        # REST API (JSON)
        json_req = request.get_json(silent=True)

        sepal_len = json_req['sepal_length']
        sepal_wid = json_req['sepal_width']
        petal_len = json_req['petal_length']
        petal_wid = json_req['petal_width']

        req = [[sepal_len, sepal_wid,
                petal_len, petal_wid]]

        # predict - return ndarray
        prediction = model.predict(req)

        # result
        result = prediction[0].astype('str')
        result = dict(list(enumerate(result, start=0)))
        category = max(result, key=result.get)
        value = get_label(category)
        res = {'category': category, 'value': value}

        # REST test
        return res, 200
    else:
        # HTTP (HTML)
        sepal_len = float(request.form['sepal_length'])
        sepal_wid = float(request.form['sepal_width'])
        petal_len = float(request.form['petal_length'])
        petal_wid = float(request.form['petal_width'])

        req = [[sepal_len, sepal_wid,
                petal_len, petal_wid]]

        # predict - return ndarray
        prediction = model.predict(req)

        # result
        result = prediction[0].astype('str')
        result = dict(list(enumerate(result, start=0)))
        category = max(result, key=result.get)
        res = get_label(category)

        return render_template('tf_iris_res.html', data=res)


def get_label(label_num):
    labels = {'0': 'iris-setosa',
              '1': 'iris-versicolor',
              '2': 'iris-virginica'}

    return labels.get(str(label_num))


if __name__ == '__main__':
    app.run(debug=True, port=9092)
