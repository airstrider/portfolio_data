from flask_restful import reqparse
from flask import Flask, render_template, request

import numpy as np
import pickle as p
import json

modelfile = '/Users/ryankim/Development/workspace/study/goog_cert/portfolio_data/portfolio_data/data/iris_prediction.pickle'
model = p.load(open(modelfile, 'rb'))

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('home.html')


@app.route('/predict/', methods=['POST'])
def predict():
    sepal_length = request.form['sepal_length']
    sepal_width = request.form['sepal_width']
    petal_length = request.form['petal_length']
    petal_width = request.form['petal_width']
    # convert input to array
    arr = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width]],
        dtype=float)

    # predict - return ndarray
    prediction = model.predict(arr)

    # result
    # out = prediction[0]
    out = get_label(prediction[0])
    print('------\nout:', out)

    return render_template('after.html', data=out)

    # REST test
    # return out, 200


def get_label(label_num):
    labels = {'0': 'iris-setosa',
              '1': 'iris-versicolor',
              '2': 'iris-virginica'}

    return labels.get(str(label_num))


if __name__ == '__main__':
    app.run(debug=True, port=9090)

# 5.0 3.6 1.4 0.2   0
# 6.7 3.0 5.2 2.3   2
