from flask import Flask, render_template, request
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import icecream as ic

naive_bayes_object = Path(__file__).parent / \
    '../model/naive_bayes.pkl'
bayes_model = joblib.load(naive_bayes_object)

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('naive_bayes_req.html')


@app.route('/bayes', methods=['POST'])
def bayes():
    if request.is_json:
        # REST API (JSON)
        dict = request.get_json(silent=True)
        df = pd.DataFrame(dict)
        prediction = bayes_model.predict(df)
        value = get_label(str(prediction[0]))
        results = {'category': str(prediction[0]), 'value': value}
        return results, 200
    else:
        # HTTP (HTML)
        print('HTML')
        gc = pd.Series([request.form['glucose']])
        bp = pd.Series([request.form['bloodpressure']])
        dict = {'glucose': gc, 'bloodpressure': bp}
        df = pd.DataFrame(dict)

        prediction = bayes_model.predict(df)
        res = get_label(str(prediction[0]))
        return render_template('naive_bayes_res.html', data=res)


def get_label(label_num):
    labels = {'0': 'no diabetes',
              '1': 'diabetes'}

    return labels.get(str(label_num))


if __name__ == '__main__':
    app.run(debug=True, port=9093)
