from flask import Flask, render_template, request
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import json

simple_linear_object = Path(__file__).parent / \
    '../data/simple_linear_regression_model.pkl'
simple_model = joblib.load(simple_linear_object)

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('simple_linear_req.html')


@app.route('/linear/simple', methods=['POST'])
def simple():
    if request.is_json:
        # REST API (JSON)
        dict = request.get_json(silent=True)
        df = pd.DataFrame(dict)

        prediction = simple_model.predict(df)
        results = prediction.to_dict()
        return results, 200
    else:
        # HTTP (HTML)
        radio = request.form['radio_budget']
        dict = {'Radio': [radio]}
        df = pd.DataFrame(dict, dtype=float)

        prediction = simple_model.predict(df)
        df['sales($)'] = prediction
        df = df.rename(columns={'Radio': 'radio($)'})
        results = df.to_html(index=False, justify='left', border=0)
        return render_template('simple_linear_res.html', output=results)


if __name__ == '__main__':
    app.run(debug=True, port=9090)
