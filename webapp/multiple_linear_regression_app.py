from flask import Flask, render_template, request
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import json

multiple_linear_object = Path(__file__).parent / \
    '../data/multiple_linear_regression_model.pkl'
multiple_model = joblib.load(multiple_linear_object)

app = Flask(__name__)


@app.route('/')
def main():
    return render_template('multiple_linear_req.html')


@app.route('/linear/multiple', methods=['POST'])
def multiple():
    if request.is_json:
        # REST API (JSON)
        dict = request.get_json(silent=True)
        df = pd.DataFrame(dict)

        prediction = multiple_model.predict(df)
        results = prediction.to_dict()
        return results, 200
    else:
        # HTTP (HTML)
        tv = pd.Series([request.form['tv_level']], dtype='str')
        rd = pd.Series([request.form['radio_budget']], dtype='float')
        dict = {'TV': tv, 'Radio': rd}
        df = pd.DataFrame(dict)

        prediction = multiple_model.predict(df)
        df['sales($)'] = prediction
        df = df.rename(columns={'TV': 'tv(level)', 'Radio': 'radio($)'})
        results = df.to_html(index=False, justify='left', border=0)
        return render_template('multiple_linear_res.html', output=results)


if __name__ == '__main__':
    app.run(debug=True, port=9091)
