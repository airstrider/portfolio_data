from flask import render_template
from pathlib import Path

import pandas as pd
import joblib

simple_linear_object = Path(__file__).parent / \
    '../model/simple_linear_regression_model.pkl'
simple_model = joblib.load(simple_linear_object)

def simple_linear(request):
    print("request type: ",type(request))
    if request.is_json:
        # REST API (JSON)
        print("JSON")
        dict = request.get_json(silent=True)
        df = pd.DataFrame(dict)

        prediction = simple_model.predict(df)
        results = prediction.to_dict()
        return results, 200
    else:
        # HTTP (HTML)
        print("HTML")
        radio = request.form['radio_budget']
        dict = {'Radio': [radio]}
        df = pd.DataFrame(dict, dtype=float)

        prediction = simple_model.predict(df)
        df['sales($)'] = prediction
        df = df.rename(columns={'Radio': 'radio($)'})
        results = df.to_html(index=False, justify='left', border=0)
        return render_template('simple_linear_res.html', output=results)
