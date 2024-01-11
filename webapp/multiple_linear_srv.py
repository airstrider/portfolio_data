from flask import render_template
from pathlib import Path

import pandas as pd
import joblib

multiple_linear_object = Path(__file__).parent / \
    '../model/multiple_linear_regression_model.pkl'
multiple_model = joblib.load(multiple_linear_object)

def multiple_linear(request):
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
