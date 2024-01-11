from flask import render_template
from pathlib import Path

import pandas as pd
import joblib

naive_bayes_object = Path(__file__).parent / \
    '../model/naive_bayes.pkl'
bayes_model = joblib.load(naive_bayes_object)

def bayes(request):
    if request.is_json:
        # REST API (JSON)
        dict = request.get_json(silent=True)
        df = pd.DataFrame(dict)
        prediction = bayes_model.predict(df)
        value = __get_label(str(prediction[0]))
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
        res = __get_label(str(prediction[0]))
        return render_template('naive_bayes_res.html', data=res)

def __get_label(label_num):
    labels = {'0': 'no diabetes',
              '1': 'diabetes'}

    return labels.get(str(label_num))
