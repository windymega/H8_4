from flask import Flask, render_template, request, url_for
import numpy as np
import pickle
import pandas as pd
import json
import plotly
import plotly.express as px

# Decision Tree Pickle
model1 = pickle.load(open("model/model_dt.pkl", "rb"))
# Linear Regression Pickle
model2 = pickle.load(open("model/model_lr.pkl", "rb"))
# Random Forest Pickle
model3 = pickle.load(open("model/model_rf.pkl", "rb"))


app = Flask(__name__, template_folder="templates")


@app.route("/")
def main():
    return render_template('index.html')


# Decision Tree
@app.route('/predict_1', methods=['POST'])
def predict_1():
    '''
    For rendering results on HTML GUI
    '''
    features_1 = [x for x in request.form.values()]
    final_features_1 = [np.array(features_1)]
    prediction_1 = model1.predict(final_features_1)

    output_1 = round(prediction_1[0], 2)

    return render_template('index.html', prediction_text_1='Prediksi Tarif Decision Tree yaitu : $ {}'.format(output_1))

# Linear regression


@app.route('/predict_2', methods=['POST'])
def predict_2():
    '''
    For rendering results on HTML GUI
    '''
    features_2 = [y for y in request.form.values()]
    final_features_2 = [np.array(features_2)]
    prediction_2 = model2.predict(final_features_2)

    output_2 = round(prediction_2[0], 2)

    return render_template('index.html', prediction_text_2='Prediksi Tarif Linear Regression yaitu : $ {}'.format(output_2))

# Random Forest


@app.route('/predict_3', methods=['POST'])
def predict_3():
    '''
    For rendering results on HTML GUI
    '''
    features_3 = [z for z in request.form.values()]
    final_features_3 = [np.array(features_3)]
    prediction_3 = model3.predict(final_features_3)

    output_3 = round(prediction_3[0], 2)

    return render_template('index.html', prediction_text_3='Prediksi Tarif Random Forest yaitu : $ {}'.format(output_3))


@app.route('/graphic')
def graphic():
    data_kind = [
        ("Lux", 51235),
        ("Lux Black", 51235)
        ("Lux Black XL", 51235)
        ("Lyft Taxi", 51235)
        ("Lyft XL", 51235)
        ("Shared", 51233)
        ("Black SUV", 55096)
        ("Uber XL", 55096)
        ("WAV", 55096)
        ("Black", 55095)
        ("Uber X", 55094)
        ("Uber Pool", 55091)
    ]

    labels = [row[0] for row in data_kind]
    values = [row[1] for row in data_kind]

    return render_template('index.html', labels=labels, values=values)


if __name__ == '__main__':
    app.run(debug=True)
