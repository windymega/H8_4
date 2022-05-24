from flask import Flask, render_template, request
import numpy as np
from logging import debug
import pickle

# For Model
model = pickle.load(open('model/model_ETC.pkl', 'rb'))

app = Flask(__name__, template_folder="templates")


@app.route("/")
def main():
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    usia = float(request.form["usia"])
    anaemia = float(request.form["anaemia"])
    creatinin_fosfokinase = float(request.form["creatinin_fosfokinase"])
    diabetes = float(request.form["diabetes"])
    fraksi_ejeksi = float(request.form["fraksi_ejeksi"])
    tekanan_darah_tinggi = float(request.form["tekanan_darah_tinggi"])
    platelets = float(request.form["platelets"])
    kreatinin_serum = float(request.form["kreatinin_serum"])
    sodium_serum = float(request.form["sodium_serum"])
    jenis_kelamin = float(request.form["jenis_kelamin"])
    perokok = float(request.form["perokok"])
    time = float(request.form["time"])

    float_feature = [
        usia,
        anaemia,
        creatinin_fosfokinase,
        diabetes,
        fraksi_ejeksi,
        tekanan_darah_tinggi,
        platelets,
        kreatinin_serum,
        sodium_serum,
        jenis_kelamin,
        perokok,
        time
    ]

    final_feature = [np.array(float_feature)]
    prediction = model.predict(final_feature)

    output = {
        0: "Tidak Meninggal",
        1: "Meninggal"
    }

    return render_template("index.html", prediction_text="Patients with heart failure have : {}".format(output[prediction[0]]))


if __name__ == '__main__':
    app.run(debug=True)
