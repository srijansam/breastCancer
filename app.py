from flask import Flask, render_template, redirect, url_for,request,session


import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)
@app.route("/")
def cancer():
    return render_template("cancer.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    df = pd.DataFrame(features_value, columns=features_name)
    model = pickle.load(open('model.pkl', 'rb'))
    output = model.predict(df)
    if output == 4:
        res_val = "a high risk of Breast Cancer"
    else:
        res_val = "a low risk of Breast Cancer"

    return render_template('cancer_result.html', prediction_text='Patient has {}'.format(res_val))
if __name__ == '__main__':
    app.run(debug=True)