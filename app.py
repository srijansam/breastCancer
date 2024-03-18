from flask import Flask, render_template, request, jsonify  # Modified import

import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Removed route for rendering HTML template, as APIs typically don't render templates
# Instead, the API will directly return JSON responses


@app.route('/predict', methods=['POST'])
def predict():
    # Receive input data from POST request
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    
    # Create DataFrame from input data
    df = pd.DataFrame(features_value, columns=features_name)
    
    # Load the trained model
    model = pickle.load(open('model.pkl', 'rb'))
    
    # Make prediction
    output = model.predict(df)
    
    # Define response message based on prediction
    if output == 4:
        res_val = "a high risk of Breast Cancer"
    else:
        res_val = "a low risk of Breast Cancer"
    
    # Return JSON response with prediction result
    return jsonify({'prediction': res_val})


if __name__ == '__main__':
    app.run(debug=True)
