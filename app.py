from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('fish_species_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from form
    features = [float(request.form['weight']),
                float(request.form['length1']),
                float(request.form['length2']),
                float(request.form['length3']),
                float(request.form['height']),
                float(request.form['width'])]
    
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    # Pass the prediction to the results page
    return render_template('result.html', prediction_text=f'The predicted fish species is {prediction[0]}')

if __name__ == "__main__":
    app.run(debug=True)
