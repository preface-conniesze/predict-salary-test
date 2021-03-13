#importing libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from flask_ngrok import run_with_ngrok

# creating the flask object
app = Flask(__name__)

#making your Flask app available upon running
run_with_ngrok(app)

#loading the model
model = pickle.load(open('model.pkl', 'rb'))

#create routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text='Employee Salary should be $ {}'.format(output))

if __name__ == "__main__":
    app.run()
