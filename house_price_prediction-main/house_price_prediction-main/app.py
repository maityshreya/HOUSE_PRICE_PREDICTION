from flask import Flask, render_template, request
import pandas as pd 
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv("house_price_prediction-main\dataset\cleaned_house_data.csv")
model = pickle.load(open(r'house_price_prediction-main\notebook\pune_house_price_model.pkl', 'rb'))


@app.route('/')
def index():
    try:
        locations = sorted(data['location'].unique())
        return render_template('index.html', locations=locations)
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area_type = request.form.get('area_type')
        availability=request.form.get('availability')
        society=request.form.get('society')
        all_no_of_rooms = float(request.form.get('all_no_of_rooms'))
        price = float(request.form.get('price'))
        total_sqft = float(request.form.get('total_sqft'))
        site_location=request.form.get('site_location')
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([[area_type, total_sqft, society, availability,price,site_location,all_no_of_rooms]], columns=['location', 'total_sqft', 'bath', 'BHK'])
        
        # Make prediction
        prediction = model.predict(input_data)[0] * 1e5
            
        return str(np.round(prediction, 2))
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)