from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)


model=joblib.load('house_price_model.pkl')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'running'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    required = [
        'OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
        'TotalBsmtSF', '1stFlrSF', 'TotRmsAbvGrd', 'FullBath',
        'YearBuilt', 'KitchenAbvGr', 'EnclosedPorch', 'MSSubClass'
    ]
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing field: {field}'}), 400
    
    input_data = pd.DataFrame([{
        'OverallQual':  data['OverallQual'],
        'GrLivArea':    data['GrLivArea'],
        'GarageCars':   data['GarageCars'],
        'GarageArea':   data['GarageArea'],
        'TotalBsmtSF':  data['TotalBsmtSF'],
        '1stFlrSF':     data['1stFlrSF'],
        'TotRmsAbvGrd': data['TotRmsAbvGrd'],
        'FullBath':     data['FullBath'],
        'YearBuilt':    data['YearBuilt'],
        'KitchenAbvGr': data['KitchenAbvGr'],
        'EnclosedPorch':data['EnclosedPorch'],
        'MSSubClass':   data['MSSubClass']
    }])
    predicted_price=model.predict(input_data)[0]
    return jsonify({'predicted_price': float(predicted_price)})

if __name__ == '__main__':
    app.run(debug=True)