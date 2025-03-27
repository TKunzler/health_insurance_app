import pickle
import pandas as pd
from flask import Flask, request, Response
from healthinsurance import HealthInsurance

# Load the model
model = pickle.load(open( 'models/model_health_insurance.pkl', 'rb' ) )

# Initialize API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def health_insurance_predict():
    test_json = request.get_json()

    if test_json:  # There is data
        if isinstance(test_json, dict):  # For a single line
            test_raw = pd.DataFrame(test_json, index=[0])
        else:  # For multiple lines
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())

        # Instantiate Healthcare Class
        pipeline = HealthInsurance()
        
        # Feature Engineering
        df1 = pipeline.feature_engineering( test_raw )
    
        # Data Preparation
        df2 = pipeline.data_preparation( df1 )
        
        # Prediction
        df_response = pipeline.get_prediction( model, test_raw, df2 )
    
        # Return
        return df_response
    
    else:
        return Response('{No Data}', status=200, mimetype='application/json')

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run('0.0.0.0', port=port) 