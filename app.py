from flask import Flask, request, jsonify
import pandas as pd
import logging
import traceback
import mlflow.sklearn

#setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

def add_features(X):
    X = X.copy()
    X['RoomsPerBedroom'] = X['AveRooms'] / X['AveBedrms']
    X['PopulationPerBedroom'] = X['Population'] / X['AveBedrms']
    X['MedInc_HouseAge'] = X['MedInc'] * X['HouseAge']
    X['Latitude_Longitude'] = X['Latitude'] * X['Longitude']
    return X

# Load the trained model and preprocessing pipeline
def load_model(model_name="model", stage="Production"):
    logging.info(f"Loading model {model_name} from stage {stage}")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{stage}")
    return model

pipeline = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Received prediction request: {data}")

        input_data = pd.DataFrame([data])
        prediction = pipeline.predict(input_data)[0]
        prediction = float(prediction)

        logging.info(f"Prediction: {prediction}")
        return jsonify({'prediction': prediction})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        logging.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)