from best_pp_model import SimplePricePredictor
from flask import Flask, request, jsonify, render_template
import traceback

# Initialize Flask app
app = Flask(__name__)

# Load the predictor
predictor = SimplePricePredictor()
print("Training model...")
metrics = predictor.train('data/Bengaluru_House_Data_preprocessed.csv')
print("Model trained successfully")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the JSON request
        data = request.get_json()

        size = int(data['size'])
        total_sqft = float(data['total_sqft'])
        bath = int(data['bath'])
        balcony = int(data['balcony'])
        location_name = data['location_name']

        # Make prediction
        result = predictor.predict_price(size, total_sqft, bath, balcony, location_name)

        # Return results as JSON
        return jsonify(result)

    except Exception as e:
        error_message = traceback.format_exc()
        return jsonify({"error": str(e), "trace": error_message})

if __name__ == "__main__":
    app.run(debug=True)
