from flask import Flask, request, jsonify
import numpy as np
from joblib import load

app = Flask(__name__)

# Load the saved random forest model
model = joblib.load("./random_forest.joblib")


@app.route("/predict", methods=["POST"])
def predict():
    # Get the input array from the request
    input_array = request.get_json()["input_array"]

    # Convert the input array to a numpy array
    X_new = np.array(input_array)

    # Reshape the array to a 2D array if necessary
    if X_new.ndim == 1:
        X_new = X_new.reshape(1, -1)

    # Make a prediction using the loaded model
    prediction = model.predict(X_new)

    # Return the prediction as a JSON response
    return jsonify({"prediction": prediction[0]})


if __name__ == "__main__":
    app.run(debug=True, port=1080)
