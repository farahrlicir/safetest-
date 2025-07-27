from flask import Flask, request, jsonify
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Initialize Flask app
app = Flask(__name__)

# Load model
model = tf.keras.models.load_model("path_to_your_model/best_acc_loss_model.keras")

# Load word index
with open("path_to_your_model/word_index_50dim_backbackforwrd.pkl", "rb") as f:
    word_index = pickle.load(f)

# Constants
max_len = 500
oov_token = "<UNK>"
id2label = {
    0: "ADHD",
    1: "Anxiety",
    2: "Bipolar",
    3: "Depression",
    4: "PTSD",
    5: "None"
}

# Preprocessing function
def preprocess(text):
    tokens = text.lower().split()
    tokens = [token if token in word_index else oov_token for token in tokens]
    ids = [word_index.get(token, word_index[oov_token]) for token in tokens]
    padded = pad_sequences([ids], maxlen=max_len, padding='post', truncating='post')
    return padded

# Inference function
@app.route('/predict', methods=['POST'])
def predict():
    text = request.json.get('text')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    x = preprocess(text)
    preds = model.predict(x)
    class_id = preds.argmax()
    label = id2label[class_id]
    confidence = preds[0][class_id]
    
    return jsonify({
        "prediction": f"Predicted class: {label}",
        "confidence": f"Prediction Confidence: {confidence:.2f}"
    })

if __name__ == '__main__':
    app.run(debug=True)
