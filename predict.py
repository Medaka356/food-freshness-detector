# Simple prediction script
import tensorflow as tf
import numpy as np
import cv2
import sys

def load_and_preprocess_image(image_path):
    """Load and preprocess image for prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = img / 255.0  # Normalize
    return np.expand_dims(img, axis=0)

def predict_fruit_freshness(image_path, model_path='fruit_freshness_model.h5'):
    """Predict if fruit in image is fresh or rotten"""
    try:
        # Load model
        model = tf.keras.models.load_model(model_path)
        
        # Preprocess image
        processed_image = load_and_preprocess_image(image_path)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)[0][0]
        
        # Interpret results
        if prediction > 0.5:
            label = "Fresh"
            confidence = prediction
        else:
            label = "Rotten" 
            confidence = 1 - prediction
            
        return label, confidence, prediction
        
    except Exception as e:
        return f"Error: {str(e)}", 0, 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    label, confidence, raw_score = predict_fruit_freshness(image_path)
    
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}")
    print(f"Raw Score: {raw_score:.4f}")