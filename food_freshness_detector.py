# =============================================
#  FOOD FRESHNESS DETECTOR AI
# =============================================

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("Food Freshness Detector AI System")
print("=====================================")
print("TensorFlow version:", tf.__version__)

# Check GPU availability
if tf.config.list_physical_devices('GPU'):
    print("✅ GPU is available for training!")
else:
    print("⚠️ Training on CPU")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def create_synthetic_fruit_dataset():
    """
    Create synthetic dataset for fruit freshness detection
    In real scenario, you would use actual fruit images
    """
    print("🔄 Creating synthetic fruit dataset...")
    
    num_samples = 1200
    img_size = 150
    
    # Fresh fruits: bright colors, uniform texture
    # Rotten fruits: dark colors, spotty texture
    
    X = []
    y = []
    
    for i in range(num_samples):
        if i % 2 == 0:  # Fresh fruit
            # Base fresh fruit color (bright)
            base_color = np.random.uniform(0.6, 0.9, 3)
            img = np.ones((img_size, img_size, 3)) * base_color
            
            # Add fruit shape (circle)
            center_y, center_x = img_size//2, img_size//2
            radius = img_size//3
            y_grid, x_grid = np.ogrid[:img_size, :img_size]
            mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= radius**2
            
            # Make fruit slightly different color
            fruit_color = base_color * np.random.uniform(0.8, 1.2, 3)
            fruit_color = np.clip(fruit_color, 0, 1)
            img[mask] = fruit_color
            
            # Add some texture
            texture = np.random.normal(0, 0.05, (img_size, img_size, 3))
            img = np.clip(img + texture, 0, 1)
            
            label = 1  # Fresh
            
        else:  # Rotten fruit
            # Base rotten fruit color (darker)
            base_color = np.random.uniform(0.3, 0.6, 3)
            img = np.ones((img_size, img_size, 3)) * base_color
            
            # Add fruit shape
            center_y, center_x = img_size//2, img_size//2
            radius = img_size//3
            y_grid, x_grid = np.ogrid[:img_size, :img_size]
            mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= radius**2
            
            # Darker fruit color
            fruit_color = base_color * np.random.uniform(0.7, 0.9, 3)
            img[mask] = fruit_color
            
            # Add rotten spots
            for _ in range(np.random.randint(3, 8)):
                spot_y = np.random.randint(0, img_size)
                spot_x = np.random.randint(0, img_size)
                spot_size = np.random.randint(5, 15)
                spot_mask = (x_grid - spot_x)**2 + (y_grid - spot_y)**2 <= spot_size**2
                img[spot_mask] *= 0.3  # Dark spots
            
            # Add more noise/texture
            texture = np.random.normal(0, 0.1, (img_size, img_size, 3))
            img = np.clip(img + texture, 0, 1)
            
            label = 0  # Rotten
        
        X.append(img)
        y.append(label)
    
    return np.array(X), np.array(y)

# Create dataset
print("📊 Generating dataset...")
X, y = create_synthetic_fruit_dataset()
print(f"✅ Dataset created: {X.shape}")

# Display sample images
plt.figure(figsize=(12, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(X[i])
    plt.title(f"{'Fresh' if y[i] == 1 else 'Rotten'}")
    plt.axis('off')
plt.suptitle('Sample Fruit Images - Fresh vs Rotten', fontsize=16)
plt.tight_layout()
plt.savefig('sample_images.png', dpi=300, bbox_inches='tight')
plt.show()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"📈 Training samples: {X_train.shape[0]}")
print(f"📊 Validation samples: {X_val.shape[0]}")
print(f"🧪 Test samples: {X_test.shape[0]}")

def create_cnn_model():
    """Create CNN model for fruit freshness classification"""
    model = tf.keras.Sequential([
        # First convolutional block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                              input_shape=(150, 150, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Second convolutional block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Third convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Fourth convolutional block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Classifier
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    return model

# Create and compile model
print("Building CNN Model...")
model = create_cnn_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
)

print("✅ Model architecture:")
model.summary()

# Callbacks
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.2, patience=5, min_lr=1e-7, verbose=1)
]

# Train the model
print("🎯 Starting model training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

print("✅ Training completed!")

# Plot training history
plt.figure(figsize=(15, 5))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Precision-Recall plot
plt.subplot(1, 3, 3)
plt.plot(history.history['precision'], label='Training Precision')
plt.plot(history.history['val_precision'], label='Validation Precision')
plt.plot(history.history['recall'], label='Training Recall')
plt.plot(history.history['val_recall'], label='Validation Recall')
plt.title('Precision & Recall')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

# Evaluate the model
print("📊 Evaluating model performance...")
test_loss, test_accuracy, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)

print("\n" + "="*50)
print("📈 FINAL MODEL PERFORMANCE")
print("="*50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Calculate F1-score
test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
print(f"Test F1-Score: {test_f1:.4f}")

# Make predictions
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Classification report
print("\n📋 CLASSIFICATION REPORT")
print("="*50)
print(classification_report(y_test, y_pred, target_names=['Rotten', 'Fresh']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Rotten', 'Fresh'],
            yticklabels=['Rotten', 'Fresh'])
plt.title('Confusion Matrix - Fruit Freshness Detection')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# Save the model
model.save('fruit_freshness_model.h5')
print("Model saved as 'fruit_freshness_model.h5'")

# Prediction function
def predict_freshness(image_array, model, threshold=0.5):
    """
    Predict if fruit is fresh or rotten
    """
    if len(image_array.shape) == 3:
        image_array = np.expand_dims(image_array, axis=0)
    
    prediction = model.predict(image_array, verbose=0)[0][0]
    confidence = prediction if prediction > threshold else 1 - prediction
    label = "Fresh" if prediction > threshold else "Rotten"
    
    return label, confidence, prediction

# Test predictions on some samples
print("\n🔍 SAMPLE PREDICTIONS")
print("="*50)

# Test on 5 random samples
test_indices = np.random.choice(len(X_test), 5, replace=False)

plt.figure(figsize=(15, 8))
for i, idx in enumerate(test_indices):
    test_image = X_test[idx]
    true_label = "Fresh" if y_test[idx] == 1 else "Rotten"
    
    # Make prediction
    pred_label, confidence, raw_pred = predict_freshness(test_image, model)
    
    # Plot
    plt.subplot(2, 5, i+1)
    plt.imshow(test_image)
    plt.title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.2f}')
    plt.axis('off')

plt.suptitle('Sample Predictions - Fruit Freshness Detection', fontsize=16)
plt.tight_layout()
plt.savefig('sample_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

# Business impact analysis
print("\n💡 BUSINESS IMPACT ANALYSIS")
print("="*50)
print("This AI system can help:")
print("✅ Reduce food waste by early detection of spoilage")
print("✅ Improve quality control in food industry")
print("✅ Enhance consumer trust with transparent quality checks")
print("✅ Optimize inventory management for retailers")
print("✅ Support sustainable food practices")

print("\n🎉 FOOD FRESHNESS DETECTOR COMPLETED!")
print("=====================================")
print("Next steps:")
print("1. Deploy as web application")
print("2. Integrate with mobile camera")
print("3. Expand to more fruit types")
print("4. Add real-time processing")

# Save training report
with open('model_report.txt', 'w') as f:
    f.write("FOOD FRESHNESS DETECTOR - MODEL REPORT\n")
    f.write("="*50 + "\n")
    f.write(f"Final Test Accuracy: {test_accuracy:.4f}\n")
    f.write(f"Final Test Precision: {test_precision:.4f}\n")
    f.write(f"Final Test Recall: {test_recall:.4f}\n")
    f.write(f"Final Test F1-Score: {test_f1:.4f}\n")
    f.write(f"Model saved: fruit_freshness_model.h5\n")


print("Model report saved as 'model_report.txt'")
