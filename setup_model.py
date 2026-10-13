"""
Model Extraction Script for Streamlit App

This script helps extract the trained model from your Jupyter notebook
and save it in a format that can be loaded by the Streamlit app.

Run this script after training your model in the notebook to make it 
available for the Streamlit application.
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers, models
from sklearn.utils import shuffle

def create_and_train_model(sample_size=1000):
    """
    Create and train a sample model for demonstration purposes.
    In practice, you would load your actual trained model.
    """
    print("Creating sample model for demonstration...")
    
    # Create sample data
    X_sample = np.random.random((sample_size, 96, 96, 3))
    y_sample = np.random.randint(0, 2, sample_size)
    
    # Build model (same architecture as in notebook)
    model = tf.keras.Sequential([
        layers.Input(shape=(96, 96, 3)),
        layers.Conv2D(32, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2,2),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    print("Training sample model...")
    # Train for a few epochs (just for demonstration)
    model.fit(X_sample, y_sample, epochs=2, batch_size=32, verbose=1)
    
    return model

def save_model_for_streamlit():
    """
    Save the model in the correct format for the Streamlit app
    """
    try:
        # Option 1: Load existing model from notebook
        # If you have already trained and saved a model, uncomment this:
        # model = tf.keras.models.load_model('path_to_your_trained_model.keras')
        
        # Option 2: Create a demonstration model
        model = create_and_train_model()
        
        # Save the model
        model.save('best_model.keras')
        print("✅ Model saved successfully as 'best_model.keras'")
        print("The Streamlit app can now load this model for predictions.")
        
        # Display model summary
        print("\n📋 Model Summary:")
        model.summary()
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")
        return False

def copy_trained_model():
    """
    Instructions for copying your actual trained model
    """
    print("""
    🔄 To use your actual trained model:
    
    1. In your Jupyter notebook, after training, save the model:
       model.save('best_model.keras')
    
    2. Copy the saved model file to this directory
    
    3. The Streamlit app will automatically load it
    
    📍 Current directory: """ + os.getcwd())

if __name__ == "__main__":
    import os
    
    print("🔬 Histopathologic Cancer Detection - Model Setup")
    print("=" * 50)
    
    choice = input("""
    Choose an option:
    1. Create a demonstration model for testing
    2. Show instructions for using your trained model
    
    Enter choice (1 or 2): """).strip()
    
    if choice == "1":
        success = save_model_for_streamlit()
        if success:
            print("\n🚀 You can now run the Streamlit app:")
            print("streamlit run streamlit_app.py")
    
    elif choice == "2":
        copy_trained_model()
    
    else:
        print("Invalid choice. Please run the script again.")
