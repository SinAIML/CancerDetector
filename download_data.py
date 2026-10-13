"""
Download and Setup Kaggle Histopathologic Cancer Detection Dataset
This script downloads the dataset and sets up the project structure.
"""

import os
import zipfile
import requests
from pathlib import Path
import pandas as pd

def create_directories():
    """Create necessary directories for the project"""
    directories = [
        'data',
        'data/train',
        'data/test',
        'models',
        'sample_images'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_kaggle_data():
    """
    Instructions for downloading Kaggle data manually
    """
    print("=" * 60)
    print("KAGGLE DATA DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print("\nTo download the Histopathologic Cancer Detection dataset:")
    print("\n1. Install Kaggle API:")
    print("   pip install kaggle")
    print("\n2. Get your Kaggle API credentials:")
    print("   - Go to https://www.kaggle.com/account")
    print("   - Click 'Create API Token'")
    print("   - Place kaggle.json in ~/.kaggle/ (or C:\\Users\\<username>\\.kaggle\\)")
    print("\n3. Download the dataset:")
    print("   kaggle competitions download -c histopathologic-cancer-detection")
    print("\n4. Extract to data/ folder:")
    print("   - Extract train.zip to data/train/")
    print("   - Extract test.zip to data/test/")
    print("   - Place train_labels.csv and sample_submission.csv in data/")
    print("\nAlternatively, you can download manually from:")
    print("https://www.kaggle.com/competitions/histopathologic-cancer-detection/data")
    print("=" * 60)

def create_sample_data():
    """Create sample data for testing if full dataset is not available"""
    
    # Create sample train_labels.csv
    sample_labels = pd.DataFrame({
        'id': [f'sample_{i:05d}' for i in range(100)],
        'label': [0, 1] * 50  # Alternating labels for demo
    })
    
    sample_labels.to_csv('data/train_labels.csv', index=False)
    print("Created sample train_labels.csv")
    
    # Create sample submission file
    sample_submission = pd.DataFrame({
        'id': [f'test_{i:05d}' for i in range(50)],
        'label': [0] * 50  # All zeros for demo
    })
    
    sample_submission.to_csv('data/sample_submission.csv', index=False)
    print("Created sample sample_submission.csv")
    
    print("\nSample data created for testing. Replace with actual Kaggle data when available.")

def check_data_availability():
    """Check if the actual dataset is available"""
    required_files = [
        'data/train_labels.csv',
        'data/sample_submission.csv'
    ]
    
    required_dirs = [
        'data/train',
        'data/test'
    ]
    
    files_exist = all(os.path.exists(f) for f in required_files)
    dirs_exist = all(os.path.exists(d) and len(os.listdir(d)) > 0 for d in required_dirs)
    
    if files_exist and dirs_exist:
        print("✅ Dataset appears to be properly set up!")
        return True
    else:
        print("❌ Dataset not found or incomplete.")
        return False

def main():
    """Main setup function"""
    print("Setting up Histopathologic Cancer Detection Project...")
    
    # Create directories
    create_directories()
    
    # Check if data exists
    if not check_data_availability():
        print("\n🔄 Setting up sample data for development...")
        create_sample_data()
        download_kaggle_data()
    else:
        print("✅ Data setup complete!")
    
    print("\n🚀 Project setup finished!")
    print("\nNext steps:")
    print("1. If using sample data, download the actual Kaggle dataset")
    print("2. Run the training notebook to train your model")
    print("3. Use the trained model in your Streamlit app")

if __name__ == "__main__":
    main()
