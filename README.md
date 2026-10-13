# Histopathologic Cancer Detection Streamlit App

A web application for detecting metastatic breast cancer in histopathologic images using a Convolutional Neural Network (CNN).

## 🔬 Project Overview

This Streamlit application provides an interactive interface for the CNN model developed for histopathologic cancer detection. The model analyzes 96x96 pixel tissue images and provides binary classification results to identify potential cancerous regions.

## 🚀 Features

- **Image Upload & Prediction**: Upload histopathologic images and get instant cancer detection results
- **Interactive Visualizations**: Real-time gauge charts showing prediction confidence
- **Model Analysis**: Comprehensive performance metrics and confusion matrices
- **Architecture Display**: Detailed model structure and layer information
- **Educational Content**: Information about the dataset and methodology

## 📋 Requirements

- Python 3.8 or higher
- GPU support (optional, for faster inference)
- Required packages (see requirements.txt)

## 🛠️ Installation

1. **Clone or download the project files**
   ```bash
   cd "path\to\your\project"
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Running the Application

1. **Start the Streamlit app**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, navigate to the URL shown in the terminal

## 📁 File Structure

```
Streamlit Project CNN/
├── streamlit_app.py           # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                 # This file
├── best_model.keras          # Trained model (if available)
└── cnn-gpu-histopathologic-cancer-detection (1).ipynb  # Original notebook
```

## 🔍 How to Use

### 1. Home Page
- Overview of the project and dataset information
- Quick statistics about model performance

### 2. Image Prediction
- Upload a histopathologic image (PNG, JPG, JPEG, TIF formats)
- View the original image and preprocessing information
- Get instant prediction results with confidence scores
- Interactive gauge chart showing cancer probability

### 3. Model Analysis
- View model architecture and layer details
- Examine confusion matrices for training and validation sets
- Analyze training history with accuracy and loss plots
- Review potential improvements and optimization strategies

### 4. About
- Detailed project information and technical specifications
- Dataset details and methodology
- Performance metrics and limitations

## 📊 Model Information

- **Architecture**: Custom CNN with batch normalization and dropout
- **Input Size**: 96x96x3 (RGB images)
- **Output**: Binary classification (cancer/no cancer)
- **Training**: 15 epochs with early stopping
- **Performance**: ~89% training accuracy

## ⚠️ Important Notes

1. **Model File**: If you have a trained model file (`best_model.keras`), place it in the same directory as the app
2. **Demo Mode**: Without a trained model, the app runs in demo mode with simulated predictions
3. **Medical Use**: This is for educational purposes only and should not be used for actual medical diagnosis
4. **GPU Support**: TensorFlow will automatically use GPU if available and properly configured

## 🔧 Troubleshooting

### Common Issues:

1. **Import Errors**
   - Ensure all packages are installed: `pip install -r requirements.txt`
   - Check Python version compatibility

2. **Model Loading Issues**
   - Verify the model file path and name
   - Ensure TensorFlow version compatibility

3. **Memory Issues**
   - Close other applications to free up RAM
   - Reduce batch size if modifying the code

4. **Port Issues**
   - If port 8501 is busy, Streamlit will suggest an alternative
   - Or specify a different port: `streamlit run streamlit_app.py --server.port 8502`

## 🎨 Customization

You can modify the app by editing `streamlit_app.py`:

- **Styling**: Modify the CSS in the markdown sections
- **Functionality**: Add new prediction features or visualizations
- **Layout**: Change the page structure or add new pages
- **Model**: Update the model architecture or loading mechanism

## 📈 Performance Optimization

For better performance:

1. **Enable GPU**: Ensure TensorFlow-GPU is properly installed
2. **Batch Processing**: Modify code to handle multiple images at once
3. **Model Optimization**: Consider model quantization or pruning
4. **Caching**: Streamlit's `@st.cache_resource` is already used for model loading

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is for educational purposes. Please ensure compliance with any dataset licenses and medical regulations if adapting for clinical use.

## 📧 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Verify all requirements are installed
3. Check the Streamlit documentation for additional help
4. Review the original Jupyter notebook for model details

---

**Happy Cancer Detection! 🔬**
