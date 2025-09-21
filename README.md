# Butterfly Identification

## Overview
A deep learning web application for butterfly species identification using Convolutional Neural Networks (CNN). This project classifies butterfly images across 75 different species using TensorFlow/Keras and provides a web interface built with Flask.

## Features
- **Multi-class Classification**: Identifies 75 different butterfly species
- **Deep Learning Model**: Custom CNN architecture with 3.3M parameters
- **Web Interface**: User-friendly Flask web application
- **Image Processing**: Automatic image preprocessing and normalization
- **Real-time Prediction**: Upload images and get instant classification results

## Dataset
- **Source**: Kaggle - Butterfly Image Classification Dataset
- **Classes**: 75 butterfly species
- **Training Images**: 5,200 validated images
- **Validation Images**: 1,299 validated images
- **Test Images**: 2,786 images
- **Image Size**: 128x128 pixels

## Model Architecture
- **Type**: Sequential CNN
- **Input Shape**: (128, 128, 3)
- **Layers**:
  - Conv2D (32 filters, 3x3) + ReLU + MaxPooling2D
  - Conv2D (64 filters, 3x3) + ReLU + MaxPooling2D
  - Conv2D (128 filters, 3x3) + ReLU + MaxPooling2D
  - Flatten
  - Dense (128 units) + ReLU + Dropout (0.5)
  - Dense (75 units) + Softmax
- **Total Parameters**: 3,314,315 (12.64 MB)
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy

## Project Structure
```
butterfly-identification/
├── app.py                    # Flask web application
├── task.ipynb               # Model training notebook
├── best_butterfly_model.keras # Trained model file
├── requirements.txt         # Python dependencies
├── static/                  # CSS and JavaScript files
│   ├── style.css
│   └── script.js
└── templetes/              # HTML templates
    └── index.html
```

## Dependencies
```
Flask==2.3.2
tensorflow==2.13.0
numpy==1.26.0
pandas==2.1.1
h5py==3.9.0
Pillow==10.1.0
gunicorn==22.1.0
```

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Thiyanesh07/butterfly-identification.git
cd butterfly-identification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset (for training)
```python
import kagglehub
path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")
```

### 4. Run the web application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## Usage

### Web Application
1. Start the Flask server using `python app.py`
2. Open your browser and navigate to `http://localhost:5000`
3. Upload a butterfly image using the web interface
4. Get instant species classification results

### Model Training
1. Open `task.ipynb` in Jupyter Notebook
2. Run all cells to:
   - Download and prepare the dataset
   - Build and compile the CNN model
   - Train the model with early stopping and model checkpointing
   - Evaluate model performance
   - Save the trained model

## Model Training Details
- **Training Strategy**: 80/20 train-validation split
- **Batch Size**: 32
- **Max Epochs**: 100
- **Early Stopping**: Patience of 30 epochs on validation loss
- **Model Checkpoint**: Saves best model based on validation accuracy
- **Data Augmentation**: Basic rescaling (1/255)

## File Descriptions

- **`app.py`**: Flask web application with image upload and prediction endpoints
- **`task.ipynb`**: Complete model training pipeline from dataset download to model evaluation
- **`best_butterfly_model.keras`**: Pre-trained model file (12.64 MB)
- **`requirements.txt`**: All Python package dependencies
- **`static/`**: Frontend assets (CSS styling and JavaScript functionality)
- **`templetes/`**: HTML templates for the web interface

## Technical Notes

- **Image Preprocessing**: Images are resized to 128x128 and normalized to [0,1] range
- **Model Format**: Keras format (.keras) for TensorFlow 2.13.0
- **Deployment Ready**: Includes Gunicorn for production deployment
- **Error Handling**: Comprehensive error handling in the Flask application
- **File Management**: Automatic cleanup of uploaded files after prediction

## Performance
The model achieves good classification performance across 75 butterfly species. Training progress can be monitored through the Jupyter notebook with accuracy and loss metrics.

## Contributing
Feel free to fork this repository and submit pull requests for improvements.

## License
This project is open source and available under the MIT License.
