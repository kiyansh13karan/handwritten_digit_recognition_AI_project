# Handwritten Digit Recognition

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

An interactive web application that uses deep learning to recognize handwritten digits with high accuracy.

## 📋 Overview

This project implements a neural network trained on the MNIST dataset to recognize handwritten digits (0-9) from images. The system features a user-friendly web interface built with Streamlit, allowing users to upload images of handwritten digits and receive instant predictions with confidence scores.

## ✨ Features

- **Deep Learning Model**: Neural network trained on the MNIST dataset with ~98% accuracy
- **Interactive Web Interface**: Clean, responsive UI built with Streamlit
- **Real-time Predictions**: Upload images and get instant digit recognition
- **Confidence Visualization**: Bar chart showing prediction probabilities for each digit
- **Responsive Design**: Works on desktop and mobile devices

## 🖼️ Screenshots

![App Screenshot]![Image](https://github.com/user-attachments/assets/6bacc7e5-b527-42c1-97c8-c512a202a798)

## 🧠 Model Architecture

The neural network consists of:
- Input layer (784 neurons - flattened 28×28 pixel images)
- Two hidden layers (128 neurons each with ReLU activation)
- Output layer (10 neurons with Softmax activation)

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/handwritten-digit-recognition.git
cd handwritten-digit-recognition
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

### Running the Web App

```bash
streamlit run app.py
```

This will start the application and open it in your default web browser. You can then:
1. Upload an image containing a handwritten digit
2. Click "Predict Digit"
3. View the predicted digit and confidence levels

### Using the Script Directly

```bash
python handwritten_digits_recognition.py
```

This script will train a new model if one doesn't exist, then process any images in the `digits/` directory.

## 🛠️ Project Structure

```
handwritten-digit-recognition/
├── app.py                        # Streamlit web application
├── handwritten_digits_recognition.py  # Core model training script
├── handwritten_digits.keras      # Saved model (generated after training)
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
└── digits/                       # Directory for test images
    ├── digit1.png
    ├── digit2.png
    └── ...
```

## 👥 Team Members

- Karan Nayal : Model Development
- Paras Joshi 2: Web UI Development
- Diya Tiwari 3: Image Processing & Integration
- Manish Bhatt 4: Documentation & DevOps

## 📚 Technical Implementation

- **TensorFlow/Keras**: For building and training the neural network
- **OpenCV**: For image preprocessing and manipulation
- **Streamlit**: For building the interactive web interface
- **Matplotlib**: For visualizing prediction results
- **NumPy**: For numerical operations

## 🔮 Future Enhancements

- Support for drawing digits directly in the browser
- Batch processing of multiple images
- Extended model to recognize alphabets
- Improved preprocessing for better accuracy with varied input images
- Model retraining option in the UI

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- The MNIST dataset creators
- TensorFlow and Keras documentation
- Streamlit community for UI inspiration
