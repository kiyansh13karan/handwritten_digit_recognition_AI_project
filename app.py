import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main content styling */
    .main .block-container {
        max-width: 1400px;
        padding: 2rem 3rem;
        background-color: #f8f9fa;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Paragraphs and text */
    p, .stMarkdown {
        color: #34495e;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 1.05rem;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    
    /* Info box styling */
    .info-box {
        background-color: #ffffff;
        border: 1px solid #dfe6e9;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #3498db;
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        margin: 1rem 0;
    }
    
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Metric styling */
    .metric-card {
        background-color: #ffffff;
        border: 1px solid #dfe6e9;
        border-radius: 10px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin: 0.5rem;
    }
    
    .metric-card .stMetric {
        color: #2c3e50;
        font-size: 1.8rem;
        font-weight: 700;
    }
    
    .metric-card .stMetricLabel {
        color: #7f8c8d;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #bdc3c7;
        border-radius: 12px;
        padding: 2rem;
        background-color: #ffffff;
        margin: 1rem 0;
    }
    
    /* Chart styling */
    .stPlotlyChart {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #dfe6e9;
    }

    /* Success message styling */
    .stAlert.stAlert-success {
        background-color: #d5f5e3;
        border-color: #2ecc71;
        color: #27ae60;
        border-radius: 8px;
    }

    /* Error message styling */
    .stAlert.stAlert-error {
        background-color: #fadbd8;
        border-color: #e74c3c;
        color: #c0392b;
        border-radius: 8px;
    }

    /* Loading spinner styling */
    .stSpinner {
        color: #3498db;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #2c3e50;
        color: white;
    }
    
    /* Custom card styling */
    .custom-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #dfe6e9;
    }
    
    /* Custom tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: #ecf0f1;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
        margin-right: 5px !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: #3498db !important;
        color: white !important;
    }
    
    /* Image container */
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1.5rem;
        border: 1px solid #dfe6e9;
        text-align: center;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #7f8c8d;
        font-size: 0.9rem;
        margin-top: 2rem;
        border-top: 1px solid #dfe6e9;
    }
    
    /* Responsive columns */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h2 style='color: white;'>✍ Digit Recognition</h2>
            <p style='color: #bdc3c7;'>Upload an image of a handwritten digit (0-9) for AI prediction</p>
        </div>
    """, unsafe_allow_html=True)
  
    
    st.markdown("### How to Use")
    st.markdown("""
        <div style='color: #bdc3c7;'>
            <ol>
                <li>Upload an image of a digit</li>
                <li>Click 'Predict Digit'</li>
                <li>View results and probabilities</li>
            </ol>
        </div>
    """, unsafe_allow_html=True)
    

# Main content
st.title("Handwritten Digit Recognition")
st.markdown("""
    <p style='color: #7f8c8d; font-size: 1.1rem;'>
        This application uses a neural network to recognize handwritten digits (0-9) from images.
        Upload an image below to see the AI in action.
    </p>
""", unsafe_allow_html=True)

# Load or train the model
@st.cache_resource
def load_or_train_model():
    try:
        # Try loading the .keras model first
        model = tf.keras.models.load_model('handwritten_digits.keras')
        st.success("Model loaded successfully from cache!")
        return model
    except Exception as e:
        with st.spinner("Training new model... This may take a few minutes."):
            try:
                # Load MNIST dataset
                mnist = tf.keras.datasets.mnist
                (X_train, y_train), (X_test, y_test) = mnist.load_data()
                
                # Normalize the data
                X_train = tf.keras.utils.normalize(X_train, axis=1)
                X_test = tf.keras.utils.normalize(X_test, axis=1)
                
                # Create the model
                model = tf.keras.models.Sequential()
                model.add(tf.keras.layers.Flatten())
                model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
                model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
                model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))
                
                # Compile the model
                model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                # Train the model
                model.fit(X_train, y_train, epochs=3, validation_data=(X_test, y_test))
                
                # Evaluate the model
                val_loss, val_acc = model.evaluate(X_test, y_test)
                st.success(f"Model trained successfully! Test accuracy: {val_acc:.2%}")
                
                # Save the model
                model.save('handwritten_digits.keras')
                return model
            except Exception as e:
                st.error(f"Error training model: {str(e)}")
                return None

model = load_or_train_model()

# Main content layout
tab1, tab2 = st.tabs(["📷 Image Upload", "ℹ About Project"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Your Image")
        with st.container():
            uploaded_file = st.file_uploader(
                "Choose an image file (PNG, JPG, JPEG)",
                type=['png', 'jpg', 'jpeg'],
                label_visibility="collapsed"
            )
            
            if uploaded_file is not None:
                # Display the uploaded image
                image = Image.open(uploaded_file)
                st.markdown("#### Uploaded Image Preview")
                st.image(image, caption='Your uploaded digit', width=300)
                
                # Convert image to grayscale and preprocess
                img_array = np.array(image)
                if len(img_array.shape) == 3:
                    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                
                # Resize image to 28x28
                img_array = cv2.resize(img_array, (28, 28))
                
                # Normalize and invert
                img_array = np.invert(img_array)
                img_array = img_array / 255.0
                
                # Reshape for prediction
                img_array = img_array.reshape(1, 28, 28)
                
                if st.button("Predict Digit", type="primary"):
                    if model is not None:
                        with st.spinner("Analyzing the digit..."):
                            # Make prediction
                            prediction = model.predict(img_array)
                            predicted_digit = np.argmax(prediction)
                            confidence = np.max(prediction) * 100
                            
                            # Display results
                            st.markdown("### Prediction Results")
                            col_res1, col_res2 = st.columns(2)
                            with col_res1:
                                st.markdown("""
                                    <div class='metric-card'>
                                        <div class='stMetricLabel'>Predicted Digit</div>
                                        <div class='stMetric'>{}</div>
                                    </div>
                                """.format(predicted_digit), unsafe_allow_html=True)
                            with col_res2:
                                st.markdown("""
                                    <div class='metric-card'>
                                        <div class='stMetricLabel'>Confidence Level</div>
                                        <div class='stMetric'>{:.1f}%</div>
                                    </div>
                                """.format(confidence), unsafe_allow_html=True)
                            
                            # Display prediction visualization
                            st.markdown("### Probability Distribution")
                            fig = plt.figure(figsize=(10, 5))
                            bars = plt.bar(range(10), prediction[0], color='#3498db')
                            bars[predicted_digit].set_color('#e74c3c')
                            plt.title("Model Confidence for Each Digit", pad=20)
                            plt.xlabel("Digit", labelpad=10)
                            plt.ylabel("Probability", labelpad=10)
                            plt.xticks(range(10))
                            plt.ylim(0, 1)
                            plt.grid(axis='y', alpha=0.3)
                            st.pyplot(fig)
                    else:
                        st.error("Model not loaded. Please try again.")

with tab2:
    st.markdown("""
        ### About This Project  
        This handwritten digit recognition system uses a neural network trained on the MNIST dataset,  
        which contains 60,000 training images and 10,000 test images of handwritten digits.  

        #### Technical Details  
        The model architecture consists of:  
        - *Input layer* (784 neurons - flattened 28×28 image)  
        - *Two hidden layers* (128 neurons each with ReLU activation)  
        - *Output layer* (10 neurons with softmax activation)  

        The model achieves approximately *98% accuracy* on the MNIST test set.  

        #### Potential Applications  
        - Digitizing handwritten forms  
        - Bank check processing  
        - Educational tools  
        - Accessibility applications  
    """, unsafe_allow_html=True)