# ❤️ Face Mask Detection Using CNN

**An Image Classification Project Using Deep Learning, Computer Vision, and Streamlit**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-Deep%20Learning-blue?style=for-the-badge)

## Project Overview

Face Mask Detection is a computer vision project that uses a **Convolutional Neural Network (CNN)** to classify facial images into three categories:

- 😷 **With Mask:** A person is wearing a face mask correctly.
- ⚠️ **Incorrect Mask:** A person is wearing a mask incorrectly.
- 🚫 **Without Mask:** A person is not wearing a face mask.

The project combines image preprocessing, deep learning, model evaluation, and an interactive Streamlit web application. Users can upload an image and view the predicted category along with the model's confidence scores.

## Problem Statement

Monitoring whether people are wearing face masks correctly can require time and manual effort, particularly in environments where many people need to be checked.

The objective of this project is to develop an automated image classification system that can identify three face-mask conditions using deep learning. The system aims to demonstrate how computer vision can recognize visual patterns in images and generate predictions through a user-friendly application.

## Project Objectives

- Develop a CNN model for three-class face-mask classification.
- Preprocess facial images into a consistent format for model training.
- Learn visual features using convolutional and pooling layers.
- Evaluate the model using accuracy, precision, recall, F1-score, and a confusion matrix.
- Build an interactive Streamlit application for image-based predictions.
- Display the predicted class and class-wise probability scores.
- Save the trained model for reuse without retraining every time.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming and model implementation |
| TensorFlow | Deep learning framework |
| Keras | CNN model development and training |
| NumPy | Numerical operations and array processing |
| Pillow (PIL) | Image loading and preprocessing |
| Matplotlib | Image visualization and training plots |
| Scikit-learn | Model evaluation and classification metrics |
| Streamlit | Interactive web application |
| Jupyter Notebook | Step-by-step model development |
| Git and GitHub | Version control and project documentation |

## Methodology

### 1. Dataset Preparation

The images are organized into three labeled categories:

- `with_mask`
- `without_mask`
- `incorrect_mask`

Each image is associated with its corresponding class label. The dataset is divided into training and validation subsets for model development and evaluation.

### 2. Image Preprocessing

Before the images are passed to the CNN, the following preprocessing steps are performed:

- Convert images into RGB format.
- Resize images to **128 × 128 pixels**.
- Normalize pixel values from the range 0–255 to the range 0–1.
- Convert images into numerical arrays suitable for TensorFlow.

### 3. CNN Model Development

A Convolutional Neural Network is used to learn visual patterns from the training images.

The model contains:

- **Convolutional layers:** Learn image features such as edges, shapes, and mask-related patterns.
- **Max-pooling layers:** Reduce feature-map dimensions and computational requirements.
- **Flatten layer:** Converts feature maps into a one-dimensional representation.
- **Dense layer:** Learns combinations of features useful for classification.
- **Dropout layer:** Helps reduce overfitting.
- **Softmax output layer:** Produces probability scores for the three classes.

### 4. Model Training

The model is compiled using:

- **Optimizer:** Adam
- **Loss function:** Sparse categorical cross-entropy
- **Evaluation metric:** Accuracy
- **Training epochs:** 10 in the basic Jupyter Notebook implementation

The model learns by comparing its predictions with the correct labels and updating its weights to reduce classification loss.

### 5. Model Evaluation

The model's performance is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix. Training and validation accuracy and loss curves are also plotted to observe learning behavior and identify possible overfitting.

### 6. Streamlit Application

The trained model is integrated into a Streamlit application that allows users to:

- Upload an image in JPG, JPEG, PNG, or WebP format.
- Generate a prediction using the trained CNN.
- View the predicted face-mask category.
- See the confidence score for the predicted class.
- Compare probability scores across all three categories.

## Model Performance

The previously trained model achieved **94.25% test accuracy** on a held-out test dataset of 313 images.

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| Incorrect Mask | 0.89 | 0.95 | 0.92 | 106 |
| With Mask | 0.99 | 0.97 | 0.98 | 104 |
| Without Mask | 0.96 | 0.90 | 0.93 | 103 |
| **Weighted Average** | **0.94** | **0.94** | **0.94** | **313** |

**Interpretation:**

- The `with_mask` class achieved the strongest F1-score in this evaluation.
- The `incorrect_mask` class achieved a recall of 0.95.
- The `without_mask` class had a recall of 0.90, indicating that some images in this class were missed by the model.

These results apply to the previously trained model and its test dataset. The new CNN trained in the Jupyter Notebook may produce different results because its architecture, training settings, and data split can differ.

## Key Features

- Three-class face-mask image classification.
- CNN-based visual feature learning.
- Automated image preprocessing.
- Training and validation performance visualization.
- Classification report and confusion matrix.
- Interactive image-upload interface.
- Predicted category and confidence display.
- Saved Keras model for reuse.

## Project Structure

```text
face-mask-detection/
│
├── Face_Mask_Detection.ipynb
├── app.py
├── predict.py
├── README.md
├── requirements.txt
│
├── models/
│   ├── best_mask_model.keras
│   ├── face_mask_cnn.keras
│   └── class_names.json
│
└── dataset/
    ├── with_mask/
    ├── without_mask/
    └── incorrect_mask/
```

*This is an example structure. Adjust the filenames and folders to match your actual repository. The dataset and trained model files may be kept outside GitHub if they are too large or have redistribution restrictions.*

## Installation and Setup

### Prerequisites

- Python 3.10 or another Python version compatible with your installed TensorFlow release.
- pip package manager.
- Jupyter Notebook or Visual Studio Code.
- Git, if you want to clone the repository.

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/face-mask-detection.git
```

### 2. Navigate to the project folder

```bash
cd face-mask-detection
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install the dependencies

```bash
pip install -r requirements.txt
```

If you want to run the Jupyter Notebook:

```bash
python -m pip install notebook
jupyter notebook
```

Open `Face_Mask_Detection.ipynb` in Jupyter and execute the cells in order.

## Requirements

The `requirements.txt` file should contain the dependencies used by your application and notebook. For example:

```text
tensorflow
numpy
matplotlib
Pillow
scikit-learn
streamlit
notebook
```

For reproducible results, use compatible and tested package versions. You can record your installed versions with:

```bash
pip freeze
```

## Running the Streamlit Application

Ensure that `app.py`, the required model file, and the class names file are available in the expected locations.

Run:

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open the URL in your browser, upload a face image, and view the prediction.

## Making a Prediction

The prediction process follows these steps:

1. Load the saved CNN model.
2. Read the uploaded image.
3. Convert it to RGB.
4. Resize it to 128 × 128 pixels.
5. Normalize pixel values by dividing by 255.
6. Generate class probabilities using the model.
7. Select the class with the highest probability.
8. Display the prediction and confidence score.

The highest probability is the model's predicted confidence, not a guarantee that the prediction is correct.

## Learning Outcomes

Through this project, I gained practical experience in:

- Image preprocessing and computer vision fundamentals.
- CNN architecture and deep learning workflows.
- Model training, validation, and performance evaluation.
- Interpreting precision, recall, F1-score, and confusion matrices.
- Building an interactive machine learning application with Streamlit.
- Saving and reusing trained models.
- Documenting a machine learning project using GitHub.

## 🎥 Project Demo

Watch the complete project demonstration on YouTube:

[▶️ Watch Project Demo](https://youtu.be/zPzJjkHX-YM)

## Future Improvements

- Explore transfer learning with pretrained image-classification models.
- Add more diverse images to improve generalization.
- Evaluate the model on an independent real-world test dataset.
- Investigate data augmentation to improve robustness.
- Improve performance under varying lighting and camera angles.
- Extend the system to detect multiple faces in a single image.
- Deploy the application to a cloud hosting platform.

## Limitations

- Predictions depend on the quality and diversity of the training data.
- Performance may vary with lighting, face angle, image resolution, and unseen environments.
- A high confidence score does not guarantee a correct prediction.
- This project is an educational prototype and should not be used as the sole basis for safety-critical decisions.

## Author

**Prajakta Bhondave**

Aspiring Data Analyst | Python | Machine Learning | Deep Learning | Data Visualization

- **LinkedIn:** [Connect with me](https://www.linkedin.com/in/prajakta-bhondave-773b092a6/)

**Keywords:** Face Mask Detection, CNN, Convolutional Neural Network, Deep Learning, Computer Vision, Image Classification, TensorFlow, Keras, Python, Streamlit, Machine Learning Project.
