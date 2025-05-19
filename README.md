# Parkinson's Disease Detection using Multilayer Perceptron

## Project Overview
This project builds a neural network-based model to detect Parkinson's disease from biomedical voice measurements. The model uses a multilayer perceptron architecture to classify patients as having Parkinson's disease or not based on various vocal features.

## The Problem
Parkinson's disease is a progressive neurological condition that affects movement, causing symptoms like tremors, stiffness, and impaired balance. Early detection is crucial for better management and treatment. Voice changes are often one of the earliest indicators of the disease, making voice analysis a promising approach for early detection.

## The Solution
This project creates a machine learning model using a multilayer perceptron (MLP) neural network to analyze vocal features and predict the presence of Parkinson's disease. The model learns patterns from voice measurements that differentiate people with Parkinson's from those without it.

## Dataset
The project uses the Parkinson's Data Set from UCI Machine Learning Repository:
- Dataset source: [Parkinson's Data Set on Kaggle](https://www.kaggle.com/datasets/thecansin/parkinsons-data-set)
- The dataset contains 195 instances with 23 features derived from voice recordings
- Target variable: 'status' column (1 for Parkinson's disease, 0 for healthy)

### Citation
If you use this dataset, please cite:
'Exploiting Nonlinear Recurrence and Fractal Scaling Properties for Voice Disorder Detection', Little MA, McSharry PE, Roberts SJ, Costello DAE, Moroz IM. BioMedical Engineering OnLine 2007, 6:23 (26 June 2007)

## Model Architecture
The model is a sequential neural network with the following architecture:
- Input layer: 22 nodes (vocal features)
- First hidden layer: 12 nodes with ReLU activation
- Second hidden layer: 8 nodes with ReLU activation
- Output layer: 1 node with sigmoid activation (binary classification)

## Technologies Used
- Python 3
- Libraries:
  - TensorFlow/Keras: For building and training the neural network
  - Scikit-learn: For data preprocessing and model evaluation
  - Pandas: For data manipulation
  - NumPy: For numerical operations
  - Matplotlib/Seaborn: For data visualization
  - SciKeras: For integrating Keras models with scikit-learn

## Project Structure
```
├── data/
│   └── parkinsons.data
├── notebooks/
│   └── parkinson_detection.ipynb
├── requirements.txt
└── README.md
```

## Setup and Installation
1. Clone the repository
```bash
git clone https://github.com/yowlshi/Parkinson-Detection-Machine-Learning.git
cd parkinsons-detection
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/thecansin/parkinsons-data-set) and place it in the `data` folder

## Model Performance
- Accuracy: 100% on the test set
- Loss: 0.19%

The model achieved excellent performance on the dataset, showing the potential of voice analysis for Parkinson's disease detection.

## Usage
To use the trained model for prediction:

```python
# Prepare your input data
input_data = [feature1, feature2, ..., feature22]  # Voice measurement features
  
# Convert to numpy array and reshape
input_data_np = np.asarray(input_data)
input_data_reshaped = input_data_np.reshape(1, -1)
  
# Standardize the data
scaler = StandardScaler()
scaler.fit(input_data_reshaped)
standard_data = scaler.transform(input_data_reshaped)
  
# Make prediction
pred = model.predict(standard_data)
if (pred[0] == 0):
    print("No Parkinson's")
else:
    print("Parkinson's detected")
```

## Future Improvements
- Gather more diverse data to improve model generalization
- Experiment with different neural network architectures
- Implement feature selection to identify the most relevant vocal parameters
- Develop a user-friendly interface for clinicians


## Acknowledgments
- Dataset providers: Max Little, Oxford University
- UCI Machine Learning Repository
