# fuel price prediction using TensorFlow

## Overview  
This project aims to predict the fuel efficiency (MPG - Miles Per Gallon) of automobiles using a regression model built with TensorFlow and Keras. The dataset used for this task is the Auto MPG dataset from Kaggle, containing various car attributes like cylinders, displacement, horsepower, and weight.


## Dataset   
The model is trained on the MPG (Miles Per Gallon) dataset from Kaggle, which provides information about vehicle fuel efficiency.mpg   
The datsetet contains the following information/columns  
- cylinders  
- displacement  
- horsepoweweight  
- acceleration  
- model year  
- origin  
- car name  
- The target variable is MPG (Miles Per Gallon).  
The dataset is preprocessed to handle missing values, normalize features, and split into training and testing sets. 

## Tech Stack
- Python  
- TensorFlow & Keras  
- Pandas & NumPy  
- Matplotlib & Seaborn  
- Scikit-learn  

## Exploratory Data Analysis (EDA)
- Handled missing values (e.g., missing horsepower values).   
- Converted categorical features (like "Origin") into numerical format using one-hot encoding.  
- Visualized relationships between MPG and other features.  
- Normalized the dataset for better model performance.  

## Model Architecture
The model is a deep neural network built using TensorFlow's Keras API:  

- Input layer with normalized features  
- Two hidden layers with 64 neurons each (ReLU activation)  
- Output layer with a single neuron (linear activation) for regression  

- Epochs: 50  
- Batch Size: 32  
- verbose: 1  
- Validation split: 30  
- Train-Test Split: 80-20  

## Model Training
Optimizer: Adam  
Loss Function: Mean Squared Error (MSE)  
Metrics: Mean Absolute Error (MAE), Mean Absolute Percentage Error (MAPE)  

## Results  
### Training Performance:  
- Training Loss: 5.23  
- Training MAE: 5.22   
- Training MAPE: 16.02%   

### Validation Performance:  
- Validation Loss: 6.11  
- Validation MAE: 6.11   
- Validation MAPE: 8.97%  
The model predicts fuel efficiency with an average absolute error of 6.11 MPG on the validation set.

## Loss & Accuracy Graphs 
The loss and accuracy trends during training are given in graph.   

## How to Run

### Clone the repository:
git clone https://github.com/Ayesha0017/FuelEfficiencyPrediction.git  

### Install dependencies:
pip install -r requirements.txt  

### Run the Jupyter Notebook or Python script:
jupyter notebook FuelEfficiency.ipynb  

## Conclusion
- The model demonstrates a reasonable ability to predict fuel efficiency.  
- Further improvements can be made by:  
- Hyperparameter tuning (adjusting learning rate, batch size, number of layers).  
- Feature engineering (adding interaction terms, polynomial features).  
- Using different architectures like CNNs or Transformer-based models.  
