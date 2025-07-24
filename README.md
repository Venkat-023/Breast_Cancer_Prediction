🧬 Breast Cancer Prediction
This project predicts the presence of breast cancer using two approaches:
Traditional Machine Learning models – including Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest.
Deep Learning using Artificial Neural Network (ANN) – built and optimized using Keras and Keras Tuner.
The models were trained and evaluated on a dataset from Kaggle containing 32 clinical and diagnostic features.

📊 Objective
To develop an accurate and robust predictive system that aids in the early diagnosis of breast cancer, improving clinical decision support using patient diagnostic data.

📁 Dataset
Source: Kaggle - Breast Cancer Wisconsin (Diagnostic) Dataset
Samples: 569
Features: 32 (mostly numerical)

Target Variable: diagnosis
M = Malignant
B = Benign

🔍 Approach 1: Traditional Machine Learning
⚙️ Preprocessing
Plotted correlation heatmap with target to identify the most important features
Removed columns with least correlation
Since all features were numerical, no encoding was needed
Used StandardScaler for feature scaling
Explored pair plots, which revealed distinct class separation

🧠 Models Used
Model	Accuracy (After Tuning)
K-Nearest Neighbors	94.75%
Random Forest	94.74%
Logistic Regression	92.98%

🛠️ Hyperparameter Tuning
Used RandomizedSearchCV for model optimization
Tuned:
n_neighbors, weights, metric for KNN
n_estimators, max_depth for Random Forest

📈 Evaluation Metrics
Accuracy
Confusion Matrix (visualized using Seaborn heatmap)
Precision, Recall, and F1-score.

🔍 Approach 2: Deep Learning (ANN)
🔬 EDA & Preprocessing
Full Exploratory Data Analysis conducted in a separate notebook
Identified and removed less relevant features using correlation
Data was clearly linearly separable, making it suitable for ANN
Applied scaling before feeding into the neural network

🧠 Model: ANN with Keras
Built using Keras Sequential API
Performed hyperparameter tuning using Keras Tuner

Optimized:
Number of layers and neurons
Activation functions
Batch size and number of epochs
Optimizer types (Adam, RMSProp, etc.)

🏆 ANN Accuracy: 99.12%
Achieved using the tuned deep learning model
Evaluated using:
Confusion Matrix
F1-score, Precision, Recall

📌 Key Highlights
Two-pronged approach: Classical ML and Deep Learning
High-performing traditional models with >94% accuracy
ANN model achieved 99.12% accuracy, demonstrating excellent generalization
Applied hyperparameter tuning in both approaches for performance optimization
Robust EDA for feature selection and model readiness

🛠️ Tech Stack
Languages: Python

Libraries:
Pandas, NumPy, Matplotlib, Seaborn
Scikit-learn (for ML models and tuning)
Keras, TensorFlow (for ANN)
Keras Tuner (for ANN hyperparameter tuning)
Platform: Jupyter Notebook

▶️ How to Run
Clone the repository:
bash
Copy
Edit
git clone https://github.com/yourusername/breast-cancer-prediction.git
cd breast-cancer-prediction
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run Notebooks:

breast_cancer_prediction.ipynb: Traditional ML approach with EDA

breast_cancer_prediction_with_dvp.ipynb: ANN approach with hyperparameter tuning

