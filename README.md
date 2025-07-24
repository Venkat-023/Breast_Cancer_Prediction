This project predicts the presence of breast cancer using various machine learning models, including Logistic Regression, K-Nearest Neighbors (KNN), and Random Forest.
The model was trained and evaluated on a dataset from Kaggle with around 32 clinical and diagnostic features.

📊 Objective
To develop an accurate predictive system that can assist in early diagnosis of breast cancer using patient clinical data.

📁 Dataset
Source: Kaggle (Breast Cancer Dataset)
Contains ~32 features such as mean radius, texture, perimeter, area, smoothness, etc.
Target variable: diagnosis (benign or malignant)

⚙️ Preprocessing
Handled missing values and outliers through domain-informed strategies.
Performed feature scaling and encoding where necessary.
🧠 Models Used
Model	Accuracy (%)
K-Nearest Neighbors	94.75
Random Forest	94.74
Logistic Regression	92.98
Both KNN and Random Forest models achieved the highest accuracy, indicating strong predictive power for this classification problem.

🔍 Hyperparameter Tuning
Employed RandomizedSearchCV to optimize model parameters and improve accuracy.
Tuned parameters such as number of neighbors, weights, and distance metrics for KNN.
Random Forest parameters like number of estimators and max depth were also tuned.

📈 Evaluation Metrics
Accuracy
Confusion Matrix (plotted for better visualization)
Precision, Recall, and F1-score for balanced assessment

📌 Key Highlights
Robust comparison between multiple classifiers
Hyperparameter tuning using RandomizedSearchCV for optimal performance
Final models achieving over 94% accuracy, suitable for early breast cancer diagnosis support

🛠️ Tech Stack
Python (Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn)
Jupyter Notebook

▶️ How to Run
Clone the repository

Install dependencies:
bash
Copy
Edit
pip install -r requirements.txt
Run the Jupyter Notebook to see the full workflow and results.
