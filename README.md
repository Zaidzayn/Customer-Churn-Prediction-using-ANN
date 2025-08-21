## Customer Churn Prediction using ANN
Predict customer churn using a deep learning approach (Artificial Neural Network) with a user-friendly Streamlit web app.

##🚩 Project Overview
This project addresses the key business challenge of customer churn—identifying which customers are likely to stop using a service. It uses an ANN trained on real-world customer data and provides predictions through an interactive web interface.

🛠 Technologies Used
Python (Jupyter Notebook, .py)
TensorFlow / Keras (model.h5)
scikit-learn (Label Encoder & OneHotEncoder, Model serialization)
pandas
Streamlit (for UI and deployment)
Pickle (for encoder/scaler persistence)
Deployment: Streamlit Cloud

##📂 Repository Structure
├── Churn_Modelling.csv        # Dataset (sample/used for model training)
├── app.py                     # Streamlit app source code
├── experiments.ipynb          # EDA & model development notebook
├── prediction.ipynb           # Batch or individual prediction notebook
├── model.h5                   # Trained ANN model
├── label_encoder_gender.pkl   # Pickled LabelEncoder for 'Gender'
├── one_hot_encoder.pkl        # Pickled OneHotEncoder for other categorical features
├── scaler.pkl                 # Pickled Scaler (e.g., StandardScaler)
├── requirements.txt           # Python dependencies

##🌐 Live Demo
Access the deployed web app:
https://customer-churn-prediction-using-ann-yh2pfvdj2npakqig4bcmfw.streamlit.app/#customer-churn-prediction

Interactive form to input customer features (manually or file upload)
Data preprocessing (encoding, scaling)
ANN-based prediction for churn probability
Clear output: “Churn” or “Not Churn” with probability
(Optional) Visualize SHAP/feature importances and model metrics

##💡 How it Works
A. Data Preprocessing
Gender: Label encoding via label_encoder_gender.pkl
Other Categoricals: One-hot encoding via one_hot_encoder.pkl
Numerics: Scaled using the fitted scaler scaler.pkl
Processed features are fed into the ANN model (model.h5) for prediction

B. Model
3 Fully-connected ANN layers built and trained on the churn dataset
Model performance validated using accuracy/F1-score, etc.

C. User Interface
Input customer parameters through an intuitive sidebar/form
Immediate prediction result and probability display
