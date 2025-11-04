🏦 Loan Approval Prediction using Machine Learning & FastAPI

✅ Project Overview

This project predicts whether a loan application should be approved or rejected based on applicant financial information using machine learning algorithms. After training multiple models and evaluating their performance, the best model with 98% accuracy was saved using Pickle and deployed using FastAPI for real-time API-based predictions.


---

📚 Technologies Used

Language: Python
Libraries:

Data Processing – pandas, numpy

Visualization – matplotlib, seaborn

Model Building – scikit-learn

Model Saving – pickle

Deployment – FastAPI, uvicorn



---

🤖 Machine Learning Models Applied

Logistic Regression

Decision Tree

Random Forest

Support Vector Classifier (SVC)

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes

Voting Classifier

Bagging Classifier

Stacking Classifier


Among these, the best performing model was selected and saved as loan.pkl.


---

📊 Project Workflow

1. Data Loading & Cleaning: Handled missing values, encoded categorical features, and removed duplicates.


2. Exploratory Data Analysis (EDA): Visualized feature relationships using heatmaps, histograms, boxplots, etc.


3. Model Training: Trained different classification algorithms and ensemble techniques.


4. Evaluation: Used Accuracy, Confusion Matrix, and Classification Report to compare models.


5. Model Saving:

pickle.dump(best_model, open('loan.pkl', 'wb'))


6. FastAPI Deployment: Loaded pickle model and created API endpoints to accept user input and return loan approval results.




---

🖥 FastAPI Deployment (loan.py) – Summary

/ → Home route: Displays project welcome message.

/predict → POST API: Accepts input such as gender, income, loan amount, education, employment status, credit history, asset values, etc.

Loads trained model using:

classifier = pickle.load(open("loan.pkl", "rb"))

Returns response:

"Approved" if prediction is 1

"Rejected" if prediction is 0



Run the API using:

uvicorn loan:app --reload

Access the API documentation at:
http://127.0.0.1:8000/docs (Swagger UI)


---

📁 Project Structure

├── loan.py                  # FastAPI deployment file  
├── loan.pkl                 # Saved ML model  
├── loan_approval_dataset.csv              # Dataset used  
├── loanApproval.ipynb      # Training & evaluation code  
├── requirements.txt  
├── README.md


---

🚀 Future Enhancements

✔ Add Streamlit/HTML frontend
✔ Connect database to store user requests
✔ Deploy on Render / Railway / AWS
✔ Add hyperparameter tuning for improvement


---

📌 Conclusion

This project successfully combines Machine Learning and FastAPI to build a real-time Loan Approval Prediction System. With 98% accuracy and API integration, it can be used in financial services for faster and automated loan decisions.

