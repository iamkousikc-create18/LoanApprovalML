🏦 Loan Approval Prediction using Machine Learning & FastAPI

📌 Project Overview

This project predicts whether a loan application should be approved based on applicant details such as income, credit history, employment status, loan amount, assets, and more.
Multiple machine learning algorithms were trained, compared, and the best-performing model (98% accuracy) was saved using Pickle.
The model is deployed using FastAPI for real-time loan approval prediction via API.
During preprocessing, both StandardScaler and MinMaxScaler were applied for feature scaling to improve model accuracy and stability.


---

🛠 Technologies Used

Language: Python
Deployment Framework: FastAPI, Uvicorn
Libraries:

Data Handling – pandas, numpy

Data Visualization – matplotlib, seaborn

Feature Scaling – StandardScaler, MinMaxScaler

Machine Learning – scikit-learn

Model Saving – pickle



---

🤖 Machine Learning Models Used

Logistic Regression

Decision Tree Classifier

Random Forest Classifier

Support Vector Classifier (SVC)

K-Nearest Neighbors (KNN)

Gaussian Naive Bayes

Voting Classifier

Bagging Classifier

Stacking Classifier


The best model was chosen after evaluation and saved as loan.pkl.


---

📊 Workflow

1. Data Collection & Cleaning

Loaded dataset using pandas

Handled missing values, duplicates, and encoded categorical variables



2. Feature Scaling
✅ Applied StandardScaler for models sensitive to feature distribution (SVM, Logistic Regression, KNN)
✅ Applied MinMaxScaler for normalization where values were brought between 0 and 1

from sklearn.preprocessing import StandardScaler, MinMaxScaler
ss = StandardScaler()
X_ss = ss.fit_transform(X)

ms = MinMaxScaler()
X_ms = ms.fit_transform(X)


3. Model Training & Evaluation

Trained multiple classifiers

Evaluated using accuracy, confusion matrix, and classification report



4. Model Saving

import pickle
pickle.dump(best_model, open('loan.pkl', 'wb'))


5. FastAPI Deployment

Loaded saved model using Pickle

Created API routes:

/ → Welcome message

/predict → Takes input and returns “Approved” or “Rejected”


Run server using:

uvicorn loan:app --reload





---

📁 Project Structure

├── loan.py               # FastAPI deployment  
├── loan.pkl              # Saved ML model  
├── loanApproval.ipynb   # Model training notebook  
├── loan_approval_dataset.csv  
├── requirements.txt  
├── README.md


---

🚀 Future Enhancements

✔ Deploy on Render/Railway/AWS
✔ Add Streamlit/Flask frontend
✔ Use database to store user requests
✔ Apply hyperparameter tuning using GridSearchCV


---

✅ Conclusion

This project demonstrates a complete pipeline from data preprocessing (StandardScaler + MinMaxScaler), ML model training, evaluation, saving with pickle, and FastAPI deployment. With 98% accuracy and API integration, it provides an efficient and scalable loan approval prediction system.
---

✍ Author

👤 Kousik Chakraborty
📧 Email: www.kousik.c.in@gmail.com
🔗 GitHub Profile: https://github.com/iamkousikc-create18
🔗 Project Repository: https://github.com/iamkousikc-create18/LoanApprovalML
