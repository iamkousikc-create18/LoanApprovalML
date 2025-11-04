import uvicorn
from fastapi import FastAPI
import pickle
app=FastAPI()
classifier=pickle.load(open("loan.pkl","rb"))

@app.get('/')
def index():
    return {"Deployment":"Welcome To My Loan Approval Machine Learning Project"}
@app.post('/predict')
def predict(no_of_dependents:int,education:int,self_employed:int,income_annum:int,loan_amount:int,loan_term:int,cibil_score:int,residential_assets_value:int,commercial_assets_value:int,luxury_assets_value:int,bank_asset_value:int):
    prediction=classifier.predict([[no_of_dependents,education,self_employed,income_annum,loan_amount,loan_term,cibil_score,residential_assets_value,commercial_assets_value,luxury_assets_value,bank_asset_value]])
    if(prediction[0]==0):
        prediction="Rejected"
    else:
        prediction="Approved"
    return{
        "prediction":prediction
    }
if __name__=="__main__":
    uvicorn.run(app,host='127.0.0.1',port=5000)