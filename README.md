# titanic-survival-predictor
End-to-end machine learning pipeline that predicts Titanic passenger survival.

Live API url: https://titanic-survival-predictor-zfob.onrender.com  
Note: Requests made after inactivity may take an additional minute due to the API being deployed using the free tier of Render.

Languages: Python
Libraries: sqlite3, scikit-learn, pandas, Flask, joblib


**Tutorials**

**Make prediction using API**
Send a POST http request to the predict endpoint with a JSON body for the passenger.

Example:
```
{
  "Pclass": 3,
  "Sex": "male",
  "Age": 22,
  "SibSp": 1,
  "Parch": 0,
  "Fare": 7.25,
  "Embarked": "S"
}
```

Example command line (windows):  
curl -X POST https://titanic-survival-predictor-zfob.onrender.com/predict -H "Content-Type: application/json" -d "{\"Pclass\": 3, \"Sex\": \"male\", \"Age\": 22, \"SibSp\": 1, \"Parch\": 0, \"Fare\": 7.25, \"Embarked\": \"S\"}"

**Retrain model using API**
SEND a POST http request to the retrain endpoint with a csv file with columns: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked, Survived

Example command line (windows):
curl -X POST https://titanic-survival-predictor-zfob.onrender.com/retrain -H "Content-Type: text/csv" --data-binary @new_data.csv

Example data file

new_data.csv:
```
Pclass,Sex,Age,SibSp,Parch,Fare,Embarked,Survived
1,female,29,0,0,211.34,S,1
3,male,25,0,0,7.05,S,0
2,female,32,1,0,26.0,C,1
3,male,18,0,0,8.05,S,0
1,male,45,0,0,35.5,S,0
2,female,28,1,2,23.45,Q,1
3,male,40,0,0,7.25,S,0
1,female,55,0,1,59.4,C,1
3,male,22,1,0,7.25,S,0
2,male,35,0,0,13.0,S,0
```

**Run pipeline locally**
1. Clone repo  
  git clone https://github.com/Fredward617/titanic-survival-predictor.git
  cd titanic-survival-predictor

3. Install dependencies
   pip install -r requirements.txt

4. Train model
   python model_trainer.py data-sets/titanic.csv --split

5. Start API
   python app.py

6. In a different terminal make predictions and retrain model as previously shown using "http://localhost:5000" for the url
