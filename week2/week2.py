import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


import sys
import os
path_to_src = os.path.abspath(os.path.join("c:\\Users\\user\\OneDrive\\Desktop\\see\\tenx", 'week2', 'src'))
sys.path.append(path_to_src)

try:
    from load_data import load_data_from_postgres, insert_data_into_db
    print("Modules imported successfully")
except ModuleNotFoundError as e:
    print(f"ModuleNotFoundError: {e}")



query = "SELECT * FROM xdr_data;"  


df = load_data_from_postgres(query)


if df is not None:
    print("Successfully loaded the data")
else:
    print("Failed to load data.")

sessions_frequency = df.groupby('MSISDN/Number').size()
total_session_duration = df.groupby('MSISDN/Number')['Dur. (ms)'].sum()
df['total_traffic'] = df['Total UL (Bytes)'] + df['Total DL (Bytes)']
total_traffic = df.groupby('MSISDN/Number')['total_traffic'].sum()


engagement_metrics=pd.DataFrame({
    "sessions_frequency": sessions_frequency,
    "total_session_duration":total_session_duration,
    "total_traffic": total_traffic
})
engagement_metrics['satisfaction_score'] = [0] * len(engagement_metrics)



mlflow.sklearn.autolog()
n_estimators = 100
random_state = 42

X = engagement_metrics[['sessions_frequency', 'total_session_duration', 'total_traffic']]
y = engagement_metrics['satisfaction_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print("Tracking URI:", mlflow.get_tracking_uri())