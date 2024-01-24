from flask import Flask, render_template, request,jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

app = Flask(__name__)

# Load the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset (1).csv')

# Split the data into features (x) and target (y)
x = data.drop("DEATH_EVENT", axis=1)
y = data["DEATH_EVENT"]

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
grid_search.fit(x, y)

# Get the best parameters
best_params = grid_search.best_params_

# Initialize and train the Decision Tree Classifier with the best parameters
dt = DecisionTreeClassifier(**best_params)
dt.fit(x, y)

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        age = float(request.form.get("age"))
        anaemia = int(request.form.get("anaemia"))
        creatinine_phosphokinase = int(request.form.get("creatinine_phosphokinase"))
        diabetes = int(request.form.get("diabetes"))
        ejection_fraction = int(request.form.get("ejection_fraction"))
        platelets = int(request.form.get("platelets"))
        high_blood_pressure = int(request.form.get("high_blood_pressure"))
        serum_creatinine = float(request.form.get("serum_creatinine"))
        serum_sodium = int(request.form.get("serum_sodium"))
        sex = int(request.form.get("sex"))
        smoking = int(request.form.get("smoking"))
        time = int(request.form.get("time"))

        sample_data = [
            [
                age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction, platelets, high_blood_pressure,
                serum_creatinine, serum_sodium, sex, smoking, time
            ]
        ]

        prediction = dt.predict(sample_data)
        prediction_text = "Have a happy life" if prediction[0] == 0 else "Potential risk of death"

        return jsonify({"prediction": prediction_text})

    return render_template("model.html", prediction="")

if __name__ == "__main__":
    app.run(debug=True)
