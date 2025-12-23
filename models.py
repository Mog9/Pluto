import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, r2_score

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsRegressor


def get_model(model_key, task_type):
    if task_type == "Classification":
        model_map = {
            "Logistic Regression": LogisticRegression(max_iter=200),
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC()
        }
    else:
        model_map = {
            "Linear Regression": LinearRegression(),
            "KNN Regressor": KNeighborsRegressor(),
            "Decision Tree Regressor": DecisionTreeRegressor(),
            "Random Forest Regressor": RandomForestRegressor()
        }
    return model_map.get(model_key)




#training models
def train_models(df, target_column, selected_features, selected_models, task_type):
    X = df[selected_features]
    y = df[target_column]

    data = pd.concat([X, y], axis=1)
    data = data.dropna()
    X = data.drop(columns=[target_column])
    y = data[target_column]

    id_cols = [col for col in X.columns if 'id' in col.lower() or 'user' in col.lower()]
    X = X.drop(columns=id_cols, errors='ignore')

    X = X.apply(pd.to_numeric, errors='ignore')

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = []

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    for key in selected_models: 
        model = get_model(key, task_type)
        if model is None:
            continue

        if key in ['SVR', 'KNN Regressor', 'SVM', 'KNN']:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)


        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
   

        if task_type == "Classification":
            metrics = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "f1 score": round(f1_score(y_test, y_pred, average="weighted"), 4)
            }
        else: 
            metrics = {
                "r2 score": round(r2_score(y_test, y_pred), 4)
            }

        results.append({
            "model_key": key,
            "model": model,
            "metrics": metrics,
            "y_true": y_test,
            "y_pred": y_pred
        })

    return results



def get_classification_models():
    return {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC(),
        "KNN": KNeighborsClassifier()
    }

def get_regression_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "KNN Regressor": KNeighborsRegressor()
    }