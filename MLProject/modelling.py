import argparse
import pandas as pd
import mlflow, mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="ForestFireArea_preprocessing/train_processed.csv")
    parser.add_argument("--test_data", type=str, default="ForestFireArea_preprocessing/test_processed.csv")
    args = parser.parse_args()

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)
    X_train = train_df.drop("area", axis=1); y_train = train_df["area"]
    X_test = test_df.drop("area", axis=1); y_test = test_df["area"]

    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="CI_RF_Basic"):
        model = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        print("R2 test:", model.score(X_test, y_test))

if __name__ == "__main__":
    main()
