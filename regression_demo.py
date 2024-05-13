import os
from datetime import datetime
import pandas as pd
from sklearn.linear_model import LinearRegression

def create_folder_with_timestamp():
    now = datetime.now()
    folder_name = now.strftime("predict_%m%d-%H%M")
    return folder_name

def perform_regression(train_file, test_file):
    print('load data...')
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)

    X_train = train_data.drop("target", axis=1)
    y_train = train_data["target"]
    X_test = test_data

    print('train model')
    model = LinearRegression()
    model.fit(X_train, y_train)
    print('predict test data')
    predictions = model.predict(X_test)

    return predictions

def main():
    folder_name = create_folder_with_timestamp()
    current_directory = os.getcwd() 
    predicted_directory = os.path.join(current_directory, folder_name)

    if not os.path.exists(predicted_directory):
        os.mkdir(predicted_directory)  # predictedディレクトリがない場合は作成

    train_file = os.path.join(current_directory, "train.csv")
    test_file = os.path.join(current_directory, "test.csv")
    output_file = os.path.join(predicted_directory, "prediction.csv")

    predictions = perform_regression(train_file, test_file)
    print('save predictions...')
    pd.DataFrame(predictions, columns=["prediction"]).to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
