import regression_demo  # Assuming the DLL's name matches the Python file

# Call functions from the DLL:
predictions = regression_demo.perform_regression("train.csv", "test.csv")
print('Predictions:', predictions)
