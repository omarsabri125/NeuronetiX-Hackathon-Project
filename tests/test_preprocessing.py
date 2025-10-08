import os
from src.data.preprocessing import preprocessing_pipline

def test_preprocessing_output_shape():
    """Ensure preprocessing returns valid data shapes."""

    FILE_PATH = "assets/Telecom_Customers_Churn_Dataset.csv"

    assert os.path.exists(FILE_PATH), f"❌ Dataset not found at {FILE_PATH}"

    X_train, X_test, y_train, y_test = preprocessing_pipline(FILE_PATH)
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "❌ Data is empty after preprocessing!"
