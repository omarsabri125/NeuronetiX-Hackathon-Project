from src.data.preprocessing import preprocessing_pipline

def test_preprocessing_output_shape():
    """Ensure preprocessing returns valid (non-empty) data splits"""
    FILE_PATH = "D:/Churn_Customer/NeuronetiX-Hackathon-Project/assets/Telecom_Customers _Churn_Dataset.csv"
    
    X_train, X_test, y_train, y_test = preprocessing_pipline(FILE_PATH)
    
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Data is empty after preprocessing!"
