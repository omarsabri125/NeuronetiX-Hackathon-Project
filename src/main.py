import mlflow.sklearn

model_name = "forest_best"
model_version = "1"

# Load the model from the Model Registry
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.sklearn.load_model(model_uri)

print("model loaded")
