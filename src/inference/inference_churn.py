import pandas as pd

def inference(model, scalar, label_encoder, predict_request):

    # Extract features from the request
        senior_citizen = predict_request.SeniorCitizen
        partner = predict_request.Partner
        dependents = predict_request.Dependents
        tenure = predict_request.tenure
        internet_service = predict_request.InternetService
        online_security = predict_request.OnlineSecurity
        online_backup = predict_request.OnlineBackup
        device_protection = predict_request.DeviceProtection
        tech_support = predict_request.TechSupport
        contract = predict_request.Contract
        paperless_billing = predict_request.PaperlessBilling
        payment_method = predict_request.PaymentMethod
        monthly_charges = predict_request.MonthlyCharges
        total_charges = predict_request.TotalCharges

    # Encode categorical variables
        partner = label_encoder['Partner'].transform([partner])[0]
        dependents = label_encoder['Dependents'].transform([dependents])[0]
        internet_service = label_encoder['InternetService'].transform([internet_service])[0]
        online_security = label_encoder['OnlineSecurity'].transform([online_security])[0]
        online_backup = label_encoder['OnlineBackup'].transform([online_backup])[0]
        device_protection = label_encoder['DeviceProtection'].transform([device_protection])[0]
        tech_support = label_encoder['TechSupport'].transform([tech_support])[0]
        contract = label_encoder['Contract'].transform([contract])[0]
        paperless_billing = label_encoder['PaperlessBilling'].transform([paperless_billing])[0]
        payment_method = label_encoder['PaymentMethod'].transform([payment_method])[0]

        features_df = pd.DataFrame({
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })
    
    # Scale numeric features
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        features_df[numeric_cols] = scalar.transform(features_df[numeric_cols])

    # Make prediction
        prediction = model.predict(features_df)
        churn_prediction = "Yes" if prediction[0] == 1 else "No"

        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(features_df)
            churn_probability = float(prediction_proba[0][1])  # Probability of class '1' (churn)
        else:
            churn_probability = None

        return churn_prediction, churn_probability