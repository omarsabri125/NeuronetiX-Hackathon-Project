import pandas as pd
import time
from ..utils.metrics import (PREDICTION_COUNT, PREDICTION_LATENCY, CHURN_PROBABILITY,
                            PREDICTION_ERRORS, FEATURE_TENURE, FEATURE_MONTHLY_CHARGES, FEATURE_TOTAL_CHARGES)

def inference(model, scalar, label_encoder, predict_request):
    start_time = time.time()
    
    try:
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
        
        # Record feature values using Gauges
        FEATURE_TENURE.set(tenure)
        FEATURE_MONTHLY_CHARGES.set(monthly_charges)
        FEATURE_TOTAL_CHARGES.set(total_charges)
        
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
        
        # Record prediction count
        PREDICTION_COUNT.labels(prediction=churn_prediction).inc()
        
        if hasattr(model, "predict_proba"):
            prediction_proba = model.predict_proba(features_df)
            churn_probability = float(prediction_proba[0][1])  # Probability of class '1' (churn)
            
            # Record churn probability distribution
            CHURN_PROBABILITY.observe(churn_probability)
        else:
            churn_probability = None
        
        # Record prediction latency
        duration = time.time() - start_time
        PREDICTION_LATENCY.observe(duration)
        
        return churn_prediction, churn_probability
        
    except KeyError as e:
        PREDICTION_ERRORS.labels(error_type='encoding_error').inc()
        raise
    except Exception as e:
        PREDICTION_ERRORS.labels(error_type='general_error').inc()
        raise