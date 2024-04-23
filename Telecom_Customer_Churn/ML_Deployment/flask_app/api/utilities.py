import pandas as pd
import numpy as np
import joblib

# loadning model and transformers
column_names = joblib.load('models/column_names.pkl')
column_transformer = joblib.load('models/column_transformer.pkl')
scaler = joblib.load('models/scaler.pkl')
model = joblib.load('models/final_model.pkl')

# dividing the columns based on feature encoding method that will be applied
columns_label_encoding = ['Married', 'Internet Service', 'Paperless Billing']
columns_feature_scaling = ['Number of Dependents', 'Tenure in Months', 'Monthly Charge']
columns_one_hot_encoding = ['Offer', 'Internet Type', 'Online Security', 'Online Backup', 
                            'Device Protection Plan', 'Premium Tech Support', 'Streaming TV', 
                            'Streaming Movies', 'Streaming Music', 'Unlimited Data', 'Contract', 
                            'Payment Method']

def predict_churn(customer):
    # customer : dictionary
    return predict(model, customer)

# observation = customer
def predict(model, input):

    # transform input (type: dictionary) to Series and then to DataFrame
    customer = pd.Series(input, index=input.keys())
    customer = pd.DataFrame(customer).transpose()

    # reorder columns
    customer=customer[column_names]

    # apply label encoding
    customer[columns_label_encoding] = customer[columns_label_encoding].apply(lambda x: x.map({'Yes': 1, 'No':0}))

    # apply one-hot encoding
    customer = np.array(column_transformer.transform(customer))

    # apply feature scaling on the last 4 columns
    customer[:, -3:] = scaler.transform(customer[:, -3:])

    # make prediction using the model
    prediction = model.predict(customer)

    if prediction == 1:
        output = "Customer is predicted to churn."
    else:
        output = "Customer is predicted to stay."

    return output


if __name__ == "__main__":
    # reading input
    input = {"Gender": "Male", 'Age': 50, 'Married': "No", 'Number of Dependents': 0,
        'Number of Referrals': 0, 'Tenure in Months': 4, 'Offer': "Offer E", 'Phone Service': "Yes",
        'Avg Monthly Long Distance Charges': 33.65, 'Multiple Lines': "No",
        'Internet Service': "Yes", 'Internet Type': "Fiber Optic", 'Avg Monthly GB Download': 30.0,
        'Online Security': "No", 'Online Backup': "No", 'Device Protection Plan': "Yes",
        'Premium Tech Support': "No", 'Streaming TV': "No", 'Streaming Movies': "No",
        'Streaming Music': "No", 'Unlimited Data': "Yes", 'Contract': "Month-to-Month", 'Paperless Billing': "Yes",
        'Payment Method': "Bank Withdrawal", 'Monthly Charge': 73.9, 'Total Charges':280.85, 'Total Refunds': 0.00,
        'Total Extra Data Charges': 0, 'Total Long Distance Charges': 134.60,
        'Total Revenue': 415.45 }
    prediction = predict_churn(input)
    print(prediction)

