import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_curve, roc_auc_score

MODEL_FILENAME = 'loan_model_pipeline.pkl'

def generate_sample_data(n_samples=2000):
    """Generates a realistic synthetic loan applicant dataset with class imbalance."""
    np.random.seed(42)
    
    age = np.random.randint(21, 70, size=n_samples)
    income = np.random.exponential(scale=60000, size=n_samples) + 25000
    loan_amount = np.random.exponential(scale=18000, size=n_samples) + 5000
    
    employment_types = np.random.choice(['Salaried', 'Self-Employed', 'Unemployed', 'Contract'], size=n_samples, p=[0.65, 0.20, 0.05, 0.10])
    credit_tier = np.random.choice(['Good', 'Fair', 'Poor'], size=n_samples, p=[0.55, 0.30, 0.15])
    
    # Introduce missing values to test imputers
    income[np.random.choice(n_samples, 30, replace=False)] = np.nan
    employment_types[np.random.choice(n_samples, 20, replace=False)] = None
    
    # Imbalanced target rule (approx 15% default rate)
    dti = loan_amount / (income + 1e-5)
    default_prob = 1 / (1 + np.exp(-((dti * 3) - (credit_tier == 'Poor').astype(int) * 2 - 1.5)))
    default = (np.random.rand(n_samples) < default_prob).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Income': income,
        'LoanAmount': loan_amount,
        'EmploymentType': employment_types,
        'CreditTier': credit_tier,
        'Default': default
    })
    
    # Engineer domain features upfront on the DataFrame
    df['DebtToIncome'] = df['LoanAmount'] / (df['Income'] + 1e-5)
    df['LoanToAge'] = df['LoanAmount'] / (df['Age'] + 1e-5)
    
    return df

def train_and_optimize_model(X, y):
    """Builds pipeline, runs GridSearchCV, and computes optimal classification threshold."""
    print("Setting up preprocessing and modeling pipeline...")
    
    numeric_features = ['Age', 'Income', 'LoanAmount', 'DebtToIncome', 'LoanToAge']
    categorical_features = ['EmploymentType', 'CreditTier']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median'))
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
    ])

    param_grid = {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [6, 10],
        'classifier__min_samples_split': [5, 10]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    print("Running GridSearchCV with 5-Fold Stratified Cross-Validation...")
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=cv, 
        scoring='roc_auc', 
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    print(f"Best Cross-Validation ROC-AUC: {grid_search.best_score_:.4f}")
    print(f"Best Hyperparameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_

    print("Calculating optimal decision threshold...")
    y_probs = best_model.predict_proba(X)[:, 1]
    fpr, tpr, thresholds = roc_curve(y, y_probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal Decision Threshold determined: {optimal_threshold:.4f}")

    return best_model, optimal_threshold

def prepare_applicant_features(df):
    """Helper to ensure new inference inputs contain the engineered features."""
    df_out = df.copy()
    df_out['DebtToIncome'] = df_out['LoanAmount'] / (df_out['Income'] + 1e-5)
    df_out['LoanToAge'] = df_out['LoanAmount'] / (df_out['Age'] + 1e-5)
    return df_out

def main():
    print("--- Advanced Loan Default Risk Model ---")
    
    if os.path.exists(MODEL_FILENAME):
        print(f"Found saved model '{MODEL_FILENAME}'. Loading from disk...")
        try:
            saved_data = joblib.load(MODEL_FILENAME)
            pipeline = saved_data['model']
            optimal_threshold = saved_data['threshold']
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading saved model: {e}. Retraining...", file=sys.stderr)
            df = generate_sample_data(2000)
            X, y = df.drop(columns=['Default']), df['Default']
            pipeline, optimal_threshold = train_and_optimize_model(X, y)
            joblib.dump({'model': pipeline, 'threshold': optimal_threshold}, MODEL_FILENAME)
    else:
        print("Generating training dataset...")
        df = generate_sample_data(2000)
        X = df.drop(columns=['Default'])
        y = df['Default']
        
        try:
            pipeline, optimal_threshold = train_and_optimize_model(X, y)
            joblib.dump({'model': pipeline, 'threshold': optimal_threshold}, MODEL_FILENAME)
            print(f"Model saved to '{MODEL_FILENAME}'.")
        except Exception as e:
            print(f"Error during training and persistence: {e}", file=sys.stderr)
            sys.exit(1)

    print("\n--- Interactive Applicant Assessment ---")
    print("Enter applicant details to predict default risk:")

    try:
        age_input = float(input("Age (e.g., 35): "))
        income_input = float(input("Annual Income in INR (e.g., 750000): "))
        loan_input = float(input("Loan Amount in INR (e.g., 300000): "))
        emp_input = input("Employment Type (Salaried / Self-Employed / Unemployed / Contract): ").strip()
        credit_input = input("Credit Tier (Good / Fair / Poor): ").strip()

        new_applicant = pd.DataFrame([{
            'Age': age_input,
            'Income': income_input,
            'LoanAmount': loan_input,
            'EmploymentType': emp_input if emp_input else 'Salaried',
            'CreditTier': credit_input if credit_input else 'Good'
        }])

        # Apply the same feature engineering to new input
        new_applicant = prepare_applicant_features(new_applicant)

        pred_prob = pipeline.predict_proba(new_applicant)[0][1]
        is_default = (pred_prob >= optimal_threshold).astype(int)

        print("\n[Prediction Result]")
        print(f"Calculated Default Probability: {pred_prob * 100:.2f}%")
        print(f"Custom Optimal Threshold:       {optimal_threshold * 100:.2f}%")
        print(f"Risk Assessment:                {'High Risk (Default)' if is_default == 1 else 'Low Risk (Non-Default)'}")

    except ValueError as ve:
        print(f"Invalid input format: {ve}", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred during prediction: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()