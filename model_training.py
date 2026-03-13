import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif # Using f_classif for numerical features
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib # For saving/loading models
import numpy as np

# --- Configuration ---
DATA_PATH = 'C:\\Users\\tulas\\Downloads\\heart+failure+clinical+records.zip' # Replace with your dataset path
MODEL_SAVE_PATH = 'heart_failure_model.joblib'
PREPROCESSOR_SAVE_PATH = 'preprocessor.joblib'
SELECTED_FEATURES_SAVE_PATH = 'selected_features.joblib'
NUM_FEATURES_TO_SELECT = 7 # As per the paper's best results

def train_model():
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {DATA_PATH}. Please download it from Kaggle.")
        return

    X = df.drop('DEATH_EVENT', axis=1)
    y = df['DEATH_EVENT']

    # Identify categorical and numerical features
    categorical_features = ['anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
    numerical_features = ['age', 'creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium', 'time']

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Create Preprocessor
    # This preprocessor will scale numerical features and one-hot encode categorical features.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='passthrough' # Keep any other columns (though there shouldn't be any here)
    )

    # 4. Apply Preprocessing to Training and Test Data
    # X_train_processed and X_test_processed will now be explicitly defined.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # 5. Apply SMOTE on the processed training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

    # 6. Feature Selection (using f_classif which handles negative values)
    selector = SelectKBest(f_classif, k=NUM_FEATURES_TO_SELECT)
    X_train_selected = selector.fit_transform(X_train_smote, y_train_smote)
    X_test_selected = selector.transform(X_test_processed)

    # Store the indices of the selected features for later use
    selected_feature_indices = selector.get_support(indices=True)
    joblib.dump(selected_feature_indices, SELECTED_FEATURES_SAVE_PATH)
    
    # 7. Model Training (GBM)
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train_selected, y_train_smote)

    # 8. Model Evaluation
    y_pred = model.predict(X_test_selected)
    y_proba = model.predict_proba(X_test_selected)[:, 1]

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # 9. Save Model and Preprocessor
    joblib.dump(model, MODEL_SAVE_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)
    print("Model, preprocessor, and selected features saved successfully.")

if __name__ == "__main__":
    train_model()