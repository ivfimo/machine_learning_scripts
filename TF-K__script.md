import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, make_scorer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import shap  # SHAP library for explainability

# Upload the TF-K_dataset
data = pd.read_csv("data path")

# Separate features and target variable
X = data.drop(columns=['group'])
y = data['group']

# Convert target to binary labels if not already binary (assuming binary classification)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Verify class distribution
classes, class_counts = np.unique(y, return_counts=True)
print(f"Classes: {classes}, Counts: {class_counts}")

# Calculate class weights dynamically based on class distribution
class_weights = {class_label: max(class_counts) / count for class_label, count in zip(classes, class_counts)}

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a function to create the model
def create_model():
    model = Sequential([
        Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(8, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Wrapper function for cross-validation
def train_and_evaluate(X, y, cv_splits=5):
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    auc_scores = []
    for train_idx, val_idx in skf.split(X, y):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        model = create_model()
        model.fit(X_train_cv, y_train_cv, epochs=60, batch_size=12, verbose=0, class_weight=class_weights)

        # Evaluate the model on validation set
        y_val_pred_prob = model.predict(X_val_cv)
        auc_score = roc_auc_score(y_val_cv, y_val_pred_prob)
        auc_scores.append(auc_score)

    return np.array(auc_scores)

# Perform cross-validation to compute AUC
cv_auc_scores = train_and_evaluate(X_train_scaled, y_train, cv_splits=5)
print(f"Cross-Validation AUC Scores: {cv_auc_scores}")
print(f"Mean Cross-Validation AUC: {np.mean(cv_auc_scores):.4f}")

# Train the model on the full training set
final_model = create_model()
final_model.fit(X_train_scaled, y_train, epochs=60, batch_size=12, verbose=0, class_weight=class_weights)

# Evaluate on the test set
y_pred_prob = final_model.predict(X_test_scaled)
roc_auc = roc_auc_score(y_test, y_pred_prob)
y_pred = (y_pred_prob > 0.5).astype(int)

print("Test Set ROC AUC Score:", roc_auc)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- SHAP value calculation and visualization ---
# Create a SHAP explainer for the TensorFlow-Keras model
explainer = shap.Explainer(final_model, X_train_scaled)

# Calculate SHAP values for the test set
shap_values = explainer(X_test_scaled)

# Generate a SHAP summary plot
shap.summary_plot(shap_values, features=X_test, feature_names=X.columns, show=True)

