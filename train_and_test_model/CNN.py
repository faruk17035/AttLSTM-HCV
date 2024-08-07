import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam


# Load data
file_name = 'word2vec_3mer.csv'
df = pd.read_csv(file_name)
df.head()

# Split data into independent (X) and dependent (Y) columns
y = df['Target'].values
X = df.drop('Target', axis=1).values

df.shape

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape data for CNN
X = X.reshape(X.shape[0], X.shape[1], 1)

# Define 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# Initialize metrics
accuracies = []
roc_aucs = []
precisions = []
recalls = []
sensitivities = []
specificities = []
mccs = []
f1s = []

# Define CNN model
def create_cnn_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=4, activation='relu', input_shape=(512, 1)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Conv1D(filters=64, kernel_size=4, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Perform cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Create and train the model
    model = create_cnn_model()
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_test, y_test), verbose=1)
    
    # Predict and evaluate
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Metrics calculations
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    
    sensitivity = recall  # Sensitivity is the same as recall
    specificity = TN / (TN + FP)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    accuracies.append(acc)
    roc_aucs.append(roc_auc)
    precisions.append(precision)
    recalls.append(recall)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    mccs.append(mcc)
    f1s.append(f1)
    
    print(f'Confusion Matrix:\n{cm}')
    print(f'Accuracy: {acc}')
    print(f'AUC: {roc_auc}')
    print(f'Sensitivity (Recall): {sensitivity}')
    print(f'Specificity: {specificity}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')
    print(f'MCC: {mcc}')
    print('---')
