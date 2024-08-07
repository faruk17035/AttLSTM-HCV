import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, matthews_corrcoef, confusion_matrix


# Load data
file_name = 'word2vec_3mer.csv'
df = pd.read_csv(file_name)
df.head()

y = df['Target'].values
X = df.drop('Target', axis=1).values

df.shape

# Feature scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Define classifier
classifier = SVC(C = 1.0, kernel = 'rbf', gamma = 'scale', probability=True)

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

# Perform cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    start = datetime.now()
    classifier.fit(X_train, y_train)
    stop = datetime.now()
    execution_time_svm = stop - start
    print("SVM execution time is: ", execution_time_svm)
    
    # Predict and evaluate
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]
    
    # Metrics calculations
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[0][1]
    FN = cm[1][0]
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    accuracies.append(acc)
    roc_aucs.append(roc_auc)
    precisions.append(precision)
    recalls.append(recall)
    sensitivities.append(sensitivity)
    specificities.append(specificity)
    mccs.append(mcc)
    
    print(f'Confusion Matrix:\n{cm}')
    print(f'Accuracy: {acc}')
    print(f'AUC: {roc_auc}')
    print(f'Sensitivity (Recall): {sensitivity}')
    print(f'Specificity: {specificity}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'MCC: {mcc}')
    print('---')
