import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, matthews_corrcoef, precision_score, recall_score, f1_score
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import glorot_uniform
from google.colab import drive


# Load your dataset
df = pd.read_csv('word2vec_3mer.csv')

# Separate features and target variable
X_r = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]  # Target variable

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
X = scaler.fit_transform(X_r)

# Reshape input data for LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Define the LSTM with Attention model
def create_model(input_shape, learning_rate):
    # Encoder LSTM
    inputs = Input(shape=input_shape)
    encoder_lstm_output, state_h, state_c = LSTM(50, return_sequences=True, return_state=True, activation='relu', kernel_initializer=glorot_uniform())(inputs)
    
    # Attention Block
    attention_output = Attention()([encoder_lstm_output, encoder_lstm_output])
    context_vector = Concatenate(axis=-1)([encoder_lstm_output, attention_output])
    
    # Decoder LSTM
    decoder_lstm_output, _, _ = LSTM(50, return_sequences=False, activation='relu', kernel_initializer=glorot_uniform())(context_vector)
    
    # Fully Connected Layers
    dropout = Dropout(0.1)(decoder_lstm_output)
    fc1 = Dense(50, activation='relu', kernel_initializer=glorot_uniform(), bias_initializer='zeros')(dropout)
    fc2 = Dense(25, activation='relu', kernel_initializer=glorot_uniform(), bias_initializer='zeros')(fc1)
    dense_output = Dense(1, activation='sigmoid', kernel_initializer=glorot_uniform(), bias_initializer='zeros')(fc2)
    
    model = Model(inputs, dense_output)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Hyperparameters
learning_rate = 0.0001
batch_size = 32
epochs = 50

# Define 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize metrics
results = []

# Perform cross-validation
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Create and compile the model
    model = create_model((1, X_train.shape[2]), learning_rate)
    
    # Fit the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)
    
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
    
    results.append({
        'accuracy': acc,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'mcc': mcc,
        'f1': f1
    })
    
    print(f'Confusion Matrix:\n{cm}')
    print(f'Accuracy: {acc}')
    print(f'AUC: {roc_auc}')
    print(f'Sensitivity (Recall): {sensitivity}')
    print(f'Specificity: {specificity}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1-Score: {f1}')
    print(f'MCC: {mcc}')
    print(f'True Positives: {TP}')
    print(f'True Negatives: {TN}')
    print(f'False Positives: {FP}')
    print(f'False Negatives: {FN}')
    print('---')
