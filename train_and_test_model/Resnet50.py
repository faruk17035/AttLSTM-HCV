import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
import matplotlib.pyplot as plt


# Load pretraining data
pretrain_path = 'pretrained_data.xlsx'
df_pretrain = pd.read_excel(pretrain_path)

# Extract features and target for pretraining
X_pretrain = df_pretrain.iloc[:, :-1].values
y_pretrain = df_pretrain.iloc[:, -1].values

# Feature scaling for pretraining
scaler_pretrain = StandardScaler()
X_pretrain = scaler_pretrain.fit_transform(X_pretrain)

# Reshape data for 1D convolution for pretraining
X_pretrain = np.expand_dims(X_pretrain, axis=-1)

# Define ResNet50 model
def residual_block(input_layer, filters, kernel_size, dilation_rate):
    """Residual block with dilated convolution."""
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Add()([input_layer, x])  # Skip connection
    return x

def build_resnet50(input_shape, num_classes):
    """Build ResNet50 model for 1D time series data."""
    input_layer = Input(shape=input_shape)

    # Initial convolution
    x = Conv1D(64, 7, padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual blocks
    for _ in range(4):
        x = residual_block(x, filters=64, kernel_size=3, dilation_rate=1)
        x = residual_block(x, filters=64, kernel_size=3, dilation_rate=2)
        x = residual_block(x, filters=64, kernel_size=3, dilation_rate=4)
        x = residual_block(x, filters=64, kernel_size=3, dilation_rate=8)

    # Global average pooling and output
    x = GlobalAveragePooling1D()(x)
    output_layer = Dense(num_classes, activation='sigmoid')(x)  # Use sigmoid activation for binary classification

    model = Model(inputs=input_layer, outputs=output_layer)
    return model

# Pretrain the model
input_shape_pretrain = X_pretrain.shape[1:]
num_classes = 1
model = build_resnet50(input_shape_pretrain, num_classes)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_pretrain, y_pretrain, batch_size=64, epochs=10, validation_split=0.1)

# Save the pretrained model
model.save('pretrained_model.h5')

# Load new data for fine-tuning
finetune_path = 'word2vec_3mer.xlsx'
df_finetune = pd.read_excel(finetune_path)

# Extract features and target for fine-tuning
X_finetune = df_finetune.iloc[:, :-1].values
y_finetune = df_finetune.iloc[:, -1].values

# Feature scaling for fine-tuning
scaler_finetune = StandardScaler()
X_finetune = scaler_finetune.fit_transform(X_finetune)

# Reshape data for 1D convolution for fine-tuning
X_finetune = np.expand_dims(X_finetune, axis=-1)

# Load the pretrained model
model = load_model('pretrained_model.h5')

# Adjust input shape for new data
input_shape_finetune = X_finetune.shape[1:]
model.layers[0].batch_input_shape = (None,) + input_shape_finetune
model.build(input_shape_finetune)

# Compile the model again
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

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

# Perform cross-validation
for train_index, test_index in skf.split(X_finetune, y_finetune):
    X_train, X_test = X_finetune[train_index], X_finetune[test_index]
    y_train, y_test = y_finetune[train_index], y_finetune[test_index]

    # Train the model on the new dataset
    model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_test, y_test), verbose=1)
    
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
    print(f'True Positives: {TP}')
    print(f'True Negatives: {TN}')
    print(f'False Positives: {FP}')
    print(f'False Negatives: {FN}')
    print('---')