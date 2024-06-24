#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd

file_path = '/Users/vishaldutt/Downloads/fdata.csv'
data = pd.read_csv(file_path)
initial_data_overview = data.head()

missing_values_before = data.isnull().sum()

numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())

categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

missing_values_after = data.isnull().sum()


initial_data_overview, missing_values_before, missing_values_after


# In[79]:


import matplotlib.pyplot as plt
import seaborn as sns

original_data = pd.read_csv(file_path)

missing_before = original_data.isnull()
missing_after = data.isnull()
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.heatmap(missing_before, cmap="viridis", yticklabels=False, cbar=False, ax=axes[0])
axes[0].set_title('Missing Data Before Imputation')
sns.heatmap(missing_after, cmap="viridis", yticklabels=False, cbar=False, ax=axes[1])
axes[1].set_title('Missing Data After Imputation')

plt.show()


# In[80]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

z_threshold = 3
Q1 = data[numerical_cols].quantile(0.25)
Q3 = data[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
iqr_threshold = 1.5
mean_values = data[numerical_cols].mean()
median_values = data[numerical_cols].median()

z_scores = np.abs(stats.zscore(data[numerical_cols]))
outliers_z = (z_scores > z_threshold)

outliers_iqr = ((data[numerical_cols] < (Q1 - iqr_threshold * IQR)) | (data[numerical_cols] > (Q3 + iqr_threshold * IQR)))

data_clean_z = data[(z_scores < z_threshold).all(axis=1)]

data_clean_iqr = data[~outliers_iqr.any(axis=1)]

fig, axes = plt.subplots(2, len(numerical_cols), figsize=(20, 10))
for i, col in enumerate(numerical_cols):
    sns.boxplot(y=data[col], ax=axes[0, i])
    axes[0, i].set_title(f'Before Z-Score: {col}')
    sns.boxplot(y=data_clean_z[col], ax=axes[1, i])
    axes[1, i].set_title(f'After Z-Score: {col}')

plt.tight_layout()

outliers_z_count = (z_scores > z_threshold).sum(axis=0)


outliers_iqr_count = outliers_iqr.sum(axis=0)

outliers_detected = pd.DataFrame({
    'Z-Score Outliers': outliers_z_count,
    'IQR Outliers': outliers_iqr_count
})

outliers_detected


# In[81]:


data.head()


# In[82]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data_clean_iqr[numerical_cols])
data_normalized_df = pd.DataFrame(data_normalized, columns=numerical_cols)
for col in data_clean_iqr.columns.difference(numerical_cols):
    data_normalized_df[col] = data_clean_iqr[col].values

data_normalized_df = data_normalized_df[data_clean_iqr.columns]

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
sns.boxplot(data=data_clean_iqr[numerical_cols], ax=axes[0], palette="Set1")
axes[0].set_title('Before Normalization')

sns.boxplot(data=data_normalized_df[numerical_cols], ax=axes[1], palette="Set2")
axes[1].set_title('After Normalization')
plt.show()

normalized_data_head = data_normalized_df.head()
normalized_data_head


# In[83]:


data.head()


# In[84]:


numeric_data_for_corr = data_normalized_df.select_dtypes(include=[np.number])


correlation_matrix = numeric_data_for_corr.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Matrix Heatmap")
plt.show()
upper_corr_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
correlation_sorted = upper_corr_matrix.unstack().sort_values(kind="quicksort").dropna()

correlation_sorted


# In[85]:


class_distribution = data['status'].value_counts(normalize=True)

plt.figure(figsize=(8, 6))
class_distribution.plot(kind='bar')
plt.title('Class Distribution')
plt.xlabel('Status')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.show()

class_distribution



# In[86]:


data.head(15)


# In[ ]:





# In[87]:


from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse=False)
gender_encoded = encoder.fit_transform(data[['Gender']])

encoded_df = pd.DataFrame(gender_encoded, columns=encoder.get_feature_names_out(['Gender']))
data_encoded = pd.concat([data.drop('Gender', axis=1), encoded_df], axis=1)
X_encoded = data_encoded.drop('status', axis=1)  # Features
y_encoded = data_encoded['status']  # Target variable

from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X_encoded, y_encoded)
new_class_distribution = pd.Series(y_resampled).value_counts(normalize=True)
plt.figure(figsize=(8, 6))
new_class_distribution.plot(kind='bar')
plt.title('Class Distribution After Applying SMOTE')
plt.xlabel('Status')
plt.ylabel('Proportion')
plt.xticks(rotation=0)
plt.show()

new_class_distribution




# In[88]:


def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)


marks_columns = ['1st', '2nd', '3rd', '4th', '5th']
marks_data = data[marks_columns].values


sequence_length = 3

sequences = create_sequences(marks_data, sequence_length)

X = sequences[:, :-1, :]  
y = sequences[:, -1, :]  

X.shape, y.shape



# In[89]:





# In[90]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

model = Sequential()

input_shape = (X.shape[1], X.shape[2])

model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=50, return_sequences=False))


model.add(Dropout(0.2))

model.add(Dense(units=50, activation='relu'))

model.add(Dense(units=5))

model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()
data.head(20)


# In[91]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
input_shape = (X.shape[1], X.shape[2])

model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
model.add(LSTM(units=50, return_sequences=False))

model.add(Dropout(0.2))

model.add(Dense(units=50, activation='relu'))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(
    optimizer=Adam(learning_rate=0.001),  
    loss='binary_crossentropy',  
    metrics=['accuracy'] 
)

model.summary()


# In[92]:


y_aligned = data['status'][sequence_length - 1:].values

X_train, X_val, y_train, y_val = train_test_split(X, y_aligned, test_size=0.2, random_state=42)

print(f'X_train samples: {X_train.shape[0]}, y_train samples: {y_train.shape[0]}')
print(f'X_val samples: {X_val.shape[0]}, y_val samples: {y_val.shape[0]}')


# In[95]:


from tensorflow.keras.callbacks import EarlyStopping

batch_size = 32  
epochs = 100    
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_val: {X_val.shape}")
print(f"Shape of y_val: {y_val.shape}")

y_train = y_train.reshape(-1, 1)
y_val = y_val.reshape(-1, 1)

print(f"After reshape, shape of y_train: {y_train.shape}")
print(f"After reshape, shape of y_val: {y_val.shape}")

early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)


from tensorflow.keras.backend import clear_session
clear_session()
model.compile(
    optimizer=Adam(learning_rate=0.001),  
    loss='binary_crossentropy',           
    metrics=['accuracy']                  
)

model.summary()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[early_stopping_monitor],
    verbose=1
)

model.save('LSTM_Model.h5')

get_ipython().system('pwd')

#!ls


# In[96]:


import matplotlib.pyplot as plt

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()


# In[97]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error
from tensorflow.keras.utils import to_categorical

y_pred_probs = model.predict(X_val)
y_pred = (y_pred_probs > 0.5).astype('int32') 

accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Mean Absolute Error: {mae:.4f}')


# In[98]:


import matplotlib.pyplot as plt

accuracy = 0.8537
precision = 1.0000
recall = 0.2500
f1 = 0.4000
mae = 0.1463

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'Mean Absolute Error']
values = [accuracy, precision, recall, f1, mae]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color=['blue', 'green', 'red', 'purple', 'orange'])
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.title('Model Evaluation Metrics')
plt.ylim(0, 1)  

for i in range(len(values)):
    plt.text(i, values[i] + 0.02, f'{values[i]:.2f}', ha='center')

plt.show()



# In[99]:


import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[tensorboard_callback],
    verbose=1
)


# In[101]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model

file_path_test = '/Users/vishaldutt/Downloads/fdata.csv'
test_data = pd.read_csv(file_path_test)

initial_data_overview = data.head()

missing_values_before = data.isnull().sum()

numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())
data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].median())


categorical_cols = data.select_dtypes(include=['object']).columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
missing_values_after = data.isnull().sum()


initial_data_overview, missing_values_before, missing_values_after

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

z_threshold = 3

Q1 = data[numerical_cols].quantile(0.25)
Q3 = data[numerical_cols].quantile(0.75)
IQR = Q3 - Q1
iqr_threshold = 1.5

mean_values = data[numerical_cols].mean()
median_values = data[numerical_cols].median()

z_scores = np.abs(stats.zscore(data[numerical_cols]))
outliers_z = (z_scores > z_threshold)

outliers_iqr = ((data[numerical_cols] < (Q1 - iqr_threshold * IQR)) | (data[numerical_cols] > (Q3 + iqr_threshold * IQR)))
data_clean_z = data[(z_scores < z_threshold).all(axis=1)]

data_clean_iqr = data[~outliers_iqr.any(axis=1)]

#***********


sequence_length = 3  
marks_columns = ['1st', '2nd', '3rd', '4th', '5th']

def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequence = data[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

test_sequences = create_sequences(test_data[marks_columns].values, sequence_length)
X_test = test_sequences[:, :-1, :]
model = load_model('/Users/vishaldutt/LSTM_Model.h5')  
y_test_pred_probs = model.predict(X_test)
y_test_pred = (y_test_pred_probs > 0.5).astype(int) 

suspicious_records = np.where(y_test_pred == 1)[0]

print("Indices of suspicious records in the test set:", suspicious_records)

#print("Suspicious records data:", test_data.iloc[suspicious_records])


# In[ ]:




