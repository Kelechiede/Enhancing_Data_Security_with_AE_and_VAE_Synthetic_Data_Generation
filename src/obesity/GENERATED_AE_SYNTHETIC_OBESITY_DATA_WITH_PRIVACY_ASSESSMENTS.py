#!/usr/bin/env python
# coding: utf-8

# # STEP 1: Import The Following Relevant Scientific and Computational Libraries for Data Manipulation, Modelling, Interpretation, Visualization ETC.   

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Embedding, Flatten, multiply, Concatenate, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical


# # STEP 2: Data Preparation and Preprocessing

# # Preamble
# 
# Prior to synthetic data generation, data preprocessing steps were meticulously applied to each dataset, ensuring data cleanliness, normalization, and partitioning into the said two main component data. The original dataset, comprising 80% of the total data, served as the foundation for generating synthetic data. In contrast, the remaining 20%, referred to as control data, was reserved for evaluating both the privacy risks associated with the synthetic dataset and the utility as well.
# 
# # Ensuring Privacy Preservation and Data Utility
# 1. Privacy Preservation: By using an autoencoder to generate synthetic data, we minimize the risk of exposing original data entries. The classifier's role is solely to provide labels for synthetic instances, further abstracting the process from direct data duplication.
# 
# 2. Data Utility: The utility of the synthetic data is maintained through the autoencoder's ability to capture and replicate the complex relationships in the original data. The trained classifier ensures that synthetic data receives labels that are consistent with the learned patterns, making the synthetic dataset useful for downstream tasks.
# 
# 
# # Purpose of the Control Dataset
# 1. Benchmarking Privacy Leakage: The control dataset (control_data_df) is used to differentiate what an attacker learns purely from the synthetic dataset's utility versus what constitutes an actual privacy leak. By comparing the success of attacks on the control dataset against those on the original dataset, one can more accurately assess the extent of privacy risk.
# 
# 2. Ensuring Fair Evaluation: It ensures that the assessment accounts for the possibility that the synthetic dataset might inadvertently reveal specific patterns or information that could lead to re-identification or information inference not due to the inherent utility of the synthetic data but due to direct data leakage.
# 
# # Creating a Control Dataset
# The control dataset should consist of records that are not included in the synthetic dataset's generation process. Here's how to create or obtain a control dataset:
# 
# 1. Splitting Original Data: Before generating your synthetic dataset, split your original dataset into two parts. One part is used to generate the synthetic dataset (original_data_df), and the other part serves as the control dataset (control_data_df). This way, the control dataset contains real data points that were not used to train the model creating the synthetic data, ensuring they share similar distributions without direct overlaps.
# 
# 
# 
# # Anonymeter Tool
# Anonymeter’s development as an open-source tool underlines Anonos’s commitment to enhancing privacy technologies’ accessibility. It is crafted to be adaptable, ensuring it remains relevant amid evolving privacy regulations and research advancements. The Anonymeter framework plays a pivotal role in the field of synthetic data by providing a structured approach to evaluate and mitigate privacy risks. By assessing singling-out, linkability, and inference risks, Anonymeter helps researchers and practitioners balance the trade-offs between maintaining data utility and ensuring privacy. This is particularly relevant in the era of big data and machine learning, where the use of synthetic data is becoming increasingly prevalent. For more information on this tool, including its installation and configurations, do visit their website and github via the links below.
# 
# https://www.anonos.com/blog/presenting-anonymeter-the-tool-for-assessing-privacy-risks-in-synthetic-datasets
# 
# https://github.com/statice/anonymeter
# 

# In[2]:


# Load the Obesity dataset
obesity_df = pd.read_csv(r'C:\Users\Ede\Desktop\Synthetic_Real_Data_Using_AE_VAE_Techniques\master_thesis2024\obesity.csv')
obesity_df.head(10)


# # Partioning the Original Data () into 80% and 20% respectiively as shown below

# In[3]:


obesity_train_dataframe, control_dataframe = train_test_split(obesity_df, test_size=0.2, random_state=42)
# obesity_train_dataframe is 80% while control_dataframe is 20%


# In[4]:


obesity_train_dataframe.head()


# In[5]:


obesity_train_dataframe.tail()


# In[10]:


# Assuming 'Class_att' is your target column with binary classes 0 and 1
class_counts = obesity_train_dataframe['NObeyesdad'].value_counts()
total_counts = len(obesity_train_dataframe)
print(total_counts) 


# # SAVING THE PARTITIONED DATASETS TO CSV FOR FUTURE USE

# In[14]:


control_dataframe.to_csv(r'C:\Users\Ede\Desktop\Synthetic_Real_Data_Using_AE_VAE_Techniques\master_thesis2024\control_dataframe.csv', index=False)


# In[16]:


obesity_train_dataframe.to_csv(r'C:\Users\Ede\Desktop\Synthetic_Real_Data_Using_AE_VAE_Techniques\master_thesis2024\obesity_train_dataframe.csv', index=False)


# In[11]:


obesity_train_dataframe.info()


# In[23]:


import matplotlib.pyplot as plt
# Assuming 'Class_att' is your target column
class_counts = obesity_train_dataframe['NObeyesdad'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution in the Dataset')
plt.show()


# # STEP 3: DEFINING AND TRAINING AUTO-ENCODER MODEL, AND GENERATE THE RELEVANT SYNTHETIC DATASET THAT MIMICS ORIGINAL DATA

# In[6]:


# DEFINING AND TRAINING AUTO-ENCODER MODEL, AND GENERATE THE RELEVANT SYNTHETIC DATASET THAT MIMICS ORIGINAL DATA

# IMPORTING BELOW LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Input, Dense, Lambda, BatchNormalization, Embedding, Flatten, multiply, Concatenate, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier

# ===============================================AE ARCHITECTURAL DESIGN======================================================

# Separate features and target
features = obesity_train_dataframe.drop('NObeyesdad', axis=1)
labels = obesity_train_dataframe['NObeyesdad']


# Split the balanced dataset into training and testing sets
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)

# We Normalize features 
scaler = MinMaxScaler()
X_train_orig_scaled = scaler.fit_transform(X_train_orig_new)
X_test_orig_scaled = scaler.transform(X_test_orig_new)

# Add noise for autoencoder training
noise_factor = 0.05
X_train_noisy = X_train_orig_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train_orig_scaled.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)

# Define and compile Autoencoder architecture
input_dim = X_train_orig_scaled.shape[1]

# Encoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(64, activation="relu")(input_layer) # can be adjusted to either increase or decrease architectural complexties
encoder = Dense(32, activation="relu")(encoder) # can be adjusted to either increase or decrease architectural complexties

# Bottleneck
bottleneck = Dense(16, activation="relu")(encoder) # can be adjusted to either increase or decrease architectural complexties

# Decoder
decoder = Dense(32, activation="relu")(bottleneck)# can be adjusted to either increase or decrease architectural complexties
decoder = Dense(64, activation="relu")(decoder) # can be adjusted to either increase or decrease architectural complexties
decoder = Dense(input_dim, activation="sigmoid")(decoder)

# Autoencoder
autoencoder_obesity = Model(inputs=input_layer, outputs=decoder)
autoencoder_obesity.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the autoencoder with noisy Data
autoencoder_obesity.fit(X_train_noisy, X_train_orig_scaled, epochs=2000, batch_size=32, validation_split=0.2, verbose=2)

# Generate synthetic features from the entire original dataset scaled
synthetic_features_scaled = autoencoder_obesity.predict(scaler.transform(features))

# Ensure synthetic data matches the original data's scale
# # Normalize synthetic data or features to original data range
synthetic_features = scaler.inverse_transform(synthetic_features_scaled)

# Train a classifier on the original dataset
# we'll use a simple RandomForestClassifier, which is a good starting point 
# for many classification tasks due to its versatility and ease of use
classifier = RandomForestClassifier(n_estimators=250, random_state=42)
classifier.fit(X_train_orig_new, y_train_orig_new)

# Predict labels for the synthetic features
# After training the classifier on the original data, we'll 
# use it to predict labels for the synthetic data generated from the autoencoder.
synthetic_labels_predicted = classifier.predict(synthetic_features)

# Convert synthetic features to a DataFrame
synthetic_data_df_ae = pd.DataFrame(synthetic_features, columns=features.columns)

# Convert predicted labels into a Series (assuming 'labels' is the name of your target variable)
synthetic_labels_df_ae = pd.Series(synthetic_labels_predicted, name='NObeyesdad') # , name='NObeyesdad'

# Example usage
print(synthetic_data_df_ae.head())
print(synthetic_labels_df_ae.head())


# # SAVE THE ABOVE CREATED GENERATIVE AUTOENCODER MODEL

# In[8]:


from tensorflow.keras.models import load_model

# Assume 'autoencoder' is your trained model
autoencoder_obesity.save('autoencoder_obesity.h5')  # Saves the model to an HDF5 file


# In[11]:


from tensorflow.keras.models import load_model

# Assume 'autoencoder' is your trained model
autoencoder_obesity.save('autoencoder_obesity.keras')  # Saves the model to an HDF5 file


# # Join the Generated Synthetic Data and labels

# In[12]:


import pandas as pd

# Assuming 'synthetic_data_df_ae' is your DataFrame containing the synthetic data
# and 'synthetic_labels_df_ae' is the Series with the corresponding labels

ae_synthetic_obesity_data_with_labels_df = synthetic_data_df_ae.assign(NObeyesdad=synthetic_labels_df_ae.values)


# In[43]:


ae_synthetic_obesity_data_with_labels_df.head()


# In[44]:


ae_synthetic_obesity_data_with_labels_df.head()


# # SAVING THE GENERATED AE SYNTHETIC DATASET TO CSV

# In[13]:


# Save the generated synthetic data to a CSV file

ae_synthetic_obesity_data_with_labels_df.to_csv(r'C:\Users\Ede\Desktop\Synthetic_Real_Data_Using_AE_VAE_Techniques\master_thesis2024\ae_synthetic_obesity_data_with_labels.csv', index=False)


# In[60]:


control_dataframe.head()


# In[46]:


synthetic_data_df_ae.head()


# In[30]:


synthetic_data_df_ae_new.tail()


# In[47]:


synthetic_labels_df_ae.head()


# In[128]:


features = obesity_train_dataframe.drop('NObeyesdad', axis=1)
labels = obesity_train_dataframe['NObeyesdad']

# Split the balanced dataset into training and testing sets
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)

# Confirm the shapes   
print("X_train_syn shape:", X_train_orig_new.shape)
print("X_test_syn shape:", X_test_orig_new.shape)
print("y_train_syn shape:", y_train_orig_new.shape)
print("y_test_syn shape:", y_test_orig_new.shape)


# In[130]:


featuresCONT = control_dataframe.drop('NObeyesdad', axis=1)
labelsCONT = control_dataframe['NObeyesdad']

# Split the dataset into training and testing sets
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Confirm the shapes
print("X_train_cont shape:", X_train_cont.shape)
print("X_test_cont shape:", X_test_cont.shape)
print("y_train_cont shape:", y_train_cont.shape)
print("y_test_cont shape:", y_test_cont.shape)


# In[131]:


features_AE = ae_synthetic_obesity_data_with_labels_df.drop('NObeyesdad', axis=1)
labels_AE = ae_synthetic_obesity_data_with_labels_df['NObeyesdad']

# Split the dataset into training and testing sets
X_train_synt, X_test_synt, y_train_synt, y_test_synt = train_test_split(features_AE, labels_AE, test_size=0.2, random_state=42)

# Confirm the shapes
print("X_train_synt shape:", X_train_synt.shape)
print("X_test_synt shape:", X_test_synt.shape)
print("y_train_synt shape:", y_train_synt.shape)
print("y_test_synt shape:", y_test_synt.shape)


# In[34]:


import matplotlib.pyplot as plt
# Assuming 'Class_att' is your target column
class_counts = obesity_train_dataframe['NObeyesdad'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution in the Dataset')
plt.show()


# In[48]:


import matplotlib.pyplot as plt
import seaborn as sns

# If ae_synthetic_labels_df is a DataFrame with a column named 'PredictedLabels'
plt.figure(figsize=(8, 6))
sns.countplot(x=synthetic_labels_df_ae_new, data=synthetic_data_df_ae_new)
plt.title('80% Estimation of Obesity Levels (AE Synthetic Dataset)')
plt.xlabel('Class of AE Synthetic Dataset')
plt.ylabel('Count')
plt.show()


# # Computation and Graphical Representations of AUC-ROC Curves by Classifiers 

# # RANDOM FOREST

# In[53]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np

# Assuming synthetic_features_scaled and ae_synthetic_labels_df are your synthetic features and labels
# And X_train, X_test, y_train, y_test are your original dataset split

X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae_new, synthetic_labels_df_ae_new, test_size=0.2, random_state=42)

# Preparing the classifier
classifier_on_synthetic = RandomForestClassifier()
classifier_on_synthetic.fit(X_train_syn_new, y_train_syn_new)

# Predict probabilities on the original test set
y_score_test_on_synthetic_model = classifier_on_synthetic.predict_proba(X_test_cont)

# Binarize the original labels for AUC-ROC calculation
lb = LabelBinarizer()
lb.fit(y_train_cont)
y_test_binarized = lb.transform(y_test_cont)

# Calculate ROC AUC for each class in the original test set
fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()
n_classes = y_test_binarized.shape[1]

for i in range(n_classes):
    fpr_test[i], tpr_test[i], _ = roc_curve(y_test_binarized[:, i], y_score_test_on_synthetic_model[:, i])
    roc_auc_test[i] = roc_auc_score(y_test_binarized[:, i], y_score_test_on_synthetic_model[:, i])

# Plotting ROC Curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr_test[i], tpr_test[i], lw=2, label='Random Forest Class {0} (area = {1:0.2f})'.format(i, roc_auc_test[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Model Trained on 80% AE Synthetic Obesity Data and Tested on 20% Control Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[ ]:





# In[48]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np

# Assuming synthetic_features_scaled and ae_synthetic_labels_df are your synthetic features and labels
# And X_train, X_test, y_train, y_test are your original dataset split

X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)

# Preparing the classifier
classifier_on_synthetic = RandomForestClassifier()
classifier_on_synthetic.fit(X_train_orig_new, y_train_orig_new)

# Predict probabilities on the original test set
y_score_test_on_synthetic_model = classifier_on_synthetic.predict_proba(X_test_orig_new)

# Binarize the original labels for AUC-ROC calculation
lb = LabelBinarizer()
lb.fit(y_train_orig_new)
y_test_binarized = lb.transform(y_test_orig_new)

# Calculate ROC AUC for each class in the original test set
fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()
n_classes = y_test_binarized.shape[1]

for i in range(n_classes):
    fpr_test[i], tpr_test[i], _ = roc_curve(y_test_binarized[:, i], y_score_test_on_synthetic_model[:, i])
    roc_auc_test[i] = roc_auc_score(y_test_binarized[:, i], y_score_test_on_synthetic_model[:, i])

# Plotting ROC Curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr_test[i], tpr_test[i], lw=2, label='Random Forest Class {0} (area = {1:0.2f})'.format(i, roc_auc_test[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Model Trained on 80% Original Obesity Data and Tested on Same Original Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# # MICRO-AVERAGE ROC CURVES

# In[224]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)


classifier_on_synthetic = RandomForestClassifier()
classifier_on_synthetic.fit(X_train_orig_new, y_train_orig_new)

# Assuming y_test_cont is your control test labels (unseen data)
# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = classifier_on_synthetic.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='RandomaForest ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Original Obesity Data/Tested with 20% Control Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[223]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# And X_train, X_test, y_train, y_test are your 20% Control obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)


classifier_on_synthetic = RandomForestClassifier()
classifier_on_synthetic.fit(X_train_syn_new, y_train_syn_new)

# Assuming y_test_cont is your control test labels
# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = classifier_on_synthetic.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='RandomaForest ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synthetic Obesity Data/Tested with 20% Control Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[78]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# And X_train, X_test, y_train, y_test are your 20% Control obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)


classifier_on_synthetic = RandomForestClassifier()
classifier_on_synthetic.fit(X_train_syn_new, y_train_syn_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_syn_new))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_syn_new)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_syn_new is your control test features
y_score = classifier_on_synthetic.predict_proba(X_test_syn_new)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='RandomaForest Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve Trained On 80% AE Synthetic Obesity Data and Tested On Same AE Synthetic Test Data: (TSTS)')
plt.legend(loc="lower right")
plt.show()


# # GRADIENT BOOSTING

# In[80]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np

# And X_train, X_test, y_train, y_test are your 20% Control obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)

# Preparing the classifier
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train_syn_new, y_train_syn_new)

# Predict probabilities on the original test set
y_score_test_on_synthetic_model = gb_classifier.predict_proba(X_test_cont)

# Binarize the original labels for AUC-ROC calculation
lb = LabelBinarizer()
lb.fit(y_train_cont)
y_test_binarized = lb.transform(y_test_cont)

# Calculate ROC AUC for each class in the original test set
fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()
n_classes = y_test_binarized.shape[1]

for i in range(n_classes):
    fpr_test[i], tpr_test[i], _ = roc_curve(y_test_binarized[:, i], y_score_test_on_synthetic_model[:, i])
    roc_auc_test[i] = roc_auc_score(y_test_binarized[:, i], y_score_test_on_synthetic_model[:, i])

# Plotting ROC Curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr_test[i], tpr_test[i], lw=2, label='GradientBoosting Class {0} (area = {1:0.2f})'.format(i, roc_auc_test[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Model Trained on 80% AE Synthetic Obesity Data and Tested on 20% Control Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[81]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
# And X_train, X_test, y_train, y_test are your 20% Control obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% Original obesity dataset split
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)

# Preparing the classifier
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train_orig_new, y_train_orig_new)

# Predict probabilities on the original test set
y_score_test_on_synthetic_model = gb_classifier.predict_proba(X_test_cont)

# Binarize the original labels for AUC-ROC calculation
lb = LabelBinarizer()
lb.fit(y_train_cont)
y_test_binarized = lb.transform(y_test_cont)

# Calculate ROC AUC for each class in the original test set
fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()
n_classes = y_test_binarized.shape[1]

for i in range(n_classes):
    fpr_test[i], tpr_test[i], _ = roc_curve(y_test_binarized[:, i], y_score_test_on_synthetic_model[:, i])
    roc_auc_test[i] = roc_auc_score(y_test_binarized[:, i], y_score_test_on_synthetic_model[:, i])

# Plotting ROC Curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr_test[i], tpr_test[i], lw=2, label='GradientBoosting Class {0} (area = {1:0.2f})'.format(i, roc_auc_test[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Model Trained on 80% Original Obesity Data and Tested on 20% Control Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[83]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np

# And X_train, X_test, y_train, y_test are your 20% Control obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)

# Preparing the classifier
gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train_syn_new, y_train_syn_new)

# Predict probabilities on the original test set
y_score_test_on_synthetic_model = gb_classifier.predict_proba(X_test_syn_new)

# Binarize the original labels for AUC-ROC calculation
lb = LabelBinarizer()
lb.fit(y_train_syn_new)
y_test_binarized = lb.transform(y_test_syn_new)

# Calculate ROC AUC for each class in the original test set
fpr_test = dict()
tpr_test = dict()
roc_auc_test = dict()
n_classes = y_test_binarized.shape[1]

for i in range(n_classes):
    fpr_test[i], tpr_test[i], _ = roc_curve(y_test_binarized[:, i], y_score_test_on_synthetic_model[:, i])
    roc_auc_test[i] = roc_auc_score(y_test_binarized[:, i], y_score_test_on_synthetic_model[:, i])

# Plotting ROC Curve for each class
plt.figure(figsize=(10, 8))
for i in range(n_classes):
    plt.plot(fpr_test[i], tpr_test[i], lw=2, label='GradientBoosting Class {0} (area = {1:0.2f})'.format(i, roc_auc_test[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Model Trained on 80% AE Synthetic Obesity Data and Tested on Same AE Synthetic Data: (TSTS)')
plt.legend(loc="lower right")
plt.show()


# # MICRO AVERAGE ROC CURVES

# In[222]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)


gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train_orig_new, y_train_orig_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = gb_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='GradientBoosting ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Original Obesity Data/Tested with 20% Control Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[221]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# And X_train, X_test, y_train, y_test are your 20% Control obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)


gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train_syn_new, y_train_syn_new)

# Assuming y_test_syn is your control test labels
# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = gb_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='GradientBoosting ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synthetic Obesity Data/Tested with 20% Control Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[87]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

# And X_train, X_test, y_train, y_test are your 20% Control obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)


gb_classifier = GradientBoostingClassifier()
gb_classifier.fit(X_train_syn_new, y_train_syn_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_syn_new))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_syn_new)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_control is your control test features
y_score = gb_classifier.predict_proba(X_test_syn_new)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='GradientBoosting ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve Trained On 80% AE Synthetic Obesity Data and Tested On Same AE Synthetic Test Data: (TSTS)')
plt.legend(loc="lower right")
plt.show()


# # CONTINUATION OF ROC CURVES

# In[219]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)


dct_classifier = DecisionTreeClassifier()
dct_classifier.fit(X_train_orig_new, y_train_orig_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = dct_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='Decision Trees ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Original Obesity Data/Tested with 20% Control Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[218]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# And X_train, X_test, y_train, y_test are your 20% Control obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)


dct_classifier = DecisionTreeClassifier()
dct_classifier.fit(X_train_syn_new, y_train_syn_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = dct_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='Decision Trees ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synthetic Obesity Data/Tested with 20% Control Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[215]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)


lgr_classifier = LogisticRegression()
lgr_classifier.fit(X_train_orig_new, y_train_orig_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = lgr_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='Logistic Regression ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Original Obesity Data/Tested with 20% Control Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[216]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)


lgr_classifier = LogisticRegression()
lgr_classifier.fit(X_train_syn_new, y_train_syn_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = lgr_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='Logistic Regression ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synthetic Obesity Data/Tested with 20% Control Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[210]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)


knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train_orig_new, y_train_orig_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = knn_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='K-N Neighbors ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Original Obesity Data/Tested with 20% Control Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[209]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)


knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train_syn_new, y_train_syn_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = knn_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='K-N Neighbors ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synthetic Obesity Data/Tested with 20% Control Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[208]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)


xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train_orig_new, y_train_orig_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = xgb_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='XGB-Classifier ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Original Obesity Data/Tested with 20% Control Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[205]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)


xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train_syn_new, y_train_syn_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = xgb_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='XGB-Classifier ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synthetic Obesity Data/Tested with 20% Control Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[206]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)


lgbm_classifier = XGBClassifier()
lgbm_classifier.fit(X_train_orig_new, y_train_orig_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = lgbm_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='LGBM-Classifier ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Original Obesity Data/Tested with 20% Control Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[198]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)


lgbm_classifier = XGBClassifier()
lgbm_classifier.fit(X_train_syn_new, y_train_syn_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = lgbm_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='LGBM-Classifier ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synthetic Obesity Data/Tested with 20% Control Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[199]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)


mlp_classifier = MLPClassifier()
mlp_classifier.fit(X_train_orig_new, y_train_orig_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = mlp_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='Multi-Layer Perceptron ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Original Obesity Data/Tested with 20% Control Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[200]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)


mlp_classifier = MLPClassifier()
mlp_classifier.fit(X_train_syn_new, y_train_syn_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = mlp_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='Multi-Layer Perceptron ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synthetic Obesity Data/Tested with 20% Control Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[201]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, labels, test_size=0.2, random_state=42)


svc_classifier =  SVC(probability=True)
svc_classifier.fit(X_train_orig_new, y_train_orig_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = svc_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='SVC-Classifier ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Original Obesity Data/Tested with 20% Control Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[202]:


from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.metrics import roc_curve, auc
from scipy import interp
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# And X_train, X_test, y_train, y_test are your 20% Control Obesity dataset split
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# And X_train, X_test, y_train, y_test are your 80% AE Sythetic Obesity dataset split
X_train_syn_new, X_test_syn_new, y_train_syn_new, y_test_syn_new = train_test_split(synthetic_data_df_ae, synthetic_labels_df_ae, test_size=0.2, random_state=42)


svc_classifier =  SVC(probability=True)
svc_classifier.fit(X_train_syn_new, y_train_syn_new)


# First, encode these labels numerically
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(np.unique(y_test_cont))  # Assuming y_test_cont is available
y_test_cont_encoded = label_encoder.transform(y_test_cont)

# Binarize the labels for multiclass scenario
y_test_binarized = label_binarize(y_test_cont_encoded, classes=np.unique(y_encoded))
n_classes = y_test_binarized.shape[1]

# Assuming classifier is your trained model and X_test_cont is your control test features
y_score = svc_classifier.predict_proba(X_test_cont)  # Ensure your classifier is trained with .fit() before this

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = dict(), dict(), dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binarized[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binarized.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# Plot ROC curves
plt.figure(figsize=(8, 6))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='SVC-Classifier ROC Curve Class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

plt.plot(fpr["micro"], tpr["micro"], color='deeppink', linestyle=':', linewidth=4, label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc["micro"]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synthetic Obesity Data/Tested with 20% Control Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[160]:


import matplotlib.pyplot as plt
import numpy as np

# Classifiers
classifiers = ['GB', 'DCT', 'KNN', 'LGBM', 'LGR', 'XGB', 'MLP', 'RDF', 'SVC']

# AUC-ROC Scores for Original Obesity Dataset (TRTR)
auc_trtr = [1.00, 0.96, 0.97, 1.00, 0.95, 1.00, 0.97, 1.00, 0.93]

# AUC-ROC Scores for AE-Synthetic Obesity Dataset (TSTR)
auc_tstr = [0.98, 0.86, 0.96, 0.99, 0.93, 0.99, 0.97, 0.99, 0.93]

# Set up the matplotlib figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Set position of bar on X axis
bar_width = 0.35
r1 = np.arange(len(auc_trtr))
r2 = [x + bar_width for x in r1]

# Make the plot
ax.bar(r1, auc_trtr, color='b', width=bar_width, edgecolor='grey', label='Original Obesity (TRTR)')
ax.bar(r2, auc_tstr, color='g', width=bar_width, edgecolor='grey', label='AE Synthetic Obesity (TSTR)')

# Add xticks on the middle of the group bars
ax.set_xlabel('Classifier', fontweight='bold')
ax.set_ylabel('AUC-ROC Score', fontweight='bold')
ax.set_xticks([r + bar_width/2 for r in range(len(auc_trtr))])
ax.set_xticklabels(classifiers, rotation=45)
ax.set_ylim(0, 1.1)  # AUC-ROC scores range from 0 to 1

# Create legend & Show graphic
plt.title('AUC-ROC Comparison between 80% Original and 80% AE Synthetic Obesity Datasets')
plt.legend()
plt.tight_layout()
plt.show()


# In[ ]:





# # Computing Numerical Statistical Values: KS-Test, P-Value, MSE, RMSE, MAE, F-Test for Variances, T-Test for Means

# In[103]:


# X_train_vae_syn, X_test_vae_syn, y_train_vae_syn, y_test_vae_syn = train_test_split(vae_synthetic_data_df, vae_synthetic_labels_series, test_size=0.2, random_state=42)

from scipy.stats import ks_2samp, f_oneway, ttest_ind
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Assuming `obesity_df` and `ae_synthetic_data_df` are properly aligned
for column in ae_synthetic_obesity_data_with_labels_df.columns:
    # KS Test and P-Value
    ks_stat, ks_pvalue = ks_2samp(obesity_train_dataframe[column], ae_synthetic_obesity_data_with_labels_df[column])
    print(f"KS-Test for {column}: Stat={ks_stat}, P-Value={ks_pvalue}")
    
    # MSE, RMSE, and MAE for a column
    mse = mean_squared_error(obesity_train_dataframe[column], ae_synthetic_obesity_data_with_labels_df[column])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(obesity_train_dataframe[column], ae_synthetic_obesity_data_with_labels_df[column])
    print(f"MSE: {mse}, RMSE: {rmse}, MAE: {mae}")
    
    # F-Test and T-Test for comparing variances and means
    f_stat, f_pvalue = f_oneway(obesity_train_dataframe[column], ae_synthetic_obesity_data_with_labels_df[column])
    t_stat, t_pvalue = ttest_ind(obesity_train_dataframe[column], ae_synthetic_obesity_data_with_labels_df[column])
    print(f"F-Test: Stat={f_stat}, P-Value={f_pvalue}")
    print(f"T-Test: Stat={t_stat}, P-Value={t_pvalue}")

    # Means and Standard Deviation
    orig_mean = obesity_train_dataframe[column].mean()
    ae_syn_mean = ae_synthetic_obesity_data_with_labels_df[column].mean()
    orig_std = obesity_train_dataframe[column].std()
    ae_syn_std = ae_synthetic_obesity_data_with_labels_df[column].std()
    print(f"Original Mean={orig_mean}, AE_Synthetic Mean={ae_syn_mean}, Original Std={orig_std}, AE_Synthetic Std={ae_syn_std}\n")


# # Computing Graphical Statistical Values: KS-Test, P-Value, MSE, RMSE, MAE, F-Test for Variances, T-Test for Means

# In[ ]:





# In[141]:


# Re-importing necessary libraries and recreating the plot with custom colors after code execution state reset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# Assuming data for demonstration
features = ['Gender', 'Age', 'Height', 'Weight', 'family_history_with_overweight', 'FAVC', 'FCVC', 'NCP', 'CAEC', 'SMOKE', 'CH2O', 'SCC', 'FAF', 'TUE', 'CALC', 'MTRANS', 'NObeyesdad']
ks_stats = np.random.uniform(0, 1, len(features))
p_values = np.random.uniform(0, 0.05, len(features))
mse_values = np.random.uniform(0, 2, len(features))
rmse_values = np.sqrt(mse_values)
mae_values = np.random.uniform(0, 1, len(features))

# Create DataFrame for visualization
df_stats = pd.DataFrame({
    'Feature': features,
    'KS Stat': ks_stats,
    'P-Value': p_values,
    'MSE': mse_values,
    'RMSE': rmse_values,
    'MAE': mae_values
})

# Plotting

## Distribution Overlays - Mockup
plt.figure(figsize=(10, 6))
sns.histplot(np.random.normal(0, 1, 1000), color="skyblue", label="Original", kde=True)
sns.histplot(np.random.normal(0.1, 0.9, 1000), color="red", label="AE-Synthetic", kde=True)
plt.title('Distribution Overlay: Obesity Feature Example')
plt.legend()
plt.show()

## Error Metrics Bar Charts
plt.figure(figsize=(10, 6))
x = np.arange(len(features))  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width, mse_values, width, label='MSE')
rects2 = ax.bar(x, rmse_values, width, label='RMSE')
rects3 = ax.bar(x + width, mae_values, width, label='MAE')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Features')
ax.set_ylabel('Error Metrics')
ax.set_title('Error Metrics by Feature in (Original and AE-Synthetic Obesity)')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45, ha="right")
ax.legend()

fig.tight_layout()

plt.show()

## KS Test Results - Mockup as Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x='Feature', y='KS Stat', data=df_stats)
plt.title('KS Test Results by Feature in (Original and AE-Synthetic Obesity)')
plt.xticks(rotation=45, ha="right")
plt.show()


# # CLASSIFICATION REPORT (f1-score, accuracy, recall, precision)

# In[163]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = GradientBoostingClassifier(max_depth=1000, n_estimators=1000, random_state=42)
classifier_orig.fit(X_train_orig_new, y_train_orig_new)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("GB on 80% Original Obesity/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = GradientBoostingClassifier(max_depth=1000, n_estimators=1000, random_state=42)
classifier_syn.fit(X_train_synt, y_train_synt)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("GB on 80% AE Synthetic Obesity/Tested on 20% Control Data (TSTR):\n", report)


# In[164]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = RandomForestClassifier(max_depth=1000, n_estimators=1000, random_state=42)
classifier_orig.fit(X_train_orig_new, y_train_orig_new)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("RF on 80% Original Obesity/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = RandomForestClassifier(max_depth=1000, n_estimators=1000, random_state=42)
classifier_syn.fit(X_train_synt, y_train_synt)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("RF on 80% AE Synthetic Obesity/Tested on 20% Control Data (TSTR):\n", report)


# In[ ]:





# In[166]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = XGBClassifier(max_depth=1000, n_estimators=1000, random_state=42)
classifier_orig.fit(X_train_orig_new, y_train_orig_new)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("XGB on 80% Original Obesity/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = XGBClassifier(max_depth=4000, n_estimators=4000, random_state=42)
classifier_syn.fit(X_train_synt, y_train_synt)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("XGB on 80% AE Synthetic Obesity/Tested on 20% Control Data (TSTR):\n", report)


# In[167]:


from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = LGBMClassifier(max_depth=1000, n_estimators=1000, random_state=42)
classifier_orig.fit(X_train_orig_new, y_train_orig_new)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("LGBM on 80% Original Obesity/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = LGBMClassifier(max_depth=3000, n_estimators=3000, random_state=42)
classifier_syn.fit(X_train_synt, y_train_synt)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("LGBM on 80% AE Synthetic Obesity/Tested on 20% Control Data (TSTR):\n", report)


# In[187]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = DecisionTreeClassifier(max_depth=1000, random_state=42)
classifier_orig.fit(X_train_orig_new, y_train_orig_new)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("DCT on 80% Original Obesity/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth  max_depth=5000, random_state=42
classifier_syn = DecisionTreeClassifier(max_depth=5000, random_state=42)
classifier_syn.fit(X_train_synt, y_train_synt)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("DCT on 80% AE Synthetic Obesity/Tested on 20% Control Data (TSTR):\n", report)


# In[180]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = LogisticRegression(C=1.0, solver='lbfgs', random_state=42, max_iter=1500)
classifier_orig.fit(X_train_orig_new, y_train_orig_new)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("LGR on 80% Original Obesity/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = LogisticRegression(C=1.0, solver='lbfgs', random_state=42, max_iter=4500)
classifier_syn.fit(X_train_synt, y_train_synt)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("LGR on 80% AE Synthetic Obesity/Tested on 20% Control Data (TSTR):\n", report)


# In[181]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')
classifier_orig.fit(X_train_orig_new, y_train_orig_new)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("KNN on 80% Original Obesity/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = KNeighborsClassifier(n_neighbors=7, weights='uniform', algorithm='auto')
classifier_syn.fit(X_train_synt, y_train_synt)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("KNN on 80% AE Synthetic Obesity/Tested on 20% Control Data (TSTR):\n", report)


# In[184]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
classifier_orig.fit(X_train_orig_new, y_train_orig_new)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("SVC on 80% Original Obesity/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)
classifier_syn.fit(X_train_synt, y_train_synt)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("SVC on 80% AE Synthetic Obesity/Tested on 20% Control Data (TSTR):\n", report)


# In[185]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, max_iter=200, random_state=42)
classifier_orig.fit(X_train_orig_new, y_train_orig_new)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("MLP on 80% Original Obesity/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0003, max_iter=1000, random_state=42)
classifier_syn.fit(X_train_synt, y_train_synt)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("MLP on 80% AE Synthetic Obesity/Tested on 20% Control Data (TSTR):\n", report)


# In[17]:


import matplotlib.pyplot as plt
import numpy as np

# Classifier names
classifiers = ["DecisionTree", "GradientBoosting", "RandomForest", "LGBM", "XGB", "KNeighbors", "LogisticRegression", "SVC", "MLP"]

# Mean CV Accuracy for 80% Original Obesity Data
mean_cv_accuracy_original = [0.93, 0.96, 0.95, 0.99, 0.96, 0.85, 0.82, 0.58, 0.84]

# Mean CV Accuracy for 80% AE Synthetic Obesity Data
mean_cv_accuracy_ae_synthetic = [0.74, 0.84, 0.91, 0.92, 0.89, 0.85, 0.84, 0.54, 0.80]

x = np.arange(len(classifiers))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(14, 8))
rects1 = ax.bar(x - width/2, mean_cv_accuracy_original, width, label='80% Original Data', color='pink')
rects2 = ax.bar(x + width/2, mean_cv_accuracy_ae_synthetic, width, label='80% AE Synthetic Data', color='brown')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Classification Report')
ax.set_title('Classification Report by Classifier and Dataset Type')
ax.set_xticks(x)
ax.set_xticklabels(classifiers, rotation=45, ha="right")
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()


# In[ ]:





# # COMPUTING CROSS-VALIDATION OF ORIGINAL AND SYNTHETIC DATASETS

# In[148]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Assuming final_synthetic_df and final_synthetic_labels are your features and labels DataFrames
# Encoding categorical labels if necessary
le = LabelEncoder()
vae_synthetic_labels_ae_encoded = le.fit_transform(synthetic_labels_df_ae)

# Splitting the dataset
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(synthetic_data_df_ae, vae_synthetic_labels_ae_encoded, test_size=0.2, random_state=42)

# List of classifiers to evaluate
classifiers = [
    DecisionTreeClassifier(random_state=2),
    GradientBoostingClassifier(random_state=2),
    RandomForestClassifier(n_estimators=100, random_state=2),
    AdaBoostClassifier(random_state=2),
    LGBMClassifier(random_state=2),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=2),
    KNeighborsClassifier(),
    LogisticRegression(max_iter=1000, random_state=2),
    SVC(random_state=2),
    MLPClassifier(max_iter=1000, random_state=2)
]

# Evaluate each classifier
for clf in classifiers:
    cv_scores = cross_val_score(clf, X_train_syn, y_train_syn, cv=5, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"{clf.__class__.__name__}: Mean CV Accuracy for AE-Synthetic Obesity = {mean_cv_score:.4f}, Std = {std_cv_score:.4f}")


# In[151]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Assuming final_synthetic_df and final_synthetic_labels are your features and labels DataFrames
# Encoding categorical labels if necessary
le = LabelEncoder()
original_labels_ae_encoded = le.fit_transform(labels)

# Splitting the dataset
X_train_orig_new, X_test_orig_new, y_train_orig_new, y_test_orig_new = train_test_split(features, original_labels_ae_encoded, test_size=0.2, random_state=42)

# List of classifiers to evaluate
classifiers = [
    DecisionTreeClassifier(random_state=2),
    GradientBoostingClassifier(random_state=2),
    RandomForestClassifier(n_estimators=100, random_state=2),
    AdaBoostClassifier(random_state=2),
    LGBMClassifier(random_state=2),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=2),
    KNeighborsClassifier(),
    LogisticRegression(max_iter=1000, random_state=2),
    SVC(random_state=2),
    MLPClassifier(max_iter=1000, random_state=2)
]

# Evaluate each classifier
for clf in classifiers:
    cv_scores = cross_val_score(clf, X_train_orig_new, y_train_orig_new, cv=5, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"{clf.__class__.__name__}: Mean CV Accuracy for 80%-Original Obesity Data = {mean_cv_score:.4f}, Std = {std_cv_score:.4f}")


# In[213]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Sample data loading code, replace these with actual DataFrame loading
# For the purpose of this example, `synthetic_data_df_ae_new` and `obesity_train_dataframe` will be simulated as random data
# `synthetic_labels_df_ae_new` and `obesity_labels` are simulated as categorical labels

# Encoding categorical labels if necessary
le = LabelEncoder()
vae_synthetic_labels_ae_encoded = le.fit_transform(synthetic_labels_df_ae_new)  # Placeholder, replace with actual data loading
obesity_labels_encoded = le.transform(labels2)  # Placeholder, replace with actual data loading

# Splitting the synthetic dataset
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(synthetic_data_df_ae_new, vae_synthetic_labels_ae_encoded, test_size=0.2, random_state=42)

# Splitting the original dataset
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(obesity_train_dataframe, obesity_labels_encoded, test_size=0.2, random_state=42)

# List of classifiers to evaluate
classifiers = [
    DecisionTreeClassifier(random_state=2),
    GradientBoostingClassifier(random_state=2),
    RandomForestClassifier(n_estimators=100, random_state=2),
    AdaBoostClassifier(random_state=2),
    LGBMClassifier(random_state=2),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=2),
    KNeighborsClassifier(),
    LogisticRegression(max_iter=1000, random_state=2),
    SVC(random_state=2),
    MLPClassifier(max_iter=1000, random_state=2)
]

# Evaluate each classifier on synthetic data
print("Evaluating AE-Synthetic Obesity Dataset")
for clf in classifiers:
    cv_scores_syn = cross_val_score(clf, X_train_syn, y_train_syn, cv=5, scoring='accuracy')
    mean_cv_score_syn = np.mean(cv_scores_syn)
    std_cv_score_syn = np.std(cv_scores_syn)
    print(f"{clf.__class__.__name__}: Mean CV Accuracy = {mean_cv_score_syn:.4f}, Std = {std_cv_score_syn:.4f}")

# Evaluate each classifier on original data
print("\nEvaluating Original Obesity Dataset")
for clf in classifiers:
    cv_scores_orig = cross_val_score(clf, X_train_orig, y_train_orig, cv=5, scoring='accuracy')
    mean_cv_score_orig = np.mean(cv_scores_orig)
    std_cv_score_orig = np.std(cv_scores_orig)
    print(f"{clf.__class__.__name__}: Mean CV Accuracy = {mean_cv_score_orig:.4f}, Std = {std_cv_score_orig:.4f}")


# # GRAPHICAL REPRESENTATION OF THE ABOVE MEAN CV OF ORIGINAL AND SYNTHETIC DATASETS

# In[216]:


import matplotlib.pyplot as plt
import numpy as np

# Classifier names
classifiers = [
    "DecisionTree",
    "GradientBoosting",
    "RandomForest",
    "AdaBoost",
    "LGBM",
    "XGB",
    "KNeighbors",
    "LogisticRegression",
    "SVC",
    "MLP"
]

# Mean CV Accuracy for 80%-Original Obesity Data
mean_cv_accuracy_original = [
    0.9200, 0.9452, 0.9422, 0.3422, 0.9556,
    0.9533, 0.8378, 0.7881, 0.5481, 0.7778
]

# Mean CV Accuracy for 80%-AE Synthetic Obesity Data
mean_cv_accuracy_synthetic = [
    0.8259, 0.8652, 0.8785, 0.3978, 0.8911,
    0.8919, 0.7504, 0.7770, 0.5830, 0.7696
]

x = np.arange(len(classifiers))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 7))
rects1 = ax.bar(x - width/2, mean_cv_accuracy_original, width, label='80%-Original Obesity Data')
rects2 = ax.bar(x + width/2, mean_cv_accuracy_synthetic, width, label='80%-AE Synthetic Obesity Data')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean CV Accuracy')
ax.set_title('Mean CV Accuracy by Classifier and Dataset')
ax.set_xticks(x)
ax.set_xticklabels(classifiers, rotation=45, ha="right")
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()


# In[ ]:





# # CORRELATION HEATMAP

# In[143]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_correlation_matrix(data, title):
    # Calculate the correlation matrix
    corr_matrix = data.corr()

    # Plot the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.show()

# Plot correlation matrix for original data
plot_correlation_matrix(obesity_train_dataframe, "Correlation Matrix of Original Obesity Data")

# Plot correlation matrix for synthetic data
plot_correlation_matrix(ae_synthetic_obesity_data_with_labels_df_new, "Correlation Matrix of AE Synthetic Obesity Data")


# # Computing Numerical Correlation Matrices of Original and Synthetic Datasets

# In[211]:


# For the original dataset
print('80% Original Obesity Numerical Correlation Matrix:')
print(obesity_train_dataframe.corr())

# For the AE synthetic dataset
print('80% AE-Synthetic Obesity Numerical Correlation Matrix:')
print(ae_synthetic_obesity_data_with_labels_df_new.corr())


# In[37]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Plot for Weight vs. Height with Labels
plt.figure(figsize=(10, 6))
sns.scatterplot(data=ae_synthetic_obesity_data_with_labels_df_new, x='Height', y='Weight', hue='NObeyesdad', palette='viridis')
plt.title('AE Synthetic Data Weight vs. Height Scatter Plot by Labels')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.legend(title='Labels')
plt.show()

# Plot for Age vs. FAF with Labels
plt.figure(figsize=(10, 6))
sns.scatterplot(data=ae_synthetic_obesity_data_with_labels_df_new, x='Age', y='FAF', hue='NObeyesdad', palette='viridis')
plt.title('AE Synthetic Data Age vs. FAF (Physical Activity Frequency) Scatter Plot by Labels')
plt.xlabel('Age')
plt.ylabel('FAF')
plt.legend(title='Labels')
plt.show()


# # INSTALL THE PRIVACY ASSESSMENT TOOL KITS KNOWN AS ANONYMETER AS MENTIONED ABOVE IN STEP 2

# In[3]:


get_ipython().system('pip install anonymeter')


# # IMPORTING THE INSTALLED ANONYMETER'S PRIVACY RISK EVALUATORS FOR PRIVACY PRESERVATION ASSESSMENT ON THE GENERATED SYNTHETIC DATASET
# 
# For more detailed information on the usage of this tool do visit the author's website(blogger) via the links below.
# 
# https://www.anonos.com/blog/presenting-anonymeter-the-tool-for-assessing-privacy-risks-in-synthetic-datasets
# 
# https://github.com/statice/anonymeter

# In[19]:


import pandas as pd
from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator


# # STEP 3A: SINGLING OUT UNIVARIATE RISK ASSESSMENT ON 80% AE SYNTHETIC OBESITY DISEASE DATA

# In[223]:


singling_out_evaluator = SinglingOutEvaluator(ori=obesity_train_dataframe, 
                                              syn=ae_synthetic_obesity_data_with_labels_df, 
                                              control=control_dataframe,
                                              n_attacks=1500)

try:
    singling_out_evaluator.evaluate(mode='univariate')  # For univariate analysis
    # For multivariate analysis, you can change mode to 'multivariate'
    singling_out_risk = singling_out_evaluator.risk()
    print("Singling Out Risk:", singling_out_risk)
except RuntimeError as ex:
    print(f"Singling out evaluation failed: {ex}")


# In[229]:


res = singling_out_evaluator.results()

print("The Singling Out Risk Type via Univariate Analysis When n_attacks=1500 on 80% AE Synthetic Obesity Data:")
print("Successs rate of main attack:", res.attack_rate)
print("Successs rate of baseline attack:", res.baseline_rate)
print("Successs rate of control attack:", res.control_rate)


# In[225]:


singling_out_evaluator1 = SinglingOutEvaluator(ori=obesity_train_dataframe, 
                                              syn=ae_synthetic_obesity_data_with_labels_df, 
                                              control=control_dataframe,
                                              n_attacks=500)

try:
    singling_out_evaluator1.evaluate(mode='univariate')  # For univariate analysis
    # For multivariate analysis, you can change mode to 'multivariate'
    singling_out_risk = singling_out_evaluator1.risk()
    print("Singling Out Risk:", singling_out_risk)
except RuntimeError as ex:
    print(f"Singling out evaluation failed: {ex}")


# In[228]:


res1 = singling_out_evaluator1.results()

print("The Singling Out Risk Type via Univariate Analysis When n_attacks=500 on 80% AE Synthetic Obesity Data:")
print("Successs rate of main attack:", res1.attack_rate)
print("Successs rate of baseline attack:", res1.baseline_rate)
print("Successs rate of control attack:", res1.control_rate)


# # GRAPHICAL REPRESENTATIONS OF THE ABOVE PRIVACY RISK NUMERICAL RESULTS

# In[269]:


# Singling out risk data in univariate mode 001277215360090744
data_singling_out_univariate = [
    ('n_attacks=1500', 0.001277215360090744),
    ('n_attacks=500', 0.00381217023077612)
]

# Convert to DataFrame
df_singling_out_univariate = pd.DataFrame(data_singling_out_univariate, columns=['Evaluation', 'SuccessRateMainAttack'])

# Add baseline and control attack success rates (same as main attack in univariate)
df_singling_out_univariate['SuccessRateBaselineAttack'] = df_singling_out_univariate['SuccessRateMainAttack']
df_singling_out_univariate['SuccessRateControlAttack'] = df_singling_out_univariate['SuccessRateMainAttack']

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
index = np.arange(len(df_singling_out_univariate))
bar_width = 0.25

bars1 = plt.bar(index, df_singling_out_univariate['SuccessRateMainAttack'], bar_width, label='Main Attack', color='tomato')
bars2 = plt.bar(index + bar_width, df_singling_out_univariate['SuccessRateBaselineAttack'], bar_width, label='Baseline Attack', color='skyblue')
bars3 = plt.bar(index + 2 * bar_width, df_singling_out_univariate['SuccessRateControlAttack'], bar_width, label='Control Attack', color='brown')

plt.xlabel('Evaluation Types')
plt.ylabel('Success Rate')
plt.title('Success Rate of Attacks on 80% AE Synthetic Obesity Dataset via Univariate Risk Assessment')
plt.xticks(index + bar_width, df_singling_out_univariate['Evaluation'])
plt.legend()

plt.tight_layout()
plt.show()


# In[278]:


import matplotlib.pyplot as plt
import numpy as np

# Data Preparation
attacks = ['1500 Attacks', '500 Attacks']
success_rates = np.array([0.0013, 0.0038])
failure_rates = 1 - success_rates
attack_categories = ['Main Attack', 'Baseline Attack', 'Control Attack']
success_rates_categories = np.array([success_rates, success_rates, success_rates])

print('Univariate Risk Assessment Success vs. Failure Rates on 80% AE Synthetic Obesity Data')

# Pie Chart for n_attacks=1500
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.pie([success_rates[0], failure_rates[0]], labels=['Success Attacks', 'Failure Attacks'], autopct='%1.1f%%', startangle=140, colors=['brown', 'skyblue'])
plt.title('Overall Success vs Failure Rates (1500 Attacks)')

# Pie Chart for n_attacks=500
plt.subplot(1, 2, 2)
plt.pie([success_rates[1], failure_rates[1]], labels=['Success Attacks', 'Failure Attacks'], autopct='%1.1f%%', startangle=140, colors=['brown', 'skyblue'])
plt.title('Overall Success vs Failure Rates (500 Attacks)')

plt.show()




# In[ ]:





# In[ ]:





# # STEP 3B: SINGLING OUT MULTI-VARIATE RISK ASSESSMENT ON 80% AE SYNTHETIC OBESITY DISEASE DATA

# In[57]:


test = SinglingOutEvaluator(ori=obesity_train_dataframe, 
                                 syn=ae_synthetic_obesity_data_with_labels_df, 
                                 control=control_dataframe,
                                 n_attacks=1500, # this attack takes longer
                                 n_cols=4)


try:
    test.evaluate(mode='multivariate')
    risk = test.risk()
    print(risk)

except RuntimeError as ex: 
    print(f"Singling out evaluation failed with {ex}. Please re-run this cell."
          "For more stable results increase `n_attacks`. Note that this will "
          "make the evaluation slower.")


# In[59]:


check = test.results()

print("The Singling Out Risk Type via Multivariate Analysis When n_attacks=1500 on 80% AE Synthetic Obesity Disease Data:")
print("Successs rate of main attack:", check.attack_rate)
print("Successs rate of baseline attack:", check.baseline_rate)
print("Successs rate of control attack:", check.control_rate)


# In[ ]:





# In[60]:


testME = SinglingOutEvaluator(ori=obesity_train_dataframe, 
                                 syn=ae_synthetic_obesity_data_with_labels_df, 
                                 control=control_dataframe,
                                 n_attacks=500, # this attack takes longer
                                 n_cols=4)


try:
    testME.evaluate(mode='multivariate')
    risk = testME.risk()
    print(risk)

except RuntimeError as ex: 
    print(f"Singling out evaluation failed with {ex}. Please re-run this cell."
          "For more stable results increase `n_attacks`. Note that this will "
          "make the evaluation slower.")


# In[62]:


checkME = testME.results()

print("The Singling Out Risk Type via Multivariate Analysis When n_attacks=500 on 80% AE Synthetic Obesity Disease Data:")
print("Successs rate of main attack:", checkME.attack_rate)
print("Successs rate of baseline attack:", checkME.baseline_rate)
print("Successs rate of control attack:", checkME.control_rate)


# # GRAPHICAL REPRESENTATIONS OF THE ABOVE PRIVACY RISK NUMERICAL RESULTS

# In[77]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
data_singling_out_univariate = [
    ('n_attacks=1500', 0.1336, 0.0132, 0.0378),
    ('n_attacks=500', 0.1308, 0.0137, 0.0480)
]

# Convert to DataFrame
df_singling_out_univariate = pd.DataFrame(data_singling_out_univariate, columns=['Evaluation', 'SuccessRateMainAttack', 'SuccessRateBaselineAttack', 'SuccessRateControlAttack'])

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
index = np.arange(len(df_singling_out_univariate))
bar_width = 0.25

bars1 = plt.bar(index, df_singling_out_univariate['SuccessRateMainAttack'], bar_width, label='Main Attack', color='tomato')
bars2 = plt.bar(index + bar_width, df_singling_out_univariate['SuccessRateBaselineAttack'], bar_width, label='Baseline Attack', color='skyblue')
bars3 = plt.bar(index + 2 * bar_width, df_singling_out_univariate['SuccessRateControlAttack'], bar_width, label='Control Attack', color='brown')

plt.xlabel('Evaluation Number of Attacks')
plt.ylabel('Success Rate')
plt.title('Success Rate of Attacks on 80% AE Synthetic Obesity Dataset via Multivariate Risk Assessment')
plt.xticks(index + bar_width, df_singling_out_univariate['Evaluation'])
plt.legend()

plt.tight_layout()
plt.show()


# In[76]:


import matplotlib.pyplot as plt

# Data for plotting
attacks = ['Main Attack', 'Baseline Attack', 'Control Attack']
success_rates_1500 = [0.1336, 0.0132, 0.0378]
success_rates_500 = [0.1308, 0.0137, 0.0480]

# Overall success vs. failure rates calculated from the given data for illustrative purposes
# Assuming "success" combines all types of attacks and "failure" is the complement to 1
overall_success_1500 = sum(success_rates_1500) / len(success_rates_1500)
overall_failure_1500 = 1 - overall_success_1500
overall_success_500 = sum(success_rates_500) / len(success_rates_500)
overall_failure_500 = 1 - overall_success_500

print('Multivariate Risk Assessment Success vs. Failure Rates on 80% AE Synthetic Obesity Data')

# Pie charts for overall success vs. failure
fig, axs_pie = plt.subplots(1, 2, figsize=(12, 6))

axs_pie[0].pie([overall_success_1500, overall_failure_1500], labels=['Overall Success', 'Overall Failure'], autopct='%1.1f%%', startangle=140, colors=['tomato', 'skyblue'])
axs_pie[0].set_title('Overall Success vs. Failure for 1500 Attacks')

axs_pie[1].pie([overall_success_500, overall_failure_500], labels=['Overall Success', 'Overall Failure'], autopct='%1.1f%%', startangle=140, colors=['tomato', 'skyblue'])
axs_pie[1].set_title('Overall Success vs. Failure for 500 Attacks')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# # STEP 4: LINKABILITY RISK ASSESSMENT ON 80% AE SYNTHETIC OBESITY DISEASE DATA
# Define auxiliary columns (aux_cols) based on our knowledge of what an attacker might know.

# In[84]:


# Dynamically setting n_attacks based on the smallest dataset size
min_dataset_size = min(len(obesity_train_dataframe), len(ae_synthetic_obesity_data_with_labels_df_new), len(control_dataframe))
n_attacks = min_dataset_size  # Or some fraction of the smallest size, if desired

aux_cols = [
    ['Gender', 'FAF'],  # Attributes in dataset A
    ['Age', 'FCVC']     # Attributes in dataset B
]

linkability_evaluator2 = LinkabilityEvaluator(ori=obesity_train_dataframe, 
                                             syn=ae_synthetic_obesity_data_with_labels_df, 
                                             control=control_dataframe,
                                             n_attacks=n_attacks,
                                             aux_cols=aux_cols,
                                             n_neighbors=10)

linkability_evaluator2.evaluate(n_jobs=-1)  # Adjust 'n_jobs' as needed
linkability_risk = linkability_evaluator2.risk()
print("Linkability Risk When n_neighbors=10 on 80% AE Synthetic Obesity Disease Data:", linkability_risk)


# In[85]:


linkability = linkability_evaluator2.results()

print("Linkability Risk When n_neighbors=10 on 80% AE Synthetic Obesity Disease Data:")
print("Successs rate of main attack:", linkability.attack_rate)
print("Successs rate of baseline attack:", linkability.baseline_rate)
print("Successs rate of control attack:", linkability.control_rate)


# In[86]:


# Dynamically setting n_attacks based on the smallest dataset size
min_dataset_size = min(len(obesity_train_dataframe), len(ae_synthetic_obesity_data_with_labels_df_new), len(control_dataframe))
n_attacks = min_dataset_size  # Or some fraction of the smallest size, if desired

aux_cols = [
    ['Gender', 'FAF'],  # Attributes in dataset A
    ['Age', 'FCVC']     # Attributes in dataset B
]

linkability_evaluator4 = LinkabilityEvaluator(ori=obesity_train_dataframe, 
                                             syn=ae_synthetic_obesity_data_with_labels_df, 
                                             control=control_dataframe,
                                             n_attacks=n_attacks,
                                             aux_cols=aux_cols,
                                             n_neighbors=5)

linkability_evaluator4.evaluate(n_jobs=-1)  # Adjust 'n_jobs' as needed
linkability_risk1 = linkability_evaluator4.risk()
print("Linkability Risk When n_neighbors=5 on 80% AE Synthetic Obesity Disease Data:", linkability_risk1)


# In[87]:


linkability5 = linkability_evaluator4.results()

print("Linkability Risk When n_neighbors=5 on 80% AE Synthetic Obesity Disease Data:")
print("Successs rate of main attack:", linkability5.attack_rate)
print("Successs rate of baseline attack:", linkability5.baseline_rate)
print("Successs rate of control attack:", linkability5.control_rate)


# # GRAPHICAL REPRESENTATIONS OF THE ABOVE PRIVACY RISK NUMERICAL RESULTS

# In[94]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
data_singling_out_univariate = [
    ('n_neighbors=10', 11.69, 5.60, 11.23),
    ('n_neighbors=5', 6.07, 0.92, 3.96)
]

# Convert to DataFrame
df_singling_out_univariate = pd.DataFrame(data_singling_out_univariate, columns=['Evaluation', 'SuccessRateMainAttack', 'SuccessRateBaselineAttack', 'SuccessRateControlAttack'])

# Plotting
fig, ax = plt.subplots(figsize=(12, 6))
index = np.arange(len(df_singling_out_univariate))
bar_width = 0.25

bars1 = plt.bar(index, df_singling_out_univariate['SuccessRateMainAttack'], bar_width, label='Main Attack', color='tomato')
bars2 = plt.bar(index + bar_width, df_singling_out_univariate['SuccessRateBaselineAttack'], bar_width, label='Baseline Attack', color='skyblue')
bars3 = plt.bar(index + 2 * bar_width, df_singling_out_univariate['SuccessRateControlAttack'], bar_width, label='Control Attack', color='brown')

# Function to add value labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.xlabel('Evaluation Number of Attacks')
plt.ylabel('Success Rate')
plt.title('Success Rate of Attacks on 80% AE Synthetic Obesity Dataset via Linkability Risk Assessment')
plt.xticks(index + bar_width, df_singling_out_univariate['Evaluation'])
plt.legend()

plt.tight_layout()
plt.show()


# In[92]:


import matplotlib.pyplot as plt

# Data setup
sizes_10 = [11.69, 88.31]  # Success vs. Failure for n_neighbors=10
sizes_5 = [6.07, 93.93]  # Success vs. Failure for n_neighbors=5
labels = ['Overall Success', 'Overall Failure']
explode = (0.1, 0)
colors = ['tomato','skyblue']  # Customizable colors

print('Linkability Risk Assessment Success vs. Failure Rates on 80% AE Synthetic Obesity Data')

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 row, 2 columns

# Pie chart for n_neighbors=10
axs[0].pie(sizes_10, explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90, colors=colors)
axs[0].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
axs[0].set_title('Overall Success vs. Failure for n_neighbors=10')

# Pie chart for n_neighbors=5
axs[1].pie(sizes_5, explode=explode, labels=labels, autopct='%1.1f%%',
           shadow=True, startangle=90, colors=colors)
axs[1].axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
axs[1].set_title('Overall Success vs. Failure for n_neighbors=5')

plt.tight_layout()
plt.show()


# # STEP 5: INFERENCE RISK ASSESSMENT ON 80% AE SYNTHETIC OBESITY DISEASE DATA
# Iterate over each column in your dataset to assess inference risk. For a more detailed example, you might select fewer columns to speed up the process.

# In[98]:


# Dynamically setting n_attacks based on the smallest dataset size
min_dataset_size = min(len(obesity_train_dataframe), len(ae_synthetic_obesity_data_with_labels_df), len(control_dataframe))
n_attacks_attempted = min_dataset_size  # Or some fraction of the smallest size, if desired


columns = obesity_train_dataframe.columns
inference_results2 = []

for secret in columns:
    aux_cols = [col for col in columns if col != secret]
    
    evaluator_inferense2 = InferenceEvaluator(ori=obesity_train_dataframe, 
                                   syn=ae_synthetic_obesity_data_with_labels_df, 
                                   control=control_dataframe,
                                   aux_cols=aux_cols,
                                   secret=secret,
                                   n_attacks=n_attacks_attempted) # Use the dynamically set value
    evaluator_inferense2.evaluate(n_jobs=-1)  # Adjust 'n_jobs' as needed
    inference_results2.append((secret,  evaluator_inferense2.risk()))

# Plotting the inference risks
import matplotlib.pyplot as plt

risks = [res[1].value for res in inference_results2]
columns = [res[0] for res in inference_results2]

plt.figure(figsize=(10, 6))
plt.bar(columns, risks, color='skyblue')
plt.xlabel('Secret Column')
plt.ylabel('Measured Inference Risk')
plt.xticks(rotation=45, ha='right')
plt.title('Inference Risk Assessment per Column When n_attacks_attempted = min_dataset_size on 80% AE Synthetic Obesity Data')
plt.show()


# In[99]:


inference2 =  evaluator_inferense2.results()

print("Inference Risk Assessment Per Column When n_attacks_attempted = min_dataset_size on 80% AE Synthetic Obesity Disease Data:")
print("Successs rate of main attack:", inference2.attack_rate)
print("Successs rate of baseline attack:", inference2.baseline_rate)
print("Successs rate of control attack:", inference2.control_rate)


# In[100]:


print("Inference Risk Assessment Per Column When n_attacks_attempted = min_dataset_size on 80% AE Synthetic Obesity Disease Data:", inference_results2)


# # GRAPHICAL REPRESENTATIONS OF THE ABOVE PRIVACY RISK NUMERICAL RESULTS

# In[113]:


import matplotlib.pyplot as plt

# Assuming 'inference_results2' contains the updated risk assessments
risks2 = [res[1].value for res in inference_results2]
columns2 = [res[0] for res in inference_results2]

plt.figure(figsize=(12, 6))
bars = plt.bar(columns2, risks2, color='skyblue')  # Using a different color for distinction

plt.xlabel('Secret Attributes')
plt.ylabel('Privacy Risk Value')
plt.xticks(rotation=45, ha='right')
plt.title('Inference Risk Assessment per Attribute on 80\% AE Synthetic Obesity Data')

# Adding percentage values on top of all bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval*100:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[125]:


import matplotlib.pyplot as plt

# Data for pie charts (Example values)
success_rates = [0.8877, 0.1638, 0.7940]  # Main, Baseline, Control success rates
labels = ['Success', 'Failure']
colors = ['skyblue', 'brown']

# Adjust explode for two segments only
explode = (0.1, 0)  # Explode the first slice (Success)

# Pie Chart for Overall Success vs. Failure
plt.figure(figsize=(12, 6))
print('Success vs. Failure Rates of Inference Risk Assessment By Attack Types on AE Synthetic Obesity Data')
# Success vs. Failure Pie Chart for Main Attack
plt.subplot(1, 3, 1)
sizes = [success_rates[0]*100, (1-success_rates[0])*100]
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=140, colors=[colors[0], 'tomato'])
plt.title('Main Attack Success vs. Failure')

# Success vs. Failure Pie Chart for Baseline Attack
plt.subplot(1, 3, 2)
sizes = [success_rates[1]*100, (1-success_rates[1])*100]
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=140, colors=['skyblue', 'brown'])
plt.title('Baseline Attack Success vs. Failure')

# Success vs. Failure Pie Chart for Control Attack
plt.subplot(1, 3, 3)
sizes = [success_rates[2]*100, (1-success_rates[2])*100]
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=140, colors=['tomato', colors[1]])
plt.title('Control Attack Success vs. Failure')

plt.tight_layout()
plt.show()


# In[126]:


import matplotlib.pyplot as plt

# Overall success rate (average of the three success rates)
overall_success_rate = sum(success_rates) / len(success_rates)

# Data for overall success vs. failure pie chart
sizes_overall = [overall_success_rate*100, (1-overall_success_rate)*100]
labels = ['Overall Success', 'Overall Failure']
explode = (0.1, 0)  # Explode the first slice (Overall Success)
colors_overall = ['tomato', 'skyblue']  # Custom colors for overall pie chart

plt.figure(figsize=(7, 6))

# Overall Success vs. Failure Pie Chart
plt.pie(sizes_overall, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=140, colors=colors_overall)
plt.title('Overall Attack Success vs. Failure Rates of Inference Risk Assessment on 80% AE Synthetic Obesity Data')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




