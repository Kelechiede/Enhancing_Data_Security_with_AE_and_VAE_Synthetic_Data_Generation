#!/usr/bin/env python
# coding: utf-8

# # STEP 1: Import The Following Relevant Scientific and Computational Libraries for Data Manipulation, Modelling, Interpretation, Visualization ETC.

# In[15]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import Lambda, Input, Dense, Embedding, multiply, Flatten, Concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


# # STEP 2A: DATA EXPLORATORY

# In[3]:


# Assuming your DataFrame is named lowerBackPains_df
# print(cardio_df.describe())

# Load your dataset
# lowerBackPains_df = pd.read_csv('path_to_your_dataset.csv')

# Plotting histograms for each feature
# cardio_df.hist(bins=15, figsize=(15, 10))
# plt.show()

# Count plot for the target variable
# sns.countplot(x='cardio', data=cardio_df)
# plt.title('Distribution of Target Variable')
# plt.show()

# Heatmap of correlations
# plt.figure(figsize=(12, 8))
# sns.heatmap(cardio_df.corr(), annot=True, fmt=".2f")
# plt.title('Correlation Heatmap')
# plt.show()

# Pairplot to visualize relationships between features
# Note: This can be resource-intensive for large datasets
# sns.pairplot(cardio_df, hue='cardio')
# plt.show()



# In[4]:


# Assuming you have loaded your dataset into a DataFrame named lowerBackPains_df
# lowerBackPains_df = pd.read_csv('path_to_your_dataset.csv')

# Separate the features and the target variable
# X = lowerBackPains_df.drop('cardio', axis=1)
# y = lowerBackPains_df['cardio']

# It's a good practice to scale your features, especially when using algorithms that are distance-based
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Apply SMOTE only to training data to avoid information leakage into the test set
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Check the class distribution after applying SMOTE
# print("Class distribution after SMOTE:")
# print(pd.Series(y_train_smote).value_counts())


# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
cardio_df = pd.read_csv('C:\\Users\\Ede\\Desktop\\Synthetic_Real_Data_Using_AE_VAE_Techniques\\master_thesis2024\\cardio.csv')
# lowerBackPains_Data_Path = 'C:\\Users\\Ede\\Desktop\\Synthetic_Real_Data_Using_AE_VAE_Techniques\\cardio.csv'
# Data Preprocessing

## Convert age from days to years
cardio_df['age'] = cardio_df['age'] / 365

## Convert certain features to categorical types if necessary
categorical_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
cardio_df[categorical_cols] = cardio_df[categorical_cols].astype('category')

## Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
cardio_df[scaled_cols] = scaler.fit_transform(cardio_df[scaled_cols])

# Exploratory Data Analysis

## Distribution of each feature
plt.figure(figsize=(20, 10))
for i, col in enumerate(scaled_cols):
    plt.subplot(2, 3, i+1)
    sns.histplot(cardio_df[col], kde=True)
plt.tight_layout()
plt.show()

## Distribution of categorical features
plt.figure(figsize=(20, 10))
for i, col in enumerate(categorical_cols):
    plt.subplot(2, 4, i+1)
    sns.countplot(x=col, data=cardio_df)
plt.tight_layout()
plt.show()

## Target variable analysis
sns.countplot(x='cardio', data=cardio_df)
plt.show()

## Correlation analysis
plt.figure(figsize=(12, 8))
sns.heatmap(cardio_df.corr(), annot=True, fmt=".2f")
plt.show()

## Outliers identification
plt.figure(figsize=(20, 10))
for i, col in enumerate(scaled_cols):
    plt.subplot(2, 3, i+1)
    sns.boxplot(x=cardio_df[col])
plt.tight_layout()
plt.show()


# In[4]:


# Step 1: Basic Statistical Summary
# First, let's look at the basic statistical summary of these features to understand their range, mean, standard deviation, etc.
print("Statistical Summary for ap_hi and ap_lo:")
print(cardio_df[['ap_hi', 'ap_lo']].describe())

# Step 2: Checking for Unique Values
# It's helpful to see the unique values these features hold. This can give us insights into any potential erroneous data entries.
print("Unique values in ap_hi:", cardio_df['ap_hi'].unique())
print("Unique values in ap_lo:", cardio_df['ap_lo'].unique())

# Step 3: Visualizing Distribution
# Visualizing the distribution of these features can help us understand their spread and detect any anomalies.
import matplotlib.pyplot as plt
import seaborn as sns

# Plotting histograms for ap_hi and ap_lo
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(cardio_df['ap_hi'], kde=True, bins=30)
plt.title('Distribution of ap_hi')

plt.subplot(1, 2, 2)
sns.histplot(cardio_df['ap_lo'], kde=True, bins=30)
plt.title('Distribution of ap_lo')
plt.show()

# Step 4: Identifying Outliers
# Box plots can be effective in visualizing outliers.

# Box plots for ap_hi and ap_lo
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(cardio_df['ap_hi'])
plt.title('Box Plot of ap_hi')

plt.subplot(1, 2, 2)
sns.boxplot(cardio_df['ap_lo'])
plt.title('Box Plot of ap_lo')
plt.show()

# Step 5: Checking Value Counts
# Understanding the frequency of different values can reveal any data entry issues.
print("Value counts for ap_hi:")
print(cardio_df['ap_hi'].value_counts().head(10)) # Displaying top 10 most frequent values

print("\nValue counts for ap_lo:")
print(cardio_df['ap_lo'].value_counts().head(10)) # Displaying top 10 most frequent values

# These steps will provide a comprehensive overview of the ap_hi and ap_lo features,
# allowing us to decide if any data cleaning, transformation, or scaling is necessary.
# Keep in mind that the range of normal blood pressure values is well established in medical literature,
# and any significant deviation might indicate data quality issues.


# In[5]:


# Step 1: Identifying Extreme Values
# First, identify extreme values that are biologically implausible for blood pressure readings.
print("Extreme values in ap_hi:", cardio_df[cardio_df['ap_hi'] > 200]['ap_hi'].unique())
print("Extreme values in ap_lo:", cardio_df[cardio_df['ap_lo'] > 200]['ap_lo'].unique())

# Step 2: Filtering the Data
# Consider removing values that are physiologically impossible (e.g., systolic blood pressure above 200 mmHg or diastolic blood pressure above 120 mmHg).
filtered_df = cardio_df[(cardio_df['ap_hi'] > 0) & (cardio_df['ap_hi'] <= 200) & 
                        (cardio_df['ap_lo'] > 0) & (cardio_df['ap_lo'] <= 120)]

# Step 3: Re-evaluating the Distribution
# After filtering, re-examine the distribution of these features.
# Plotting histograms for ap_hi and ap_lo after filtering
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(filtered_df['ap_hi'], kde=True, bins=30)
plt.title('Filtered Distribution of ap_hi')

plt.subplot(1, 2, 2)
sns.histplot(filtered_df['ap_lo'], kde=True, bins=30)
plt.title('Filtered Distribution of ap_lo')
plt.show()

# Step 4: Re-visualizing Outliers
# Check the box plots again after filtering to see if the distribution appears more normal.
# Box plots for ap_hi and ap_lo after filtering
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(filtered_df['ap_hi'])
plt.title('Filtered Box Plot of ap_hi')

plt.subplot(1, 2, 2)
sns.boxplot(filtered_df['ap_lo'])
plt.title('Filtered Box Plot of ap_lo')
plt.show()

# Step 5: Re-assessing the Statistical Summary
# Finally, look at the statistical summary of the filtered data.
print("Statistical Summary for Filtered ap_hi and ap_lo:")
print(filtered_df[['ap_hi', 'ap_lo']].describe())

# These steps should help in understanding the distribution of ap_hi and ap_lo better and ensuring that the values are within a reasonable range for blood pressure readings.
# It's crucial in data preprocessing to ensure that the data you're working with is accurate and reflective of real-world measurements, especially in healthcare-related datasets.
#
#


# In[6]:


# Step 3: Investigate Extreme Values
# Examine the extreme values more closely to understand their impact on the data. 
# Look for values that are several standard deviations away from the mean.
# Identifying extreme values in ap_hi and ap_lo
extreme_values_ap_hi = cardio_df[cardio_df['ap_hi'] > 3]['ap_hi']
extreme_values_ap_lo = cardio_df[cardio_df['ap_lo'] > 3]['ap_lo']

print("Extreme values in ap_hi:", extreme_values_ap_hi.unique())
print("Extreme values in ap_lo:", extreme_values_ap_lo.unique())


# Step 4: Visualizing Distributions with Extreme Values Removed
# If the extreme values are not biologically plausible, 
# consider removing them and re-plotting the distributions.
# Filter out extreme values
filtered_cardio_df = cardio_df[(cardio_df['ap_hi'] <= 3) & (cardio_df['ap_lo'] <= 3)]

# Re-plot histograms
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(filtered_cardio_df['ap_hi'], kde=True)
plt.title('Distribution of ap_hi (Filtered)')

plt.subplot(1, 2, 2)
sns.histplot(filtered_cardio_df['ap_lo'], kde=True)
plt.title('Distribution of ap_lo (Filtered)')
plt.show()

# Step 5: Box Plot Analysis (Post-filtering)
# After filtering, re-examine the box plots to check for outliers.
# Box plots after filtering
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=filtered_cardio_df['ap_hi'])
plt.title('Box Plot of ap_hi (Filtered)')

plt.subplot(1, 2, 2)
sns.boxplot(x=filtered_cardio_df['ap_lo'])
plt.title('Box Plot of ap_lo (Filtered)')
plt.show()

# Step 6: Reassess Statistical Summary
# Finally, reassess the statistical summary of these features after the adjustments.
print("Statistical Summary for Filtered ap_hi and ap_lo:")
print(filtered_cardio_df[['ap_hi', 'ap_lo']].describe())

# These steps should provide a clearer understanding of the ap_hi and ap_lo features, ensuring that their distributions are more representative of typical blood pressure readings.
# This analysis is crucial, especially in a healthcare context, where accurate and realistic data are vital for meaningful insights.
#


# # RE-DATA VISUALIZING AFTER REMOVING OUTLIERS

# In[7]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming cardio_df is your DataFrame after outlier removal

# 1. Data Visualization

# Histogram for ap_hi
plt.figure(figsize=(10, 6))
sns.histplot(cardio_df['ap_hi'], bins=30, kde=True)
plt.title('Distribution of ap_hi after Outlier Removal')
plt.xlabel('ap_hi')
plt.ylabel('Frequency')
plt.show()

# Histogram for ap_lo
plt.figure(figsize=(10, 6))
sns.histplot(cardio_df['ap_lo'], bins=30, kde=True)
plt.title('Distribution of ap_lo after Outlier Removal')
plt.xlabel('ap_lo')
plt.ylabel('Frequency')
plt.show()

# 2. Relationship Analysis

# Scatter plot for ap_hi vs. cardio
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ap_hi', y='cardio', data=cardio_df)
plt.title('ap_hi vs. Cardiovascular Disease')
plt.xlabel('ap_hi')
plt.ylabel('Cardio')
plt.show()

# Scatter plot for ap_lo vs. cardio
plt.figure(figsize=(10, 6))
sns.scatterplot(x='ap_lo', y='cardio', data=cardio_df)
plt.title('ap_lo vs. Cardiovascular Disease')
plt.xlabel('ap_lo')
plt.ylabel('Cardio')
plt.show()

# Correlation analysis
correlation_matrix = cardio_df[['ap_hi', 'ap_lo', 'cardio']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# # STEP 2B: DATA PREPARATION AND PREPROCESSING

# In[10]:


# Data Normalization
# from sklearn.preprocessing import StandardScaler

# Columns to normalize
# columns_to_normalize = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']

# Standard Scaler for normalization
# scaler = StandardScaler()

# Apply normalization on the dataset
# cardio_df[columns_to_normalize] = scaler.fit_transform(cardio_df[columns_to_normalize])

#  Splitting the Dataset
# We'll split the dataset into a training set and a testing set.
# from sklearn.model_selection import train_test_split

# Splitting the dataset into training and testing sets
# X_train2, X_test2 = train_test_split(cardio_df, test_size=0.2, random_state=42)

# Confirm the shape of X_train and X_test
# print("Shape of X_train2:", X_train2.shape)  # Should be (samples, 12)
# print("Shape of X_test2:", X_test2.shape)  # Should be (samples, 12)


# In[6]:


cardio_df.head()


# In[13]:


cardio_df.tail()


# In[14]:


# Check for missing values
cardio_df.isnull().sum()


# In[15]:


# Assuming 'Class_att' is your target column with binary classes 0 and 1
class_counts = cardio_df['cardio'].value_counts()
total_counts = len(cardio_df)

# Calculate class percentages
class_percentages = (class_counts / total_counts) * 100

# Print class percentages
print("Class Percentages:")
print(class_percentages)

# Check if the data is imbalanced
if abs(class_percentages[0] - class_percentages[1]) > 20:  # You can adjust this threshold
    print("\nThe dataset is imbalanced.")
else:
    print("\nThe dataset is balanced.")


# In[11]:


cardio_df.info()


# In[16]:


# Assuming 'Class_att' is your target column
class_counts = cardio_df['cardio'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution in the Dataset')
plt.show()


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

# # Partioning the Original Data () into 80% and 20% respectiively as shown below

# In[8]:


cardiovascular_train_dataframe, control_cardio_dataframe = train_test_split(cardio_df, test_size=0.2, random_state=42)


# In[9]:


cardiovascular_train_dataframe.head()


# In[10]:


cardiovascular_train_dataframe.tail()


# In[11]:


control_cardio_dataframe.head()


# In[12]:


# Assuming 'Class_att' is your target column with binary classes 0 and 1
class_counts = cardiovascular_train_dataframe['cardio'].value_counts()
total_counts = len(cardiovascular_train_dataframe)
print(total_counts) 


# In[13]:


# Verify encoding
cardiovascular_train_dataframe['cardio'].head()


# In[14]:


cardiovascular_train_dataframe.info()


# In[15]:


control_cardio_dataframe.info()


# In[16]:


# Verify encoding
control_cardio_dataframe['cardio'].head()


# # SAVING THE PARTITIONED DATASETS TO CSV FOR FUTURE USE

# In[10]:


control_cardio_dataframe.to_csv(r'C:\Users\Ede\Desktop\Synthetic_Real_Data_Using_AE_VAE_Techniques\master_thesis2024\control_cardio_dataframe.csv', index=False)


# In[11]:


cardiovascular_train_dataframe.to_csv(r'C:\Users\Ede\Desktop\Synthetic_Real_Data_Using_AE_VAE_Techniques\master_thesis2024\cardiovascular_train_dataframe.csv', index=False)


# In[17]:


import matplotlib.pyplot as plt
# Assuming 'Class_att' is your target column
class_counts = cardiovascular_train_dataframe['cardio'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.xlabel('Class')
plt.ylabel('Frequency')
plt.title('Class Distribution in the Original Dataset')
plt.show()


# # STEP 3: DEFINING AND TRAINING AUTO-ENCODER MODEL, AND GENERATE THE RELEVANT SYNTHETIC DATASET THAT MIMICS ORIGINAL DATA

# In[22]:


from sklearn.ensemble import RandomForestClassifier

# Separate features and target
features = cardiovascular_train_dataframe.drop('cardio', axis=1)
labels = cardiovascular_train_dataframe['cardio']

# Apply SMOTE to balance the dataset
# smote = SMOTE(random_state=42)
# features_smote, labels_smote = smote.fit_resample(features, labels)

# Split the balanced dataset into training and testing sets
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(features, labels, test_size=0.2, random_state=42)

# We Normalize features 
scaler = MinMaxScaler()
X_train_orig_scaled = scaler.fit_transform(X_train_orig)
X_test_orig_scaled = scaler.transform(X_test_orig)

# Add noise for autoencoder training
noise_factor = 0.05
X_train_noisy = X_train_orig_scaled + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train_orig_scaled.shape)
X_train_noisy = np.clip(X_train_noisy, 0., 1.)

# Define and compile Autoencoder architecture
input_dim = X_train_orig_scaled.shape[1]

# Encoder
input_layer = Input(shape=(input_dim,))
encoder = Dense(24, activation="relu")(input_layer)
encoder = Dense(12, activation="relu")(encoder)

# Bottleneck
bottleneck = Dense(12, activation="relu")(encoder)

# Decoder
decoder = Dense(12, activation="relu")(bottleneck)
decoder = Dense(24, activation="relu")(decoder)
decoder = Dense(input_dim, activation="sigmoid")(decoder)

# Autoencoder
autoencoder_cardiovascular = Model(inputs=input_layer, outputs=decoder)
autoencoder_cardiovascular.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

# Train the autoencoder with noisy Data
autoencoder_cardiovascular.fit(X_train_noisy, X_train_orig_scaled, epochs=1000, batch_size=256, validation_split=0.2, verbose=2)

# Generate synthetic features from the entire original dataset scaled
synthetic_features_scaled = autoencoder_cardiovascular.predict(scaler.transform(features))

# Ensure synthetic data matches the original data's scale
# # Normalize synthetic data or features to original data range
synthetic_features = scaler.inverse_transform(synthetic_features_scaled)

# Train a classifier on the original dataset
# we'll use a simple RandomForestClassifier, which is a good starting point 
# for many classification tasks due to its versatility and ease of use
classifier = RandomForestClassifier(n_estimators=250, random_state=42)
classifier.fit(X_train_orig, y_train_orig)

# Predict labels for the synthetic features
# After training the classifier on the original data, we'll 
# use it to predict labels for the synthetic data generated from the autoencoder.
synthetic_labels_predicted = classifier.predict(synthetic_features)

# Convert synthetic features to a DataFrame
ae_synthetic_data_df = pd.DataFrame(synthetic_features, columns=features.columns)

# Convert predicted labels into a Series (assuming 'labels' is the name of your target variable)
ae_synthetic_labels_series = pd.Series(synthetic_labels_predicted, name='cardio') # , name='NObeyesdad'

# Example usage
print(ae_synthetic_data_df.head())
print(ae_synthetic_labels_series.head())


# # SAVE THE ABOVE CREATED GENERATIVE AUTOENCODER MODEL

# In[13]:


from tensorflow.keras.models import load_model

# Assume 'autoencoder' is your trained model
autoencoder_cardiovascular.save('autoencoder_cardiovascular.keras')  # Saves the model to an HDF5 file


# In[14]:


from tensorflow.keras.models import load_model

# Assume 'autoencoder' is your trained model
autoencoder_cardiovascular.save('autoencoder_cardiovascular.h5')  # Saves the model to an HDF5 file


# # Join the Generated Synthetic Data and labels

# In[23]:


import pandas as pd

# Join the labels with the synthetic data
ae_synthetic_cardio_data_labels_df = ae_synthetic_data_df.assign(cardio=ae_synthetic_labels_series.values)


# In[24]:


ae_synthetic_cardio_data_labels_df.head()


# # SAVING THE GENERATED AE SYNTHETIC DATASET TO CSV

# In[25]:


# Save the generated synthetic data to a CSV file

ae_synthetic_cardio_data_labels_df.to_csv(r'C:\Users\Ede\Desktop\Synthetic_Real_Data_Using_AE_VAE_Techniques\master_thesis2024\ae_synthetic_cardio_data_labels.csv', index=False)


# In[26]:


# Split the dataset into training and testing sets
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(ae_synthetic_data_df, ae_synthetic_labels_series, test_size=0.2, random_state=42)

# Confirm the shapes
print("X_train_syn shape:", X_train_syn.shape)
print("X_test_syn shape:", X_test_syn.shape)
print("y_train_syn shape:", y_train_syn.shape)
print("y_test_syn shape:", y_test_syn.shape)


# In[23]:


import matplotlib.pyplot as plt
# Assuming 'Class_att' is your target column
class_counts = cardiovascular_train_dataframe['cardio'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.xlabel('Class (Preprocessed Form)')
plt.ylabel('Count')
plt.title('80% Class Distribution of the Original Cardiovascular Disease Dataset')
plt.show()


# In[25]:


import matplotlib.pyplot as plt
# Assuming 'Class_att' is your target column
class_counts = ae_synthetic_cardio_data_labels_df['cardio'].value_counts()

# Create a bar plot
plt.figure(figsize=(8, 6))
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.xlabel('Class of AE Synthetic Dataset')
plt.ylabel('Count')
plt.title('80% Class Distribution of the AE Synthetic Cardiovascular Disease Dataset')
plt.show()


# In[19]:


from sklearn.model_selection import train_test_split

featuresORIG = cardiovascular_train_dataframe.drop('cardio', axis=1)
labelsORIG = cardiovascular_train_dataframe['cardio'] # Class group is (0,1)

# Split the original dataset
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(featuresORIG, labelsORIG, test_size=0.2, random_state=42)

# Confirm the shapes
print("X_train_orig shape:", X_train_orig.shape)
print("X_test_orig shape:", X_test_orig.shape)
print("y_train_orig shape:", y_train_orig.shape)
print("y_test_orig shape:", y_test_orig.shape)


# In[27]:


featuresAE = ae_synthetic_cardio_data_labels_df.drop('cardio', axis=1)
labelsAE = ae_synthetic_cardio_data_labels_df['cardio']

# ae_synthetic_cardio_data_labels_df
# Split the synthetic dataset
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)

# Confirm the shapes
print("X_train_syn_ae shape:", X_train_syn_ae.shape)
print("X_test_syn_ae shape:", X_test_syn_ae.shape)
print("y_train_syn_ae shape:", y_train_syn_ae.shape)
print("y_test_syn_ae shape:", y_test_syn_ae.shape)


# In[61]:


ae_synthetic_cardio_data_labels_df.info()


# In[17]:


featuresCONT = control_cardio_dataframe.drop('cardio', axis=1)
labelsCONT = control_cardio_dataframe['cardio'] # Class group is (0,1)
# Split the synthetic dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Confirm the shapes
print("X_train_cont shape:", X_train_cont.shape)
print("X_test_cont shape:", X_test_cont.shape)
print("y_train_cont shape:", y_train_cont.shape)
print("y_test_cont shape:", y_test_cont.shape)


# In[60]:


control_cardio_dataframe.info()


# In[30]:


# Split the synthetic dataset
X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(ae_synthetic_data_df, ae_synthetic_labels_series, test_size=0.2, random_state=42)

# Confirm the shapes
print("X_train_syn shape:", X_train_syn.shape)
print("X_test_syn shape:", X_test_syn.shape)
print("y_train_syn shape:", y_train_syn.shape)
print("y_test_syn shape:", y_test_syn.shape)


# # MICRO-AVERAGE ROC CURVES

# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Sample data preparation steps (replace with your actual DataFrame names)
# Assuming 'featuresORIG' and 'labelsORIG' are your original dataset features and labels
# Assuming 'featuresCONT' and 'labelsCONT' are your control (or synthetic) dataset features and labels

# Split the original dataset
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(featuresORIG, labelsORIG, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_orig, y_train_orig)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', label='Random Forest ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Orig Cardiovascular Data/Tested with 20% Cont Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Sample data preparation steps (replace with your actual DataFrame names)
# Assuming 'featuresORIG' and 'labelsORIG' are your original dataset features and labels
# Assuming 'featuresCONT' and 'labelsCONT' are your control (or synthetic) dataset features and labels

# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_syn_ae, y_train_syn_ae)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='brown', label='Random Forest ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synt Cardiovascular Data/Tested with 20% Cont Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[39]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Sample data preparation steps (replace with your actual DataFrame names)
# Assuming 'featuresORIG' and 'labelsORIG' are your original dataset features and labels
# Assuming 'featuresCONT' and 'labelsCONT' are your control (or synthetic) dataset features and labels

# Split the original dataset
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(featuresORIG, labelsORIG, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = GradientBoostingClassifier(random_state=42)
classifier.fit(X_train_orig, y_train_orig)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', label='Gradient Boosting ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Orig Cardiovascular Data/Tested with 20% Cont Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[43]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Sample data preparation steps (replace with your actual DataFrame names)
# Assuming 'featuresORIG' and 'labelsORIG' are your original dataset features and labels
# Assuming 'featuresCONT' and 'labelsCONT' are your control (or synthetic) dataset features and labels

# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = GradientBoostingClassifier(random_state=42)
classifier.fit(X_train_syn_ae, y_train_syn_ae)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='brown', label='Gradient Boosting ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synt Cardiovascular Data/Tested with 20% Cont Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[45]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Sample data preparation steps (replace with your actual DataFrame names)
# Assuming 'featuresORIG' and 'labelsORIG' are your original dataset features and labels
# Assuming 'featuresCONT' and 'labelsCONT' are your control (or synthetic) dataset features and labels

# Split the original dataset
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(featuresORIG, labelsORIG, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train_orig, y_train_orig)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', label='DecisionTree Classifier ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Orig Cardiovascular Data/Tested with 20% Cont Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[46]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train_syn_ae, y_train_syn_ae)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='brown', label='DecisionTree Classifier ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synt Cardiovascular Data/Tested with 20% Cont Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[36]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Split the original dataset
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(featuresORIG, labelsORIG, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = LogisticRegression()
classifier.fit(X_train_orig, y_train_orig)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', label='Logistic Regression ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Orig Cardiovascular Data/Tested with 20% Cont Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = LogisticRegression(random_state=42)
classifier.fit(X_train_syn_ae, y_train_syn_ae)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='brown', label='Logistic Regression ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synt Cardiovascular Data/Tested with 20% Cont Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[42]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Split the original dataset
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(featuresORIG, labelsORIG, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = KNeighborsClassifier()
classifier.fit(X_train_orig, y_train_orig)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', label='KNeighborsClassifier ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Orig Cardiovascular Data/Tested with 20% Cont Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = KNeighborsClassifier()
classifier.fit(X_train_syn_ae, y_train_syn_ae)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='brown', label='KNeighborsClassifier ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synt Cardiovascular Data/Tested with 20% Cont Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[51]:


import pandas as pd

# Assuming featuresORIG and featuresCONT are your original and control feature DataFrames
# Convert categorical columns to one-hot encoded columns
featuresORIG_encoded = pd.get_dummies(featuresORIG, drop_first=True)
featuresCONT_encoded = pd.get_dummies(featuresCONT, drop_first=True)

# Ensure both DataFrames have the same columns in the same order
featuresORIG_encoded, featuresCONT_encoded = featuresORIG_encoded.align(featuresCONT_encoded, join='inner', axis=1)

# Now, let's split the datasets again with the encoded features
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(featuresORIG_encoded, labelsORIG, test_size=0.2, random_state=42)
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT_encoded, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the XGBClassifier on the original dataset
classifier = XGBClassifier()
classifier.fit(X_train_orig, y_train_orig)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', label='XGBClassifier ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Orig Cardiovascular Data/Tested with 20% Cont Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[52]:


from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = XGBClassifier()
classifier.fit(X_train_syn_ae, y_train_syn_ae)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='brown', label='XGBClassifier ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synt Cardiovascular Data/Tested with 20% Cont Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[54]:


from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Split the original dataset
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(featuresORIG, labelsORIG, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = LGBMClassifier()
classifier.fit(X_train_orig, y_train_orig)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', label='LGBMClassifier ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Orig Cardiovascular Data/Tested with 20% Cont Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[64]:


control_cardio_dataframe2 = control_cardio_dataframe.copy()

for col in control_cardio_dataframe2.select_dtypes(include=['category']).columns:
    control_cardio_dataframe2[col] = control_cardio_dataframe2[col].cat.codes


featuresCONT2 = control_cardio_dataframe2.drop('cardio', axis=1)
labelsCONT2 = control_cardio_dataframe2['cardio'] # Class group is (0,1)
# Split the synthetic dataset
X_train_cont2, X_test_cont2, y_train_cont2, y_test_cont2 = train_test_split(featuresCONT2, labelsCONT2, test_size=0.2, random_state=42)

# Confirm the shapes
print("X_train_cont2 shape:", X_train_cont2.shape)
print("X_test_cont2 shape:", X_test_cont2.shape)
print("y_train_cont2 shape:", y_train_cont2.shape)
print("y_test_cont2 shape:", y_test_cont2.shape)


# In[66]:


from lightgbm import LGBMClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)


# Assuming X_train_syn_ae, y_train_syn_ae, and X_test_cont, y_test_cont are already defined and split
X_train_cont2, X_test_cont2, y_train_cont2, y_test_cont2 = train_test_split(featuresCONT2, labelsCONT2, test_size=0.2, random_state=42)

# Initialize LGBMClassifier
classifier = LGBMClassifier()

# Fit the classifier to the synthetic AE dataset
classifier.fit(X_train_syn_ae, y_train_syn_ae)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont2)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='brown', label='LGBMClassifier ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synt Cardiovascular Data/Tested with 20% Cont Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[68]:


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Split the original dataset
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(featuresORIG, labelsORIG, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = MLPClassifier()
classifier.fit(X_train_orig, y_train_orig)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', label='MLPClassifier ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Orig Cardiovascular Data/Tested with 20% Cont Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[70]:


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


# And X_train, X_test, y_train, y_test are your 80% Original Obesity dataset split
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier = MLPClassifier()
classifier.fit(X_train_syn_ae, y_train_syn_ae)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='brown', label='MLPClassifier ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synt Cardiovascular Data/Tested with 20% Cont Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# In[71]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Split the original dataset
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(featuresORIG, labelsORIG, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier =  SVC(probability=True)
classifier.fit(X_train_orig, y_train_orig)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='green', label='SVC ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% Orig Cardiovascular Data/Tested with 20% Cont Test Data: (TRTR)')
plt.legend(loc="lower right")
plt.show()


# In[72]:


from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


# And X_train, X_test, y_train, y_test are your 80% Original Cardiovascular dataset split
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)

# Split the control dataset
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)

# Initialize and train the RandomForestClassifier on the original dataset
classifier =  SVC(probability=True)
classifier.fit(X_train_syn_ae, y_train_syn_ae)

# Predict probabilities on the control dataset
y_score = classifier.predict_proba(X_test_cont)

# Compute ROC curve and ROC area for the positive class
fpr, tpr, _ = roc_curve(y_test_cont, y_score[:, 1])  # Assuming the positive class is labeled as '1'
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='brown', label='SVC ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('On 80% AE Synt Cardiovascular Data/Tested with 20% Cont Test Data: (TSTR)')
plt.legend(loc="lower right")
plt.show()


# # Computational and Graphical Representations of AUC-ROC Curves by Classifiers

# In[134]:


# 1. Import Necessary Libraries
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# 2. Define Data Splits for Original Dataset
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(featuresORIG, labelsORIG, test_size=0.2, random_state=42)
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)



# 3. Scale the Data
scaler = StandardScaler()
X_train_orig_scaled = scaler.fit_transform(X_train_orig)
X_test_cont_scaled = scaler.transform(X_test_cont)

# 5. Define the `plot_auc_roc` Function
def plot_auc_roc(model, X_train, y_train, X_test, y_test, data_type):
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_pred_proba = model.decision_function(X_test)
    else:
        raise AttributeError(f"{model.__class__.__name__} does not have predict_proba or decision_function.")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{model.__class__.__name__} ({data_type} data, AUC = {roc_auc:.2f})')

# 6. Define Models
models = {
    "RandomForestClassifier": RandomForestClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(),
    "SVC": SVC(probability=True),
    "XGBClassifier": XGBClassifier(),
    "LGBMClassifier": LGBMClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier()
}

# 7. Plot AUC-ROC Curves for Models Trained on Original Data
plt.figure(figsize=(10, 8))
for name, model in models.items():
    plot_auc_roc(model, X_train_orig_scaled, y_train_orig, X_test_cont_scaled, y_test_cont, "Original Cardiovascular Disease")

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('TRTR Comparison of ROC Curves')
plt.legend(loc="lower right")
plt.show()


# In[28]:


# 1. Import Necessary Libraries
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# 2. Define Data Splits for Original and Control Datasets
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)

# 3. Scale the Data
scaler = StandardScaler()
X_train_syn_ae_scaled = scaler.fit_transform(X_train_syn_ae)
X_test_syn_ae_scaled = scaler.transform(X_test_syn_ae)
X_train_cont_scaled = scaler.transform(X_train_cont)
X_test_cont_scaled = scaler.transform(X_test_cont)

# Assuming featuresAE and labelsAE are your synthetic dataset and its corresponding labels
X_original_scaled = scaler.transform(featuresORIG)

# 4. Split and Scale the Synthetic Dataset
X_train_orig_scaled, X_test_orig_scaled, y_train_orig, y_test_orig = train_test_split(X_original_scaled, labelsORIG, test_size=0.2, random_state=42)


# 5. Define the `plot_auc_roc` Function
def plot_auc_roc(model, X_train, y_train, X_test, y_test, data_type):
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_pred_proba = model.decision_function(X_test)
    else:
        raise AttributeError(f"{model.__class__.__name__} does not have predict_proba or decision_function.")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{model.__class__.__name__} ({data_type} data, AUC = {roc_auc:.2f})')

# 6. Define Models
models = {
    "RandomForestClassifier": RandomForestClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(),
    "SVC": SVC(probability=True),
    "XGBClassifier": XGBClassifier(),
    "LGBMClassifier": LGBMClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier()
}

# 7. Plot AUC-ROC Curves for Models Trained on Synthetic and Real Data
plt.figure(figsize=(10, 8))
for name, model in models.items():
    plot_auc_roc(model, X_train_syn_ae_scaled, y_train_syn_ae, X_test_cont_scaled, y_test_cont, "AE-Synthetic Cardiovascular Disease") #TSTR
    plot_auc_roc(model, X_train_orig_scaled, y_train_orig, X_test_cont_scaled, y_test_cont, "Original Cardiovascular Disease") #TRTR
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC Curves TSTR')
plt.legend(loc="lower right")
plt.show()


# In[30]:


import matplotlib.pyplot as plt
import numpy as np

# Classifier names
classifiers = ["RandomForest", "GradientBoosting", "LogisticRegression", "MLP", "SVC", "XGB", "LGBM", "KNeighbors", "AdaBoost"]

# AUC scores for AE-synthetic data
auc_synthetic = [0.77, 0.77, 0.78, 0.73, 0.75, 0.74, 0.77, 0.71, 0.74]

# AUC scores for original data
auc_original = [0.78, 0.81, 0.79, 0.81, 0.80, 0.81, 0.82, 0.76, 0.81]

x = np.arange(len(classifiers))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots(figsize=(10, 8))
# Customize colors here. Example: 'skyblue' and '#FF5733' (a shade of orange)
rects1 = ax.bar(x - width/2, auc_synthetic, width, label='AE-Synthetic Cardiovascular Disease', color='yellow')
rects2 = ax.bar(x + width/2, auc_original, width, label='Original Cardiovascular Disease', color='brown')

# Add some text for labels, title, and custom x-axis tick labels, etc.
ax.set_xlabel('Classifiers')
ax.set_ylabel('AUC Score')
ax.set_title('AUC Scores by Classifier and 80% Dataset Type')
ax.set_xticks(x)
ax.set_xticklabels(classifiers, rotation=45)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[133]:


# 1. Import Necessary Libraries
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# 2. Define Data Splits for Original and Control Datasets
X_train_cont, X_test_cont, y_train_cont, y_test_cont = train_test_split(featuresCONT, labelsCONT, test_size=0.2, random_state=42)
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)

# 3. Scale the Data
scaler = StandardScaler()
X_train_syn_ae_scaled = scaler.fit_transform(X_train_syn_ae)
X_test_syn_ae_scaled = scaler.transform(X_test_syn_ae)
X_train_cont_scaled = scaler.transform(X_train_cont)
X_test_cont_scaled = scaler.transform(X_test_cont)

# Assuming featuresAE and labelsAE are your synthetic dataset and its corresponding labels
X_original_scaled = scaler.transform(featuresORIG)

# 4. Split and Scale the Synthetic Dataset
X_train_orig_scaled, X_test_orig_scaled, y_train_orig, y_test_orig = train_test_split(X_original_scaled, labelsORIG, test_size=0.2, random_state=42)


# 5. Define the `plot_auc_roc` Function
def plot_auc_roc(model, X_train, y_train, X_test, y_test, data_type):
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_pred_proba = model.decision_function(X_test)
    else:
        raise AttributeError(f"{model.__class__.__name__} does not have predict_proba or decision_function.")
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{model.__class__.__name__} ({data_type} data, AUC = {roc_auc:.2f})')

# 6. Define Models
models = {
    "RandomForestClassifier": RandomForestClassifier(),
    "GradientBoostingClassifier": GradientBoostingClassifier(),
    "LogisticRegression": LogisticRegression(),
    "MLPClassifier": MLPClassifier(),
    "SVC": SVC(probability=True),
    "XGBClassifier": XGBClassifier(),
    "LGBMClassifier": LGBMClassifier(),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "AdaBoostClassifier": AdaBoostClassifier()
}

# 7. Plot AUC-ROC Curves for Models Trained on Synthetic and Real Data
plt.figure(figsize=(10, 8))
for name, model in models.items():
    plot_auc_roc(model, X_train_syn_ae_scaled, y_train_syn_ae, X_test_syn_ae_scaled, y_test_syn_ae, "AE-Synthetic Cardiovascular Disease") #TSTS
    plot_auc_roc(model, X_train_orig_scaled, y_train_orig, X_test_cont_scaled, y_test_cont, "Original Cardiovascular Disease") #TRTR
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Comparison of ROC Curves TSTR')
plt.legend(loc="lower right")
plt.show()


# # COMPUTING CROSS-VALIDATION OF ORIGINAL AND AE-SYNTHETIC CARDIOVASCULAR DISEASE DATASETS

# In[79]:


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


## vae_synthetic_data_df_new
# vae_synthetic_labels_series_new

# ae_synthetic_data_df
# ae_synthetic_labels_series
# X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(featuresAE, labelsAE, test_size=0.2, random_state=42)


# Assuming final_synthetic_df and final_synthetic_labels are your features and labels DataFrames
# Encoding categorical labels if necessary
le = LabelEncoder()
ae_synthetic_labels_ae_encoded = le.fit_transform(ae_synthetic_labels_series)

# Splitting the dataset
X_train_syn_ae, X_test_syn_ae, y_train_syn_ae, y_test_syn_ae = train_test_split(ae_synthetic_data_df, ae_synthetic_labels_ae_encoded, test_size=0.2, random_state=42)

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
    cv_scores = cross_val_score(clf, X_train_syn_ae, y_train_syn_ae, cv=5, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"{clf.__class__.__name__}: Mean CV Accuracy for AE-Synthetic Cardiovascular Disease = {mean_cv_score:.4f}, Std = {std_cv_score:.4f}")


# In[78]:


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

# Convert categorical columns to numeric codes for the entire dataset
featuresORIG_num = featuresORIG.copy()
for col in featuresORIG.select_dtypes(include=['category']).columns:
    featuresORIG_num[col] = featuresORIG[col].cat.codes

# Proceed with your original process by Splitting the original dataset
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(featuresORIG_num, original_labels_ae_encoded, test_size=0.2, random_state=42)


# Assuming final_synthetic_df and final_synthetic_labels are your features and labels DataFrames
# Encoding categorical labels if necessary
le = LabelEncoder()
original_labels_ae_encoded = le.fit_transform(labelsORIG)


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
    cv_scores = cross_val_score(clf, X_train_orig, y_train_orig, cv=5, scoring='accuracy')
    mean_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"{clf.__class__.__name__}: Mean CV Accuracy for 80% Original Cardiovascular Disease Data= {mean_cv_score:.4f}, Std = {std_cv_score:.4f}")


# # GRAPHICAL REPRESENTATION OF THE ABOVE MEAN CV OF ORIGINAL AND SYNTHETIC DATASETS

# In[153]:


import matplotlib.pyplot as plt
import numpy as np

# Classifier names
classifiers = ["DecisionTree", "GradientBoosting", "RandomForest", "AdaBoost", "LGBM", "XGB", "KNeighbors", "LogisticRegression", "SVC", "MLP"]

# Mean CV Accuracy for AE-Synthetic Cardiovascular Disease
ae_synthetic_acc = [0.9133, 0.8772, 0.9418, 0.8453, 0.9224, 0.9355, 0.9218, 0.8110, 0.8489, 0.8815]

# Mean CV Accuracy for 80% Original Cardiovascular Disease Data
original_acc = [0.6349, 0.7338, 0.7143, 0.7279, 0.7352, 0.7293, 0.6395, 0.7158, 0.7291, 0.7324]

x = np.arange(len(classifiers))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 8))
rects1 = ax.bar(x - width/2, ae_synthetic_acc, width, label='AE-Synthetic Cardiovascular Disease', color='brown')
rects2 = ax.bar(x + width/2, original_acc, width, label='Original Cardiovascular Disease Data', color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Classifiers')
ax.set_ylabel('Mean CV Accuracy')
ax.set_title('Mean Cross-Validation Accuracy by Classifier and Dataset Type')
ax.set_xticks(x)
ax.set_xticklabels(classifiers, rotation=45, ha="right")
ax.legend()

# Function to add labels on top of the bars
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()

plt.show()


# # Computing Numerical Statistical Values: KS-Test, P-Value, MSE, RMSE, MAE, F-Test for Variances, T-Test for Means

# In[85]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp, f_oneway, ttest_ind
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np


categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# Initialize Label Encoder
le = LabelEncoder()

# Apply Label Encoding to each categorical column
for col in categorical_columns:
    combined_data = pd.concat([ae_synthetic_cardio_data_labels_df[col], cardiovascular_train_dataframe[col]], axis=0)
    le.fit(combined_data)  # Fit on combined data to ensure all categories are covered
    ae_synthetic_cardio_data_labels_df[col] = le.transform(ae_synthetic_cardio_data_labels_df[col])
    cardiovascular_train_dataframe[col] = le.transform(cardiovascular_train_dataframe[col])

# Assuming we've already split your datasets into features and labels if necessary
# Now we can proceed with statistical analyses

results = []

for column in ae_synthetic_cardio_data_labels_df.columns.drop('cardio'):  # Assuming 'cardio' is the label column
    # Compute KS Test and P-Value
    ks_stat, ks_pvalue = ks_2samp(cardiovascular_train_dataframe[column], ae_synthetic_cardio_data_labels_df[column])

    # Compute MSE, RMSE, and MAE
    mse = mean_squared_error(cardiovascular_train_dataframe[column], ae_synthetic_cardio_data_labels_df[column])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(cardiovascular_train_dataframe[column], ae_synthetic_cardio_data_labels_df[column])

    # Compute F-Test and T-Test for comparing variances and means
    f_stat, f_pvalue = f_oneway(cardiovascular_train_dataframe[column], ae_synthetic_cardio_data_labels_df[column])
    t_stat, t_pvalue = ttest_ind(cardiovascular_train_dataframe[column], ae_synthetic_cardio_data_labels_df[column])

    # Collect results
    results.append({
        'Feature': column,
        'KS Stat': ks_stat,
        'KS P-Value': ks_pvalue,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'F-Test Stat': f_stat,
        'F-Test P-Value': f_pvalue,
        'T-Test Stat': t_stat,
        'T-Test P-Value': t_pvalue,
    })

results_df = pd.DataFrame(results)
print(results_df)


# In[149]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ks_2samp, f_oneway, ttest_ind
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Define datasets
original_data = cardiovascular_train_dataframe
synthetic_data = ae_synthetic_cardio_data_labels_df

# Identify categorical columns
categorical_columns = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']

# Initialize Label Encoder and encode categorical columns
le = LabelEncoder()
for col in categorical_columns:
    combined_data = pd.concat([synthetic_data[col], original_data[col]])
    le.fit(combined_data)
    synthetic_data[col] = le.transform(synthetic_data[col])
    original_data[col] = le.transform(original_data[col])

# Initialize the results list
results = []

# Calculate statistical measures for each feature, excluding the 'cardio' label
for column in synthetic_data.columns.drop('cardio'):
    # KS Test and P-Value
    ks_stat, ks_pvalue = ks_2samp(original_data[column], synthetic_data[column])

    # Mean Squared Error, Root Mean Squared Error, and Mean Absolute Error
    mse = mean_squared_error(original_data[column], synthetic_data[column])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(original_data[column], synthetic_data[column])

    # F-Test and T-Test for comparing variances and means
    f_stat, f_pvalue = f_oneway(original_data[column], synthetic_data[column])
    t_stat, t_pvalue = ttest_ind(original_data[column], synthetic_data[column])

    # Append results
    results.append({
        'Feature': column,
        'Original vs. Synthetic KS Stat': ks_stat,
        'Original vs. Synthetic KS P-Value': ks_pvalue,
        'Original vs. Synthetic MSE': mse,
        'Original vs. Synthetic RMSE': rmse,
        'Original vs. Synthetic MAE': mae,
        'Original vs. Synthetic F-Test Stat': f_stat,
        'Original vs. Synthetic F-Test P-Value': f_pvalue,
        'Original vs. Synthetic T-Test Stat': t_stat,
        'Original vs. Synthetic T-Test P-Value': t_pvalue,
    })

# Create DataFrame to display results
results_df = pd.DataFrame(results)
print(results_df)


# # Computing Graphical Statistical Values: KS-Test, P-Value, MSE, RMSE, MAE, F-Test for Variances, T-Test for Means

# In[122]:


import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'results_df' is the DataFrame with our statistical analysis results
results_df = pd.DataFrame({
    'Feature': ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active'],
    'KS Stat': [0.06225, 0.635661, 0.176054, 0.111018, 0.505768, 0.423911, 0.748375, 0.093232, 0.91125, 0.945161, 0.196232],
    'KS P-Value': [9.098879e-95, 0.0, 0.0, 4.960728e-301, 0.0, 0.0, 0.0, 2.924432e-212, 0.0, 0.0, 0.0]
})

plt.figure(figsize=(10, 8))
plt.barh(results_df['Feature'], results_df['KS Stat'], color='skyblue')
plt.xlabel('KS Stat Value')
plt.ylabel('Feature')
plt.title('KS Stat Values for 80% Original vs AE Synthetic Cardiovascular Disease Data')
plt.grid(axis='x')
plt.tight_layout()
plt.show()


# In[156]:


features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
mse = [3.883827e-03, 2.166981e+05, 2.931338e-01, 9.526891e-02, 4.264288e-02, 2.846829e-01, 3.559789e+07, 2.420648e+05, 7.827768e+08, 9.412467e+08, 7.879198e+06]
rmse = [mse[i]**0.5 for i in range(len(mse))]  # Calculating RMSE based on the MSE values

plt.figure(figsize=(10, 8))
plt.bar(features, mse, color='lightgreen', label='MSE')
plt.bar(features, rmse, color='orange', label='RMSE', alpha=0.5)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Error Value')
plt.title('Error Metrics for 80% Original vs AE Synthetic Cardiovascular Disease Data')
plt.legend()
plt.show()


# In[157]:


f_test_p_value = [8.913405e-01, 5.846648e-20, 1.293148e-01, 1.903612e-63, 1.511010e-01, 4.470833e-12, 0.0, 6.942658e-36, 0.0, 0.0, 0.0]
t_test_p_value = [8.913405e-01, 5.846648e-20, 1.293148e-01, 1.903589e-63, 1.511010e-01, 4.470826e-12, 0.0, 6.942658e-36, 0.0, 0.0, 0.0]

plt.figure(figsize=(10, 7))
plt.plot(features, f_test_p_value, '-o', label='F-Test P-Value', color='red')
plt.plot(features, t_test_p_value, '-s', label='T-Test P-Value', color='blue')
plt.yscale('log')
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('P-Value (Log Scale)')
plt.title('F-Test and T-Test P-Values for 80% Original vs AE Synthetic Cardiovascular Disease Data')
plt.legend()
plt.grid(True)
plt.show()



# # CLASSIFICATION REPORT FOR 80% AE SYNTHETIC AND ORIGINALCARDIOVASCULAR DISEASE DATA

# In[98]:


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = GradientBoostingClassifier(max_depth=1000, n_estimators=1000, random_state=42)
classifier_orig.fit(X_train_orig, y_train_orig)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("GB on 80% Original Cardio/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = GradientBoostingClassifier()
classifier_syn.fit(X_train_syn_ae, y_train_syn_ae)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("GB on 80% AE SyntheticCardio/Tested on 20% Control Data (TSTR):\n", report)


# In[99]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = RandomForestClassifier(max_depth=1000, n_estimators=1000, random_state=42)
classifier_orig.fit(X_train_orig, y_train_orig)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("RF on 80% Original Cardio/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = RandomForestClassifier(max_depth=1000, n_estimators=1000, random_state=42)
classifier_syn.fit(X_train_syn_ae, y_train_syn_ae)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("RF on 80% AE Synthetic Cardio/Tested on 20% Control Data (TSTR):\n", report)


# In[115]:


from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Ensure the 'enable_categorical' parameter is set to True if your XGBoost version supports it.
# This instructs XGBoost to handle categorical features directly.
classifier_orig = XGBClassifier(max_depth=5, n_estimators=100, random_state=42, enable_categorical=True)
classifier_orig.fit(X_train_orig, y_train_orig_encoded)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont)

# Generating the classification report
report = classification_report(y_test_cont_encoded, y_pred, digits=2)
print("XGB on 80% Original Cardio/Tested on 20% Control Data (TRTR):\n", report)

classifier_syn = XGBClassifier(max_depth=5, n_estimators=100, random_state=42, enable_categorical=True)
classifier_syn.fit(X_train_syn_ae, y_train_syn_ae)

# Predicting on the test set for synthetic data
y_pred1 = classifier_syn.predict(X_test_cont)

# Generating the classification report for synthetic data
report = classification_report(y_test_cont_encoded, y_pred1, digits=2)
print("XGB on 80% AE Synthetic Cardio/Tested on 20% Control Data (TSTR):\n", report)


# In[101]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, max_iter=200, random_state=42)
classifier_orig.fit(X_train_orig, y_train_orig)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("MLP on 80% Original Cardio/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = MLPClassifier(hidden_layer_sizes=(200,), activation='relu', solver='adam', alpha=0.0003, max_iter=1000, random_state=42)
classifier_syn.fit(X_train_syn_ae, y_train_syn_ae)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("MLP on 80% AE Synthetic Cardio/Tested on 20% Control Data (TSTR):\n", report)


# In[104]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = DecisionTreeClassifier()
classifier_orig.fit(X_train_orig, y_train_orig)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("DCT on 80% Original Cardio/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = DecisionTreeClassifier()
classifier_syn.fit(X_train_syn_ae, y_train_syn_ae)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("DCT on 80% AE Synthetic Cardio/Tested on 20% Control Data (TSTR):\n", report)


# In[105]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = LogisticRegression(C=1.0, solver='lbfgs', random_state=42, max_iter=1500)
classifier_orig.fit(X_train_orig, y_train_orig)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("LGR on 80% Original Cardio/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = LogisticRegression(C=1.0, solver='lbfgs', random_state=42, max_iter=1500)
classifier_syn.fit(X_train_syn_ae, y_train_syn_ae)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("LGR on 80% AE Synthetic Cardio/Tested on 20% Control Data (TSTR):\n", report)


# In[117]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = KNeighborsClassifier()
classifier_orig.fit(X_train_orig, y_train_orig)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("KNN on 80% Original Cardio/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = KNeighborsClassifier()
classifier_syn.fit(X_train_syn_ae, y_train_syn_ae)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("KNN on 80% AE Synthetic Cardio/Tested on 20% Control Data (TSTR):\n", report)


# In[119]:


from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = LGBMClassifier()
classifier_orig.fit(X_train_orig, y_train_orig)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("LGBM on 80% Original Cardio/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = LGBMClassifier()
classifier_syn.fit(X_train_syn_ae, y_train_syn_ae)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("LGBM on 80% AE Synthetic Cardio/Tested on 20% Control Data (TSTR):\n", report)


# In[ ]:





# In[120]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Adjusting model complexity by limiting the depth
classifier_orig = SVC(probability=True)
classifier_orig.fit(X_train_orig, y_train_orig)

# Predicting on the test set
y_pred = classifier_orig.predict(X_test_cont) # TRTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred, digits=2)
print("SVC on 80% Original Cardio/Tested on 20% Control Data (TRTR):\n", report)


# Adjusting model complexity by limiting the depth
classifier_syn = SVC(probability=True)
classifier_syn.fit(X_train_syn_ae, y_train_syn_ae)


# Predicting on the test set
y_pred1 = classifier_syn.predict(X_test_cont) # TSTR

# Generating the classification report
report = classification_report(y_test_cont, y_pred1, digits=2)
print("SVC on 80% AE Synthetic Cardio/Tested on 20% Control Data (TSTR):\n", report)


# In[158]:


import matplotlib.pyplot as plt
import numpy as np

# Data from the classification reports
classifiers = ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 'MLP', 'KNN', 'SVC', 'DCT']
# Accuracy for TRTR - Original
accuracy_original = [0.72, 0.74, 0.73, 0.75, 0.67, 0.74, 0.63]
# Accuracy for TSTR - AE Synthetic
accuracy_synthetic = [0.73, 0.72, 0.67, 0.71, 0.65, 0.69, 0.63]

x = np.arange(len(classifiers))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 8))
rects1 = ax.bar(x - width/2, accuracy_original, width, label='80% Original Cardio (TRTR)', color='brown')
rects2 = ax.bar(x + width/2, accuracy_synthetic, width, label='80% AE Synthetic Cardio (TSTR)', color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.xlabel('Classifiers')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy in Classification Reports by Classifier and Cardiovascular Disease')  
ax.set_xticks(x)
ax.set_xticklabels(classifiers, rotation=45, ha="right")
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()


# # CORRELATION MATRIX FOR ORIGINAL AND AE SYNTHETIC CARDIOVASCULAR DISEASE

# In[93]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

columns = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke' 'alco', 'active', 'cardio']

def plot_correlation_matrix(data, title):
    # Calculate the correlation matrix
    corr_matrix = data.corr()

    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title(title)
    plt.show()

    
    
# Assuming 'columns' is a list of column names for your original dataset
X_train_orig_df = pd.DataFrame(X_train_orig_scaled, columns=columns)

# Plot correlation matrix for original data
plot_correlation_matrix(X_train_orig_df, "Correlation Matrix of 80% Original Cardiovascular Disease Data")

# Plot correlation matrix for synthetic data   
# plot_correlation_matrix(ae_synthetic_data_df, "Correlation Matrix of AE Synthetic Data")

# Plot correlation matrix for synthetic data   ae_synthetic_data_df
plot_correlation_matrix(ae_synthetic_cardio_data_labels_df, "Correlation Matrix of 80% AE Synthetic Cardiovascular Disease Data")


# # Computing Numerical Correlation Matrices of Original and Synthetic Datasets

# In[94]:


# For the original dataset
print('80% Original Cardiovascular Disease Numerical Correlation Matrix:')
print(X_train_orig_df.corr())

# For the AE synthetic dataset
print('80% AE-Synthetic Cardiovascular Disease Numerical Correlation Matrix:')
print(ae_synthetic_cardio_data_labels_df.corr())


# # BAR GRAPH AND SCATTER PLOTS FOR RESULTS FROM CORRELATION MATRIX

# In[130]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Creating a simplified version of the correlation matrices for illustration purposes
# Original Dataset Correlations
original_corrs = {
    'Feature': ['Age', 'Gender', 'Height', 'Weight', 'AP_HI', 'AP_LO', 'Cholesterol', 'Gluc'],
    'Cardio': [-0.009441, 0.008349, -0.009704, -0.018160, 0.001909, 0.002647, 0.009375, -0.010868]
}

# AE-Synthetic Dataset Correlations
synthetic_corrs = {
    'Feature': ['Age', 'Gender', 'Height', 'Weight', 'AP_HI', 'AP_LO', 'Cholesterol', 'Gluc'],
    'Cardio': [0.530706, -0.041845, -0.051751, 0.289010, 0.037267, 0.045784, 0.407007, 0.205468]
}

original_df = pd.DataFrame(original_corrs)
synthetic_df = pd.DataFrame(synthetic_corrs)

# Plotting the Bar Graph of Key Correlations
plt.figure(figsize=(10, 6))
plt.bar(original_df['Feature'], original_df['Cardio'], width=0.4, label='80%-Original', align='center', color='lightblue')
plt.bar(synthetic_df['Feature'], synthetic_df['Cardio'], width=0.4, label='80% AE-Synthetic', align='edge', color='orange')
plt.xlabel('Feature')
plt.ylabel('Correlation with Cardiovascular Disease')
plt.title('Comparison of Feature Correlations with Cardiovascular Disease')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[128]:


# Choosing a scatter plot for enriching the report, as it can vividly showcase the dynamics of correlation shifts
# between the original and synthetic datasets for each feature with cardiovascular disease.

# Preparing data for the scatter plot
features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
orig_corrs = [1.00, -0.020278, -0.081830, 0.056823, 0.021680, 0.014829, 0.152158, 0.098642, -0.039929, -0.027936, -0.009441]
synth_corrs = [1.00, -0.022764, -0.110796, 0.081774, 0.029132, 0.020848, 0.161116, 0.096139, -0.050787, -0.028871, -0.006088]

# Plotting
plt.figure(figsize=(10, 8))
for i, feature in enumerate(features):
    plt.scatter(orig_corrs[i], synth_corrs[i], label=f'{feature}', s=100)
    
# Plotting a line for perfect agreement
plt.plot([-0.1, 1.0], [-0.1, 1.0], 'r--', lw=2)

plt.title('Comparison of Correlation Coefficients with Cardiovascular Disease')
plt.xlabel('Original Dataset Correlation Coefficients')
plt.ylabel('AE-Synthetic Dataset Correlation Coefficients')
plt.legend(title="Features", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()


# # INSTALL THE PRIVACY ASSESSMENT TOOL KITS KNOWN AS ANONYMETER AS MENTIONED ABOVE IN STEP 2

# In[ ]:


get_ipython().system('pip install anonymeter')


# # IMPORTING THE INSTALLED ANONYMETER'S PRIVACY RISK EVALUATORS FOR PRIVACY PRESERVATION ASSESSMENT ON THE GENERATED SYNTHETIC DATASET
# For more detailed information on the usage of this tool do visit the author's website(blogger) via the links below.
# 
# https://www.anonos.com/blog/presenting-anonymeter-the-tool-for-assessing-privacy-risks-in-synthetic-datasets
# 
# https://github.com/statice/anonymeter

# In[202]:


import pandas as pd
from anonymeter.evaluators import SinglingOutEvaluator, LinkabilityEvaluator, InferenceEvaluator


# In[201]:


# Convert float32 columns to float64 in the synthetic dataset
float_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
ae_synthetic_cardio_data_labels_df[float_cols] = ae_synthetic_cardio_data_labels_df[float_cols].astype('float64')

# Convert categorical columns back to category in the synthetic dataset
category_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']
for col in category_cols:
    # It's important to ensure the category mappings are consistent across datasets
    ae_synthetic_cardio_data_labels_df[col] = ae_synthetic_cardio_data_labels_df[col].astype('category')

# Now, check the info again to ensure the conversion is successful
ae_synthetic_cardio_data_labels_df.info()


# # STEP 3A: PRIVACY RISK ASSESSMENT VIA UNIVARIATE MODE ON 80% AE-SYNTHETIC CARDIO DISEASE DATASET

# In[154]:


singling_out = SinglingOutEvaluator(ori=cardiovascular_train_dataframe, 
                                              syn=ae_synthetic_cardio_data_labels_df, 
                                              control=control_cardio_dataframe,
                                              n_attacks=1500)

try:
    singling_out.evaluate(mode='univariate')  # For univariate analysis
    # For multivariate analysis, you can change mode to 'multivariate'
    singling_out_risk = singling_out.risk()
    print("Singling Out Risk Type via Univariate Analysis When n_attacks=1500 for AE Synthetc Cardiovascular Data:", singling_out_risk)
except RuntimeError as ex:
    print(f"Singling out evaluation failed: {ex}")


# In[155]:


res = singling_out.results()

print("The Singling Out Risk Type via Univariate Analysis When n_attacks=1500 for AE Synthetc Cardiovascular Data:")
print("Successs rate of main attack:", res.attack_rate)
print("Successs rate of baseline attack:", res.baseline_rate)
print("Successs rate of control attack:", res.control_rate)


# In[156]:


singling_out1 = SinglingOutEvaluator(ori=cardiovascular_train_dataframe, 
                                              syn=ae_synthetic_cardio_data_labels_df, 
                                              control=control_cardio_dataframe,
                                              n_attacks=500)

try:
    singling_out1.evaluate(mode='univariate')  # For univariate analysis
    # For multivariate analysis, you can change mode to 'multivariate'
    singling_out_risk = singling_out1.risk()
    print("Singling Out Risk Type via Univariate Analysis When n_attacks=500 for AE Synthetc Cardiovascular Data:", singling_out_risk)
except RuntimeError as ex:
    print(f"Singling out evaluation failed: {ex}")


# In[157]:


res = singling_out1.results()

print("The Singling Out Risk Type via Univariate Analysis When n_attacks=500 for AE Synthetc Cardiovascular Data:")
print("Successs rate of main attack:", res.attack_rate)
print("Successs rate of baseline attack:", res.baseline_rate)
print("Successs rate of control attack:", res.control_rate)


# # GRAPHICAL REPRESENTATIONS OF THE ABOVE PRIVACY RISK NUMERICAL RESULTS

# In[168]:


import matplotlib.pyplot as plt

# Data from the attacks
attacks = ["Main Attack", "Baseline Attack", "Control Attack"]
success_rates_1500 = [0.0013, 0.0013, 0.0013]
success_rates_500 = [0.0038, 0.0038, 0.0038]

# Bar Chart
fig, ax = plt.subplots()
bar_width = 0.35
index = range(len(attacks))

bar1 = ax.bar(index, success_rates_1500, bar_width, label='n_attacks=1500', color='grey')
bar2 = ax.bar([p + bar_width for p in index], success_rates_500, bar_width, label='n_attacks=500', color='green')

# Success Rates by Attack Type and Number of Attacks
ax.set_xlabel('Attack Type')
ax.set_ylabel('Success Rates')
ax.set_title('Univariate Risk Assessment on 80% AE Synthetic Cardiovascular Disease Data')
ax.set_xticks([p + bar_width/2 for p in index])
ax.set_xticklabels(attacks)
ax.legend()
    
ax.bar_label(bar1, padding=3, fmt='%.4f')
ax.bar_label(bar2, padding=3, fmt='%.4f')

plt.tight_layout()
plt.show()


# In[170]:


import matplotlib.pyplot as plt

print('Univariate Risk Assessment Success vs. Failure Rates via 1500 and 500 Attacks on 80% AE Synthetic Cardiovascular Disease')

# Data for 1500 attacks
success_rate_1500 = 0.0013
failure_rate_1500 = 1 - success_rate_1500

# Data for 500 attacks
success_rate_500 = 0.0038
failure_rate_500 = 1 - success_rate_500

# Overall success rate considering all attacks
overall_success_rate = (success_rate_1500 + success_rate_500) / 2
overall_failure_rate = 1 - overall_success_rate

# Custom colors
colors_1500 = ['grey', 'green']  # Custom colors for the 1500 attacks chart
colors_500 = ['coral', 'lightblue']  # Custom colors for the 500 attacks chart
overall_colors = ['gold', 'grey']  # Custom colors for the overall chart

# Pie chart for 1500 attacks
plt.figure(figsize=(12, 9))
plt.subplot(2, 2, 1)
plt.pie([success_rate_1500, failure_rate_1500], labels=['Success', 'Failure'], autopct='%1.1f%%', startangle=90, colors=colors_1500)
plt.title('Success vs Failure Rates for 1500 Attacks')

# Pie chart for 500 attacks
plt.subplot(2, 2, 2)
plt.pie([success_rate_500, failure_rate_500], labels=['Success', 'Failure'], autopct='%1.1f%%', startangle=90, colors=colors_500)
plt.title('Success vs Failure Rates for 500 Attacks')

# Overall pie chart
plt.subplot(2, 1, 2)
plt.pie([overall_success_rate, overall_failure_rate], labels=['Overall Success', 'Overall Failure'], autopct='%1.1f%%', startangle=90, colors=overall_colors)
plt.title('Overall Success vs Failure Rates Across All Attacks')

plt.tight_layout()
plt.show()


# # STEP 3B: PRIVACY RISK ASSESSMENT VIA MULTIVARIATE MODE ON 80% AE-SYNTHETIC CARDIO DISEASE DATASET

# In[158]:


singling_out2 = SinglingOutEvaluator(ori=cardiovascular_train_dataframe, 
                                              syn=ae_synthetic_cardio_data_labels_df, 
                                              control=control_cardio_dataframe,
                                              n_attacks=1500)

try:
    singling_out2.evaluate(mode='multivariate')  # For univariate analysis
    # For multivariate analysis, you can change mode to 'multivariate'
    singling_out_risk = singling_out2.risk()
    print("Singling Out Risk Type via Multivariate Analysis When n_attacks=1500 for AE Synthetc Cardiovascular Data:", singling_out_risk)
except RuntimeError as ex:
    print(f"Singling out evaluation failed: {ex}")


# In[159]:


res = singling_out2.results()

print("The Singling Out Risk Type via Multivariate Analysis When n_attacks=1500 for AE Synthetc Cardiovascular Data:")
print("Successs rate of main attack:", res.attack_rate)
print("Successs rate of baseline attack:", res.baseline_rate)
print("Successs rate of control attack:", res.control_rate)


# In[160]:


singling_out3 = SinglingOutEvaluator(ori=cardiovascular_train_dataframe, 
                                              syn=ae_synthetic_cardio_data_labels_df, 
                                              control=control_cardio_dataframe,
                                              n_attacks=500)

try:
    singling_out3.evaluate(mode='multivariate')  # For univariate analysis
    # For multivariate analysis, you can change mode to 'multivariate'
    singling_out_risk = singling_out3.risk()
    print("Singling Out Risk Type via Multivariate Analysis When n_attacks=500 for AE Synthetc Cardiovascular Data:", singling_out_risk)
except RuntimeError as ex:
    print(f"Singling out evaluation failed: {ex}")


# In[161]:


res = singling_out3.results()

print("The Singling Out Risk Type via Multivariate Analysis When n_attacks=500 for AE Synthetc Cardiovascular Data:")
print("Successs rate of main attack:", res.attack_rate)
print("Successs rate of baseline attack:", res.baseline_rate)
print("Successs rate of control attack:", res.control_rate)


# # GRAPHICAL REPRESENTATIONS OF THE ABOVE PRIVACY RISK NUMERICAL RESULTS

# In[165]:


import matplotlib.pyplot as plt
import numpy as np

# Data setup
labels = ['Main Attack', 'Baseline Attack', 'Control Attack']
attacks_1500 = [0.001277, 0.001942, 0.001277]
attacks_500 = [0.003812, 0.003812, 0.003812]
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, attacks_1500, width, label='n_attacks=1500', color='grey')
rects2 = ax.bar(x + width/2, attacks_500, width, label='n_attacks=500', color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Success Rates')
ax.set_title('Multivariate Risk Assessment on 80% AE Synthetic Cardiovascular Disease Data')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.4f')
ax.bar_label(rects2, padding=3, fmt='%.4f')

plt.show()


# In[177]:


# Defining colors for each pie chart
colors_1500 = ['grey', 'green']  # Custom colors for the 1500 attacks chart
colors_500 = ['coral', 'lightblue']  # Custom colors for the 500 attacks chart
overall_colors = ['gold', 'gray']  # Custom colors for the overall chart

# Data preparation for overall success and failure rates
overall_success = 0.001277215360090744 + 0.0038121702307761206  # Sum of successes
overall_failure = 2 - overall_success  # Subtract from total attempts (2)

# Pie chart data
data_1500 = [0.001277215360090744, 1-0.001277215360090744]
data_500 = [0.0038121702307761206, 1-0.0038121702307761206]
data_overall = [overall_success, overall_failure]

# Labels for pie charts
labels = ['Success', 'Failure']

print('Multivariate Risk Assessment Success vs. Failure Rates via 1500 and 500 Attacks on 80% AE Synthetic Cardiovascular Disease')

# Plotting all pie charts side-by-side
plt.figure(figsize=(12, 9))

# Pie chart for 1500 attacks
plt.subplot(131)
plt.pie(data_1500, labels=labels, colors=colors_1500, autopct='%1.1f%%')
plt.title('1500 Attacks Success vs. Failure')

# Pie chart for 500 attacks
plt.subplot(132)
plt.pie(data_500, labels=labels, colors=colors_500, autopct='%1.1f%%')
plt.title('500 Attacks Success vs. Failure')

# Overall pie chart
plt.subplot(133)
plt.pie(data_overall, labels=labels, colors=colors_overall, autopct='%1.1f%%')
plt.title('Overall Attack Success vs. Failure')

plt.tight_layout()
plt.show()


# In[ ]:





# # STEP 4: PRIVACY RISK ASSESSMENT VIA LINKABILITY ON 80% AE-SYNTHETIC CARDIO DISEASE DATASET

# In[153]:


# Dynamically setting n_attacks based on the smallest dataset size
# Assuming you have similar datasets for original and control in the context of cardiovascular data
min_dataset_size = min(len(cardiovascular_train_dataframe), len(ae_synthetic_cardio_data_labels_df), len(control_cardio_dataframe))
n_attacks = min_dataset_size  # Or some fraction of the smallest size, if desired

# Choosing appropriate columns based on your ae_synthetic_cardio_data_labels_df information
aux_cols = [
    ['gender', 'smoke'],  # Attributes in dataset A
    ['age', 'cholesterol']  # Attributes in dataset B
]

linkability_eval = LinkabilityEvaluator(ori=cardiovascular_train_dataframe, 
                                              syn=ae_synthetic_cardio_data_labels_df, 
                                              control=control_cardio_dataframe,
                                              n_attacks=n_attacks,
                                              aux_cols=aux_cols,
                                              n_neighbors=10)

linkability_eval.evaluate(n_jobs=-1)  # Adjust 'n_jobs' as needed
linkability_risk1 = linkability_eval.risk()
print("Linkability Risk When the n_attacks = the smallest size of dataset and n_neighbors = 10 for AE Synthetic Cardiovascular Data:", linkability_risk1)


# In[163]:


link = linkability_eval.results()

print("Linkability Risk When the n_attacks = the smallest size of dataset and n_neighbors = 10 for AE Synthetic Cardiovascular Data:", linkability_risk1)
print("Successs rate of main attack:", link.attack_rate)
print("Successs rate of baseline attack:", link.baseline_rate)
print("Successs rate of control attack:", link.control_rate)


# In[164]:


# Dynamically setting n_attacks based on the smallest dataset size
# Assuming you have similar datasets for original and control in the context of cardiovascular data
min_dataset_size = min(len(cardiovascular_train_dataframe), len(ae_synthetic_cardio_data_labels_df), len(control_cardio_dataframe))
n_attacks = min_dataset_size  # Or some fraction of the smallest size, if desired

# Choosing appropriate columns based on your ae_synthetic_cardio_data_labels_df information
aux_cols = [
    ['gender', 'smoke'],  # Attributes in dataset A
    ['age', 'cholesterol']  # Attributes in dataset B
]

linkability_eval2 = LinkabilityEvaluator(ori=cardiovascular_train_dataframe, 
                                              syn=ae_synthetic_cardio_data_labels_df, 
                                              control=control_cardio_dataframe,
                                              n_attacks=n_attacks,
                                              aux_cols=aux_cols,
                                              n_neighbors=5)

linkability_eval2.evaluate(n_jobs=-1)  # Adjust 'n_jobs' as needed
linkability_risk1 = linkability_eval2.risk()
print("Linkability Risk When the n_attacks = the smallest size of dataset and n_neighbors = 5 for AE Synthetic Cardiovascular Data:", linkability_risk1)


# In[165]:


link = linkability_eval2.results()

print("Linkability Risk When the n_attacks = the smallest size of dataset and n_neighbors = 5 for AE Synthetic Cardiovascular Data:", linkability_risk1)
print("Successs rate of main attack:", link.attack_rate)
print("Successs rate of baseline attack:", link.baseline_rate)
print("Successs rate of control attack:", link.control_rate)


# # GRAPHICAL REPRESENTATIONS OF THE ABOVE PRIVACY RISK NUMERICAL RESULTS

# In[179]:


import matplotlib.pyplot as plt
import numpy as np

# Success rates data for n_neighbors=10 and n_neighbors=5
categories = ['Main Attack', 'Baseline Attack', 'Control Attack']
success_rates_10 = [0.0020652, 0.0016367, 0.0017082]  # Success rates for n_neighbors=10
success_rates_5 = [0.0004942, 0.0005656, 0.0005656]   # Success rates for n_neighbors=5
failure_rates_10 = [1 - rate for rate in success_rates_10]
failure_rates_5 = [1 - rate for rate in success_rates_5]

x = np.arange(len(categories))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, success_rates_10, width, label='Success n_neighbors=10', color='grey')
rects2 = ax.bar(x + width/2, success_rates_5, width, label='Success n_neighbors=5', color='green')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Success Rates')
ax.set_title('Linkability Risk Assessment on AE Synthetic Cardiovascular Disease Data by Attack Type and Number of Neighbors')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

ax.bar_label(rects1, padding=3, fmt='%.4f')
ax.bar_label(rects2, padding=3, fmt='%.4f')

plt.show()


# In[199]:


import matplotlib.pyplot as plt
print('Linkability Risk Assessment Success/Overall Success vs. Failure Rates via 10 and 5 Neighbors on 80% AE Synthetic Cardiovascular Disease')

# Success rates for 10 neighbors
success_rates_10_neighbors = {
    'Main Attack': 0.0021,
    'Baseline Attack': 0.0016,
    'Control Attack': 0.0017,
    'Failure': 1 - (0.0021 + 0.0016 + 0.0017)
}

# Success rates for 5 neighbors
success_rates_5_neighbors = {
    'Main Attack': 0.00049,
    'Baseline Attack': 0.00056,
    'Control Attack': 0.00056,
    'Failure': 1 - (0.00049 + 0.00056 + 0.00056)
}

# Combined success rates for all neighbors
overall_success_rates = {
    'Main Attack': 0.0013,
    'Baseline Attack': 0.0011,
    'Control Attack': 0.0011,
    'Failure': 1 - (0.0013 + 0.0011 + 0.0011)
}

# Colors for the pie charts
colors = ['tomato', 'green', 'lightblue', 'grey']  # Updated color codes to valid values

fig, axs = plt.subplots(1, 3, figsize=(14, 6))

axs[0].pie(success_rates_10_neighbors.values(), labels=success_rates_10_neighbors.keys(), colors=colors, autopct='%1.1f%%', startangle=140)
axs[0].set_title('10 Neighbors Success vs. Failure Rates')

axs[1].pie(success_rates_5_neighbors.values(), labels=success_rates_5_neighbors.keys(), colors=colors, autopct='%1.1f%%', startangle=140)
axs[1].set_title('5 Neighbors Success vs. Failure Rates')

axs[2].pie(overall_success_rates.values(), labels=overall_success_rates.keys(), colors=colors, autopct='%1.1f%%', startangle=140)
axs[2].set_title('Overall Success vs. Failure Rates')

plt.show()


# # STEP 5: PRIVACY RISK ASSESSMENT VIA INFERENCE PER-COLUMN ON 80% AE-SYNTHETIC CARDIO DISEASE DATASET

# In[214]:


# Dynamically setting n_attacks based on the smallest dataset size
min_dataset_size = min(len(cardiovascular_train_dataframe), len(ae_synthetic_cardio_data_labels_df), len(control_cardio_dataframe))
n_attacks_attempted = min_dataset_size  # Or some fraction of the smallest size, if desired 


columns = cardiovascular_train_dataframe.columns
inference_results4 = []

for secret in columns:
    aux_cols = [col for col in columns if col != secret]
    
    evaluator_inferense4 = InferenceEvaluator(ori=cardiovascular_train_dataframe, 
                                   syn=ae_synthetic_cardio_data_labels_df, 
                                   control=control_cardio_dataframe,
                                   aux_cols=aux_cols,
                                   secret=secret,
                                   n_attacks=n_attacks_attempted) # Use the dynamically set value
    evaluator_inferense4.evaluate(n_jobs=-1)  # Adjust 'n_jobs' as needed
    inference_results4.append((secret,  evaluator_inferense4.risk()))

# Plotting the inference risks
import matplotlib.pyplot as plt

risks = [res[1].value for res in inference_results4]
columns = [res[0] for res in inference_results4]

plt.figure(figsize=(10, 6))
plt.bar(columns, risks, color='skyblue')
plt.xlabel('Secret Column')
plt.ylabel('Measured Inference Risk')
plt.xticks(rotation=45, ha='right')
plt.title('AE Synthetic Cardiovascular Disease Data Inference Risk Assessment per Column')
plt.show()


# In[218]:


print('Inference Risk Assessment on AE Synthetic Cardiovascular Data when n_attempted_attacks = smallest dataset size used:', inference_results4)



# In[219]:


tells = evaluator_inferense4.results()

print("Inference Risk When n_attacks_attempted = min_dataset_size for AE Synthetic Cardiovascular Data:")
print("Successs rate of main attack:", tells.attack_rate)
print("Successs rate of baseline attack:", tells.baseline_rate)
print("Successs rate of control attack:", tells.control_rate)


# # GRAPHICAL REPRESENTATIONS OF THE ABOVE PRIVACY RISK NUMERICAL RESULTS

# In[224]:


import matplotlib.pyplot as plt

# Extracting risk values and columns for plotting
risks = [res[1].value for res in inference_results4]
columns = [res[0] for res in inference_results4]

plt.figure(figsize=(12, 8))
bars = plt.bar(columns, risks, color='grey')

plt.xlabel('Secret Column', fontsize=14)
plt.ylabel('Measured Inference Risk', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.title('Privacy Inference Risk Assessment per Column of AE Synthetic Cardiovascular Disease', fontsize=16)

# Adding the risk value above each bar for clarity
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval,2), va='bottom') # va: vertical alignment

plt.tight_layout()
plt.show()


# In[236]:


import matplotlib.pyplot as plt

# Success rates and failure for each attack based on provided data
attacks_data = {
    'Main Attack': 0.5319,
    'Baseline Attack': 0.4995,
    'Control Attack': 0.5469,
}

# Adding failure rates by calculating the complement to 1 for each
failure_rates = {key: 1 - value for key, value in attacks_data.items()}

# Colors for the pie chart
colors = ['green', 'lightblue', 'lightcoral']

# Create pie charts for each type of attack
fig, axs = plt.subplots(1, 3, figsize=(12, 8))

for i, (key, value) in enumerate(attacks_data.items()):
    axs[i].pie([value, failure_rates[key]], labels=['Success', 'Failure'], colors=[colors[i], 'gray'], autopct='%1.1f%%', startangle=90)
    axs[i].set_title(f'{key} Success vs. Failure')

plt.show()


# In[228]:


# Calculate the overall success rate combining all attacks
overall_success_rate = (0.5319 + 0.4995 + 0.5469) / 3
overall_failure_rate = 1 - overall_success_rate

# Success rates for combined attacks
overall_success = {
    'Success': overall_success_rate,
    'Failure': overall_failure_rate
}

# Define colors for the pie chart
colors = ['lightblue', 'grey']

# Plotting the pie chart for the combined success and failure rates
fig, ax = plt.subplots()
ax.pie(overall_success.values(), labels=overall_success.keys(), colors=colors, autopct='%1.1f%%', startangle=90)
ax.set_title('Overall Success vs. Failure Rates for All Attacks')

plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




