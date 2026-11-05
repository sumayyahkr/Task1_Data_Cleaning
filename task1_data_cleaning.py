print("My AI ML Internship - Task 1")
print("Data Cleaning and Preprocessing")
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("titanic.csv")

# Display first five rows
print(df.head())
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Information:")
df.info()
print("\nStatistical Summary:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())
# Missing value analysis
missing_values = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": (df.isnull().sum() / len(df)) * 100
})

print("\nMissing Value Analysis:")
print(missing_values)
# Create a copy of original dataset
clean_df = df.copy()

# Drop columns with excessive missing values
clean_df.drop(columns=['deck', 'embark_town'], inplace=True)

# Fill missing age values with median
clean_df['age'] = clean_df['age'].fillna(clean_df['age'].median())

# Fill missing embarked values with mode
clean_df['embarked'] = clean_df['embarked'].fillna(clean_df['embarked'].mode()[0])

# Check missing values after cleaning
print("\nMissing Values After Cleaning:")
print(clean_df.isnull().sum())

print("\nCleaned Dataset Shape:")
print(clean_df.shape)
# Categorical Encoding

# Remove alive because it duplicates the survived target
encoded_df = clean_df.drop(columns=['alive'])

# Convert categorical columns into numerical columns
encoded_df = pd.get_dummies(
    encoded_df,
    columns=['sex', 'embarked', 'class', 'who'],
    drop_first=True,
    dtype=int
)

# Convert boolean columns into integers
encoded_df['adult_male'] = encoded_df['adult_male'].astype(int)
encoded_df['alone'] = encoded_df['alone'].astype(int)

print("\nDataset After Encoding:")
print(encoded_df.head())

print("\nEncoded Dataset Information:")
print(encoded_df.dtypes)

print("\nEncoded Dataset Shape:")
print(encoded_df.shape)
# Feature Scaling

from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Select numerical features
numeric_features = ['age', 'fare']

# Create a copy
scaled_df = encoded_df.copy()

# Normalization (0 to 1)
minmax_scaler = MinMaxScaler()
scaled_df[['age_normalized', 'fare_normalized']] = (
    minmax_scaler.fit_transform(encoded_df[numeric_features])
)

# Standardization (mean 0, standard deviation 1)
standard_scaler = StandardScaler()
scaled_df[['age_standardized', 'fare_standardized']] = (
    standard_scaler.fit_transform(encoded_df[numeric_features])
)

# Display results
print("\nFeature Scaling Results:")
print(scaled_df[[
    'age', 'age_normalized', 'age_standardized',
    'fare', 'fare_normalized', 'fare_standardized'
]].head())

print("\nScaling completed successfully!")
# Outlier Detection and Removal

# Visualize outliers before removal
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=encoded_df['age'])
plt.title("Age - Before Removal")

plt.subplot(1, 2, 2)
sns.boxplot(y=encoded_df['fare'])
plt.title("Fare - Before Removal")

plt.tight_layout()
plt.savefig("outliers_before.png")
plt.show()

# Calculate IQR limits
outlier_df = encoded_df.copy()

for column in ['age', 'fare']:

    Q1 = outlier_df[column].quantile(0.25)
    Q3 = outlier_df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    print(f"\n{column} outlier limits:")
    print("Lower:", lower_limit)
    print("Upper:", upper_limit)

    outlier_df = outlier_df[
        (outlier_df[column] >= lower_limit) &
        (outlier_df[column] <= upper_limit)
    ]

# Visualize after removal
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
sns.boxplot(y=outlier_df['age'])
plt.title("Age - After Removal")

plt.subplot(1, 2, 2)
sns.boxplot(y=outlier_df['fare'])
plt.title("Fare - After Removal")

plt.tight_layout()
plt.savefig("outliers_after.png")
plt.show()

print("\nOutlier Removal Results:")
print("Original rows:", len(encoded_df))
print("Rows after removal:", len(outlier_df))
print("Rows removed:", len(encoded_df) - len(outlier_df))
# Final feature scaling after outlier removal

from sklearn.preprocessing import StandardScaler, MinMaxScaler

final_df = outlier_df.copy()

scaler = StandardScaler()
final_df[['age', 'fare']] = scaler.fit_transform(
    final_df[['age', 'fare']]
)

# Save final dataset
final_df.to_csv("final_processed_titanic.csv", index=False)

print("\nFinal dataset saved successfully!")
print("Final shape:", final_df.shape)