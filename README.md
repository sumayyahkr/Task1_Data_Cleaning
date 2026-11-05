# Task 1: Data Cleaning and Preprocessing

## Project Overview
This project focuses on cleaning and preprocessing the Titanic dataset using Python. The objective is to prepare raw data for further analysis and machine learning.

## Tools and Libraries
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

## Tasks Performed

1. Loaded the Titanic dataset using Pandas.
2. Explored the dataset using `head()`, `info()`, `describe()`, and `shape`.
3. Identified missing values and calculated missing-value percentages.
4. Removed columns with excessive missing values.
5. Handled missing numerical values using median imputation.
6. Handled missing categorical values using mode imputation.
7. Performed categorical feature encoding.
8. Applied feature scaling using MinMaxScaler and StandardScaler.
9. Detected outliers using the Interquartile Range (IQR) method.
10. Visualized outliers before and after removal.
11. Saved the processed dataset as a CSV file.

## Project Files

| File | Description |
|---|---|
| `task1_data_cleaning.py` | Python source code |
| `titanic.csv` | Original dataset |
| `final_processed_titanic.csv` | Processed dataset |
| `outliers_before.png` | Outlier visualization before removal |
| `outliers_after.png` | Outlier visualization after removal |

## Results

- Original dataset: 891 rows
- Processed dataset: 718 rows
- Rows removed: 173
- Final dataset: 718 rows and 15 columns

## Conclusion

This project demonstrates fundamental data cleaning and preprocessing techniques using Python libraries. It provides practical experience in handling missing values, categorical encoding, feature scaling, and outlier detection.
