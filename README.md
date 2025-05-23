# Customer Loyalty Organic-Products
This project focuses on identifying the key behavioral and demographic factors that influence customer loyalty to organic products in a UK supermarket. Using data from a customer loyalty program, the goal is to uncover which customer characteristics (such as age, affluence, region, and promotion type) are most associated with repeat purchases of organic goods. The analysis follows the CRISP-DM framework and includes data cleaning, transformation, and exploratory analysis. As a data analyst, my role is to extract meaningful insights that can help the supermarket better target organic product promotions, personalize marketing strategies, and allocate resources more effectively. These findings support smarter decisions in loyalty program management, campaign planning, and customer retention

## Project Overview
- **Objective**: Identify key customer characteristics that predict loyalty to organic products.
- **Business Question**: What customer behaviors and demographics are most predictive of repeat purchases of organic products?
- **Target Variable**: `TargetBuy` (1 = Loyal, 0 = Not Loyal)
- **Framework**: CRISP-DM (Cross-Industry Standard Process for Data Mining)

## Dataset
  
- **Source**: UK supermarket customer loyalty program
- **Rows**: 22,223
- **Features**: 10 predictors including age, affluence, gender, region, promotion class, and spend
- **Rejected Columns**: `ID`, `DemCluster`, `TargetAmt` (not used for this analysis)

## Tools and Technologies

- **Language**: Python
- **Libraries**: pandas, numpy, seaborn, matplotlib, scikit-learn
- **Environment**: Google Colab / Jupyter Notebook
- **Methodology**: CRISP-DM framework for structured analysis

## Methodology (CRISP-DM)

1. **Business Understanding**  
   Defined the importance of organic loyalty and how it supports better marketing decisions.

2. **Data Understanding**  
   Loaded the dataset, reviewed structure and variable types, and visualized `TargetBuy`.

3. **Data Preparation**  
   - Handled missing values using mean/mode imputation
   - Encoded categorical variables using `LabelEncoder`
   - Cleaned and structured the dataset for analysis

4. **Exploratory Data Analysis (EDA)**  
   - Used boxplots and countplots to explore relationships between customer features and loyalty
   - Identified that affluence, age, and promotional spending are strong predictors of organic loyalty

## Key Insights
- Loyal organic shoppers tend to:
  - Have higher affluence scores
  - Spend more on promotions
  - Belong to higher-tier promotion classes (e.g., Gold)
- Promotional engagement and demographic segmentation can be used to enhance targeting
     
5. **Modeling Plan**
   - Proposed classification models (Logistic Regression, Random Forest, XGBoost)
   - Suggested metrics: Accuracy, Recall, F1 Score, and AUC-ROC

```python
# Customer Loyalty Analysis for Organic Products
# Following the CRISP-DM Framework

# ---------------------------------------------
# 1. Data Understanding
# ---------------------------------------------
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import os

# Load dataset
df = pd.read_excel("TopicAorganics.xlsx")

# Display dataset structure
print("Dataset shape:", df.shape)
print(df.info())
print(df.head())

# Target variable visualization
sns.countplot(x='TargetBuy', data=df)
plt.title('Customer Loyalty Distribution')
plt.savefig("images/targetbuy_distribution.png")
plt.clf()

# ---------------------------------------------
# 2. Data Preparation
# ---------------------------------------------

# Drop irrelevant columns
df = df.drop(columns=["ID", "DemCluster", "TargetAmt"])

# Separate predictors and target
X = df.drop(columns=["TargetBuy"])
y = df["TargetBuy"]

# Identify numerical and categorical columns
num_cols = X.select_dtypes(include=["float64", "int64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Impute missing values
num_imputer = SimpleImputer(strategy="mean")
cat_imputer = SimpleImputer(strategy="most_frequent")
X[num_cols] = num_imputer.fit_transform(X[num_cols])
X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

# Encode categorical features
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Final cleaned dataset
df_cleaned = X.copy()
df_cleaned["TargetBuy"] = y

# ---------------------------------------------
# 3. Exploratory Data Analysis (EDA)
# ---------------------------------------------

os.makedirs("images", exist_ok=True)

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='TargetBuy', y=col, data=df_cleaned)
    plt.title(f"{col} vs TargetBuy")
    plt.tight_layout()
    plt.savefig(f"images/{col.lower()}_vs_targetbuy.png")
    plt.clf()

for col in cat_cols:
    plt.figure(figsize=(6, 4))
    sns.countplot(x=col, hue='TargetBuy', data=df_cleaned)
    plt.title(f"{col} by TargetBuy")
    plt.tight_layout()
    plt.savefig(f"images/{col.lower()}_by_targetbuy.png")
    plt.clf()

# ---------------------------------------------
# 4. Modeling Plan (Design Only)
# ---------------------------------------------

# This is a classification problem.
# Suggested models: Logistic Regression, Decision Tree, Random Forest, XGBoost
# Suggested metrics:
# - Accuracy: General performance
# - Recall: Identifies loyal customers well
# - F1 Score: Balances precision and recall
# - AUC-ROC: Measures class separation ability

# ---------------------------------------------
# 5. Evaluation Summary
# ---------------------------------------------

# Summary of EDA:
# - Higher affluence is associated with higher loyalty
# - Promotion class and spend also align with loyalty
# - Age and region show distinguishable trends

# These insights can support targeted promotion strategies and loyalty program planning.

# ---------------------------------------------
# End of Script
# ---------------------------------------------
