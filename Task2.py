import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the dataset
try:
    df = pd.read_csv('Titanic-Dataset.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Titanic-Dataset.csv not found. Please ensure the file is in the correct directory.")
    exit()

# --- 1. Generate summary statistics ---
print("\n--- Descriptive Statistics ---")
print(df.describe())

print("\n--- Information about the DataFrame ---")
df.info()

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Basic imputation for 'Age' and 'Embarked' for visualization purposes
# For 'Age', fill with median
df['Age'].fillna(df['Age'].median(), inplace=True)
# For 'Embarked', fill with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

print("\n--- Missing Values After Imputation (for visualization) ---")
print(df.isnull().sum())

# --- 2. Create histograms and boxplots for numeric features ---
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"\nNumeric features identified: {numeric_features}")

# Remove 'PassengerId' as it's just an identifier and not a true numeric feature for distribution analysis
if 'PassengerId' in numeric_features:
    numeric_features.remove('PassengerId')

print("\n--- Histograms for Numeric Features ---")
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features):
    plt.subplot(len(numeric_features) // 2 + 1, 2, i + 1)
    sns.histplot(df[feature], kde=True)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()

print("\n--- Boxplots for Numeric Features ---")
plt.figure(figsize=(15, 10))
for i, feature in enumerate(numeric_features):
    plt.subplot(len(numeric_features) // 2 + 1, 2, i + 1)
    sns.boxplot(y=df[feature])
    plt.title(f'Boxplot of {feature}')
plt.tight_layout()
plt.show()

# --- Categorical Feature Analysis ---
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical features identified: {categorical_features}")

print("\n--- Count Plots for Categorical Features ---")
plt.figure(figsize=(15, 8))
for i, feature in enumerate(categorical_features):
    plt.subplot(len(categorical_features) // 2 + 1, 2, i + 1)
    sns.countplot(x=df[feature], palette='viridis')
    plt.title(f'Count of {feature}')
plt.tight_layout()
plt.show()

# --- 3. Use pairplot/correlation matrix for feature relationships ---
print("\n--- Correlation Matrix (Numeric Features) ---")
correlation_matrix = df[numeric_features].corr()
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()

# Pairplot for a subset of numeric features (can be slow for many features)
print("\n--- Pairplot for a subset of Numeric Features (Age, Fare, Survived) ---")
# Using a subset for better readability and performance
sns.pairplot(df[['Age', 'Fare', 'Survived']], hue='Survived', palette='coolwarm')
plt.suptitle('Pairplot of Age, Fare, and Survived', y=1.02)
plt.show()

# --- 4. Identify patterns, trends, or anomalies in the data ---
# Example: Survival rate by Sex
print("\n--- Survival Rate by Sex ---")
survival_by_sex = df.groupby('Sex')['Survived'].mean().reset_index()
print(survival_by_sex)
plt.figure(figsize=(6, 4))
sns.barplot(x='Sex', y='Survived', data=survival_by_sex, palette='pastel')
plt.title('Survival Rate by Sex')
plt.show()

# Example: Survival rate by Pclass
print("\n--- Survival Rate by Pclass ---")
survival_by_pclass = df.groupby('Pclass')['Survived'].mean().reset_index()
print(survival_by_pclass)
plt.figure(figsize=(6, 4))
sns.barplot(x='Pclass', y='Survived', data=survival_by_pclass, palette='pastel')
plt.title('Survival Rate by Pclass')
plt.show()

# --- 5. Make basic feature-level inferences from visuals ---
print("\n--- Inferences from EDA ---")
print("1. Age distribution appears slightly right-skewed, with a peak around 20-30 years.")
print("2. Fare distribution is highly right-skewed, indicating many passengers paid low fares, and a few paid very high fares (outliers).")
print("3. 'Survived' is a binary variable (0=No, 1=Yes).")
print("4. 'Sex' has a strong relationship with 'Survived': Females had a significantly higher survival rate than males.")
print("5. 'Pclass' also shows a strong relationship with 'Survived': Passengers in higher classes (1st class) had a much higher survival rate.")
print("6. 'SibSp' and 'Parch' distributions show that most passengers traveled alone or with very few siblings/spouses/parents/children.")
print("7. 'Embarked' shows that most passengers embarked from 'S' (Southampton).")
print("8. The correlation matrix confirms relationships: 'Fare' has a moderate positive correlation with 'Survived', and 'Pclass' has a moderate negative correlation with 'Survived' (higher class number means lower class, so negative correlation with survival makes sense).")

# Interactive plot example using Plotly
print("\n--- Interactive Scatter Plot: Age vs Fare by Survived (Plotly) ---")
fig = px.scatter(df, x="Age", y="Fare", color="Survived",
                 title="Age vs Fare by Survival Status",
                 hover_data=['Name', 'Sex', 'Pclass'])
fig.show()

