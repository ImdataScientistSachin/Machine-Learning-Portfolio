# ================================================================
# EXECUTIVE SUMMARY :  Logistic_Regression_stats_model 
# ================================================================

# This script demonstrates Logistic Regression using the statsmodels library.

# Logistic regression is a statistical method used for binary classification that predicts the probability of a binary outcome (one of two possible outcomes). Unlike linear regression, which predicts continuous values, logistic regression predicts a binary outcome using a logistic function.
#  Binary Outcome: The dependent variable can take only two possible outcomes, often represented as 0 and 1 (e.g., spam vs. not spam, pass vs. fail).
#  Logistic Function (Sigmoid Function): It maps any real-valued number into the interval (0, 1), making it suitable for binary classification.


# Purpose:
# - Logistic Regression is used when the dependent variable is categorical
#   (binary outcome: Yes/No, 0/1).
# - Here, we predict whether a student is admitted based on SAT scores.(Its Uses categorical value as dependent variable )

# Why It Matters (Recruiter-Friendly Narrative):
# - Logistic regression is a foundational classification technique in data science.
# - It is widely applied in admissions, credit scoring, medical diagnosis,
#   and spam detection.
# - Demonstrating mastery of logistic regression shows ability to handle
#   categorical outcomes, interpret statistical models, and communicate results.

# Key Concepts:
# - Binary Outcome: Dependent variable takes values {0,1}.
# - Logistic Function (Sigmoid): Maps real values into (0,1), representing probability.
# - Logit Model: Linear combination of predictors transformed via sigmoid.
# - Model Interpretation: Coefficients indicate how predictors affect log-odds.

# Statistical Literacy:
# - Durbin-Watson: Tests autocorrelation in residuals (not applied here).
# - QQ Plot: Tests normality of residuals (not applied here).
# - Goldfeld-Quandt: Tests heteroscedasticity (not applied here).
# - VIF: Detects multicollinearity (not applied here).
# ================================================================


# ==== Step 1: Import Required Libraries ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')  # Professional plot styling


# ==== Step 2: Load Dataset ====

# Dataset contains SAT scores and admission outcomes (Yes/No).
dataset = pd.read_csv('admitance.csv')
dataset  # Display raw dataset


# ==== Step 3: Convert Categorical to Numerical ====

# Logistic regression requires numerical dependent variables.
# Mapping: Yes → 1, No → 0
dataset['Admitted'] = dataset['Admitted'].map({'Yes': 1, 'No': 0})
dataset  # Display transformed dataset


# ==== Step 4: Define Predictor (X) and Target (y) ====

x = dataset['SAT']       # Independent variable: SAT score
y = dataset['Admitted']  # Dependent variable: Admission outcome


# ==== Step 5: Visualize Distribution ====

# Scatter plot shows relationship between SAT scores and admission.
# Note: Distribution is non-linear, motivating logistic regression.
plt.scatter(dataset['SAT'], dataset['Admitted'])
plt.xlabel('SAT')
plt.ylabel('Admitted')
plt.show()


# ==== Step 6: Build Logistic Regression Model ====
import statsmodels.api as sm

# Add constant term (intercept) to predictor
x1 = sm.add_constant(x)

# Train logistic regression model
model = sm.Logit(y, x1).fit()

# Display model summary (coefficients, p-values, etc.)
model.summary()


# ==== Step 7: Test Model Predictions ====
# Test cases: SAT scores of 1300 and 1850
test = np.array([[1.0, 1300], [1.0, 1850]])  # 1.0 = constant
result = model.predict(test)
result.round(3)  # Probabilities rounded to 3 decimals


# ==== Step 8: Manual Probability Calculation ====

# Logistic regression formula:
# P(Y=1|X) = e^(b0 + b1*X) / (1 + e^(b0 + b1*X))
ret = np.array(np.exp(-69.9128 + 0.0420*1850) /
               (1 + np.exp(-69.9128 + 0.0420*1850)))
ret.round()  # Rounded probability


# ==== Step 9: Automate with Function ====

# Function to compute probability for given SAT score
def f(x, b0, b1):
    return np.array(np.exp(b0 + b1*x) / (1 + np.exp(b0 + b1*x)))

# Apply function to SAT=1300
f(1300, -69.9128, 0.0420)


# ==== Step 10: Apply Function to Dataset ====

result1 = f(x, -69.9128, 0.0420)
result1.round()  # Rounded probabilities

# Sort probabilities and SAT values for plotting
f_sorted = np.sort(f(x, -69.9128, 0.0420))
x_sort = np.sort(x)

# Plot logistic regression curve
plt.scatter(dataset['SAT'], dataset['Admitted'])
plt.plot(x_sort, f_sorted, color='purple')
plt.xlabel('SAT')
plt.ylabel('Admitted')
plt.show()



# ================================================================
# FINAL SUMMARY
# ================================================================
# Key Takeaways:
# - Logistic regression successfully models binary outcomes (admitted vs. not admitted).
# - SAT scores positively influence admission probability.
# - The logistic curve captures non-linear relationship between SAT and admission.
#
# Professional Impact:
# - Demonstrates ability to preprocess categorical data, fit logistic models,
#   and interpret probabilities.
# - Shows statistical literacy and ability to communicate results clearly.
# - Portfolio-ready project for GitHub/LinkedIn showcasing classification skills.
# ================================================================