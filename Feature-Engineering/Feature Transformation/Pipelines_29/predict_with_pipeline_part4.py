#!/usr/bin/env python
# coding: utf-8

# ============================================================
# 📊 Feature Engineering Series
# ============================================================
# Topic: Predicting with a Saved Pipeline
#
# Objective:
# Demonstrate how to load a serialized pipeline object and
# use it to make predictions on new user input.
#
# Why Important:
# - Pipelines combine preprocessing + modeling into one object.
# - Loading a saved pipeline simplifies deployment.
# - Recruiters see practical ML engineering skills in action.
#
# Audience:
# Recruiters, peers, and learners — this script is written
# as a tutorial with clear explanations and professional style.
# ============================================================

# ------------------------------------------------------------
# STEP 1: Import Libraries
# ------------------------------------------------------------
import pickle
import numpy as np
import os

# ------------------------------------------------------------
# STEP 2: Load Saved Pipeline
# ------------------------------------------------------------
# Load pipeline object from file
pipe = pickle.load(open('pipe.pkl', 'rb'))

# Note: 'rb' mode = read binary (required for pickle files)

# ------------------------------------------------------------
# STEP 3: Define User Input
# ------------------------------------------------------------
# Input order: [Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]
test_input2 = np.array([2, 'male', 31.0, 0, 0, 10.5, 'S'], dtype=object).reshape(1, 7)
print("Raw User Input:", test_input2)

# ------------------------------------------------------------
# STEP 4: Make Prediction
# ------------------------------------------------------------
prediction = pipe.predict(test_input2)
print("Prediction:", prediction)

# Interpretation:
# Output indicates whether the passenger survived (1) or not (0).
# In this example, the pipeline predicts survival outcome
# based on the provided features.

# ------------------------------------------------------------
# 📝 Executive-Style Summary
# ------------------------------------------------------------
print("\n============================================================")
print("Executive Summary")
print("============================================================")
print("This script demonstrates how to:")
print("1. Load a saved pipeline object using pickle.")
print("2. Provide new user input in the correct feature order.")
print("3. Generate predictions with preprocessing + model applied automatically.")
print("\nKey Takeaway: Pipelines streamline deployment by combining preprocessing and modeling.")
print("Recruiters see this as evidence of practical ML engineering skills,")
print("attention to detail, and readiness for production workflows.")
print("============================================================")