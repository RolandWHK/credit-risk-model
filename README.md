# 📌 Credit Risk Modeling

The goal of this project is to preprocess banking data then using machine learning models to model PD(Probability of Default), LGD(Loss Given Default) and EAD(Exposure At Default) hence at the end computing EL(Expected Loss) which are key metrics for banks helping them to make money while keeping their business safe.

## 🔍 Project Overview
This notebook implements an end-to-end credit risk decisioning pipeline using historical loan data.
The objective is not just to predict loan default, but to support real lending decisions such as:

  * ✅ Approve
  * 🟡 Manual Review
  * ❌ Reject

The project is designed to closely mirror how risk analytics teams in banks and fintech companies build, evaluate, and operationalize credit risk models — with a strong emphasis on business impact, explainability, and decision-making.

## 🎯 Business Problem
Lending decisions involve asymmetric risk:

* Approving a bad borrower can lead to significant financial loss
* Rejecting a good borrower leads to missed revenue and customer churn

The goal of this project is to:

Estimate the **probability of loan default**, **loss given default**, **exposure at default** and **expected loss** for each applicant and translate that risk into actionable business decisions aligned with risk appetite.

## 💎 Key Design Principles
We know via Basel accords that financial models, especially those used for lending, must be "Glass Box" models rather than "Black Box" models. Hence allowing high interpretability of the model
* **Transparency:** Leveraged **Weight of Evidence (WoE)** for feature engineering to maintain full interpretability of risk drivers.
* **Prudence:** Adopted a **Conservative Risk Stance** by modeling Loss Given Default (LGD) at the point of loan origination.
* **Modularity:** Built using a **Three-Tier Architecture** (Preprocessing -> Ensemble Modeling -> Streamlit UI) for easy maintenance.
* **Accessibility:** Deployed as a **Cloud-Native Application** to bridge the gap between data science and business decision-making.

## 📊 Dataset Overview

The project utilizes the **LendingClub Loan Data (2007-2014)**, a comprehensive peer-to-peer lending dataset that provides a granular view of borrower profiles and loan outcomes.

### 🔍 Data Specifications
* **Source:** LendingClub Open Data / Kaggle.
* **Volume:** ~466,000 loan records across 74 initial features.
* **Target Variables:** * **PD:** Binary indicator of 'Default' or 'Charged-Off' status.
    * **LGD/EAD:** Recovery amounts and outstanding principal at the time of default.


## 📊 Dataset Overview

The project utilizes the **LendingClub Loan Data (2007-2014)**, a comprehensive peer-to-peer lending dataset that provides a granular view of borrower profiles and loan outcomes.

### 🔍 Data Specifications
* **Source:** LendingClub Open Data / Kaggle.
* **Volume:** ~466,000 loan records across 74 initial features.
* **Target Variables:** * **PD:** Binary indicator of 'Default' or 'Charged-Off' status.
    * **LGD/EAD:** Recovery amounts('recovery_rate') and credit conversion factor.

### 🛠 Data Preprocessing & Engineering
To ensure the model adhered to **Basel II** standards of interpretability, the following techniques were applied:

1. **Weight of Evidence (WoE) & Information Value (IV):**
   * Transformed continuous variables (e.g., Annual Income, DTI) into discrete bins.
   * Fine-grained and coarse-grained binning were used to capture non-linear relationships and ensure monotonic trends.
2. **Missing Value Imputation:**
   * Categorical missing values were treated as a separate "Missing" category to maintain data integrity.
   * Strategic imputation for credit-specific variables like `mths_since_last_delinq`.
3. **Feature Selection:**
   * Retained only variables available at the **point of application** for the PD model to simulate a real-world lending environment.

## 🏆 Model Performance

The PD (Probability of Default) model was evaluated using industry-standard metrics for credit risk, focusing on the model's ability to differentiate between "Good" and "Bad" borrowers.

### 📊 Performance Metrics
**PD**
* **AUC-ROC:** **0.70** (Indicates strong discriminatory power across all classification thresholds).
* **Gini Coefficient:** **0.40** (Measures the inequality between the cumulative default rate and the cumulative non-default rate).
* **Kolmogorov-Smirnov (KS) Statistic:** **0.30** (Measures the maximum separation between the distributions of good and bad borrowers).

**LGD**
* **AUC-ROC:** **0.64** (Indicates decent discriminatory power across all classification thresholds)
* **corr:** **0.31** (Demonstrates that the model has identified significant drivers of recovery, providing a statistically sound basis for the Expected Loss (EL) calculation compared to using a simple historical average.)

**EAD**
* **corr:** **0.53** (Indicates a high level of accuracy in predicting the outstanding principal balance at the time of default. )






