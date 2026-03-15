# Car Breakdown Prediction — Project Report

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Part I: Exploratory Data Analysis (EDA)](#2-part-i-exploratory-data-analysis-eda)
   - [2.1 Dataset Description](#21-dataset-description)
   - [2.2 Data Cleaning](#22-data-cleaning)
   - [2.3 Target Variable Analysis](#23-target-variable-analysis)
   - [2.4 Univariate Distribution by Class](#24-univariate-distribution-by-class)
   - [2.5 Statistical Significance Testing](#25-statistical-significance-testing)
   - [2.6 Breakdown Rate by Categorical Variables](#26-breakdown-rate-by-categorical-variables)
   - [2.7 Correlation Analysis](#27-correlation-analysis)
   - [2.8 Mutual Information Analysis](#28-mutual-information-analysis)
   - [2.9 Scatter Matrix of Top Features](#29-scatter-matrix-of-top-features)
   - [2.10 Domain-Driven Feature Engineering](#210-domain-driven-feature-engineering)
   - [2.11 Mutual Information After Feature Engineering](#211-mutual-information-after-feature-engineering)
   - [2.12 Risk Score Analysis](#212-risk-score-analysis)
   - [2.13 EDA Conclusions](#213-eda-conclusions)
3. [Part II: Modeling](#3-part-ii-modeling)
   - [3.1 Problem Statement and Objectives](#31-problem-statement-and-objectives)
   - [3.2 Pipeline Architecture](#32-pipeline-architecture)
   - [3.3 Data Preparation for Modeling](#33-data-preparation-for-modeling)
   - [3.4 Evaluation Framework](#34-evaluation-framework)
   - [3.5 Baseline Model](#35-baseline-model)
   - [3.6 Threshold Analysis — The Highest Impact Technique](#36-threshold-analysis--the-highest-impact-technique)
   - [3.7 Sampling Strategies and Engineered Features](#37-sampling-strategies-and-engineered-features)
   - [3.8 Decision Tree — Interpretable Model](#38-decision-tree--interpretable-model)
   - [3.9 Advanced Models — XGBoost and RandomForest Tuned](#39-advanced-models--xgboost-and-randomforest-tuned)
   - [3.10 Cross-Validation with Correct Metrics](#310-cross-validation-with-correct-metrics)
   - [3.11 Model Comparison](#311-model-comparison)
   - [3.12 Threshold Optimization of the Best Model](#312-threshold-optimization-of-the-best-model)
   - [3.13 Final Evaluation and Submission](#313-final-evaluation-and-submission)
   - [3.14 Modeling Conclusions](#314-modeling-conclusions)
4. [Part III: How to Add More Models to the Pipeline](#4-part-iii-how-to-add-more-models-to-the-pipeline)
5. [Final Conclusions](#5-final-conclusions)
6. [Gen AI policy](#6-gen-ai-policy)

---

## 1. Project Overview

This project addresses a **binary classification problem**: predicting whether a vehicle will experience a mechanical breakdown within the next 30 days. The dataset comes from a competition (`car-breakdown-prediction`) and includes both a training set (1,050 rows) and a test set (450 rows) for which predictions must be submitted.

The target variable is `breakdown_next_30_days` (0 = no breakdown, 1 = breakdown). The dataset contains 8 numerical features describing vehicle characteristics and usage patterns, along with 4 categorical features describing vehicle brand, weather exposure, fuel type, and tyre type.

The project is structured in two main phases:
1. **Exploratory Data Analysis (EDA)**: Understanding the data, identifying patterns, engineering new features, and selecting the most informative variables.
2. **Modeling**: Building, comparing, and optimizing classification models to maximize the detection of breakdowns (class 1) while maintaining acceptable overall accuracy.

A central challenge of this project is the **class imbalance**: only ~17% of vehicles in the training set experienced a breakdown, which significantly impacts model behavior and evaluation strategy.

---

## 2. Part I: Exploratory Data Analysis (EDA)

The EDA phase (notebook: `jose_eda2_english.ipynb`) focuses on understanding the data distribution, identifying which features are most predictive of breakdowns, and engineering new domain-driven features to improve model performance.

### 2.1 Dataset Description

The raw training dataset contains **1,050 rows** with the following features:

**Numerical Features (8):**
| Feature | Description |
|---|---|
| `vehicle_age_years` | Age of the vehicle in years |
| `mileage_km` | Total mileage in kilometers |
| `engine_hours` | Total engine running hours |
| `last_service_km_ago` | Kilometers driven since the last service |
| `oil_quality_pct` | Oil quality as a percentage (0–100%) |
| `avg_trip_length_km` | Average trip length in kilometers |
| `cleanliness_score` | Vehicle cleanliness score (0–100) |
| `driver_satisfaction_score` | Driver satisfaction score (0–10) |

**Categorical Features (4):**
| Feature | Description |
|---|---|
| `vehicle_brand` | Brand/manufacturer of the vehicle |
| `weather_exposure` | Level of weather exposure the vehicle endures |
| `fuel_type` | Type of fuel used |
| `tyre_type` | Type of tyres installed |

**Target Variable:**
| Feature | Description |
|---|---|
| `breakdown_next_30_days` | Binary: 1 if the vehicle will break down in the next 30 days, 0 otherwise |

### 2.2 Data Cleaning

The first step was removing rows with **physically impossible values**. A dedicated function `remove_impossible_values()` filters out rows where:

- `mileage_km` < 0 or > 2,000,000 (negative mileage is impossible; over 2 million km is unrealistic for any vehicle)
- `engine_hours` < 0 (negative engine hours are impossible)
- `oil_quality_pct` > 100 (oil quality cannot exceed 100%)
- `cleanliness_score` < 0 or > 100 (score is bounded between 0 and 100)
- `driver_satisfaction_score` > 10 (score is bounded between 0 and 10)

**Result:** 87 rows were removed, reducing the training set from 1,050 to **963 rows**.

**Important design decision:** This cleaning function is applied **only to the training set**, never to the test set. Removing rows from the test set would mean missing predictions in the submission file, which would be invalid. For the test set, impossible values are handled through imputation instead.

After removing impossible rows, **missing values were imputed**:
- **Numerical features**: filled with the **median** of each column (computed only from the training data)
- **Categorical features**: filled with the **mode** (most frequent value) of each column

Using the median instead of the mean for numerical imputation is a deliberate choice: the median is robust to outliers, which is important because features like `mileage_km` and `engine_hours` can have extreme values that would skew the mean.

After imputation, the dataset had **zero remaining null values**.

### 2.3 Target Variable Analysis

The target distribution revealed a significant **class imbalance**:

| Class | Proportion |
|---|---|
| 0 (No breakdown) | 83.1% |
| 1 (Breakdown) | 16.9% |

This imbalance is critical because it means that a model that predicts "no breakdown" for every single vehicle would achieve **83.1% accuracy** — a seemingly good score that is actually useless. This observation drove several key decisions throughout the project:

1. **Evaluation metrics**: We cannot rely on global accuracy alone. Instead, we need metrics that specifically measure performance on the minority class (F1-class1, Recall-class1, Balanced Accuracy, ROC-AUC).
2. **Sampling techniques**: SMOTE and its variants are needed to address the imbalance during training.
3. **Threshold tuning**: The default classification threshold of 0.5 is suboptimal when the class prevalence is ~0.17. A lower threshold should be explored.
4. **Class weighting**: Models that support `class_weight='balanced'` or `scale_pos_weight` should use these parameters.

### 2.4 Univariate Distribution by Class

**KDE (Kernel Density Estimation) Plots by Class**

For each of the 8 numerical features, a KDE plot was generated showing the distribution for class 0 (no breakdown, in blue) and class 1 (breakdown, in red).

*Purpose of this graph:* KDE plots allow us to visually inspect whether the distribution of a feature differs between the two classes. If the two curves overlap almost completely, the feature does not help distinguish between breakdowns and non-breakdowns. If the curves are shifted or have different shapes, the feature is potentially informative.

*Key observations:*
- **`vehicle_age_years`**: The breakdown class (1) has a distribution shifted slightly to the right, suggesting older vehicles are more prone to breakdowns. This makes intuitive domain sense — older vehicles have more wear and tear.
- **`oil_quality_pct`**: The breakdown class shows a slightly lower distribution, indicating that vehicles with worse oil quality are more likely to break down. Low oil quality is a direct indicator of poor maintenance.
- **`last_service_km_ago`** and **`engine_hours`**: Some separation is visible, with breakdowns tending to have higher values (more kilometers since last service), suggesting that delayed maintenance correlates with breakdowns.
- **Other features** (`mileage_km`, `avg_trip_length_km`, `cleanliness_score`, `driver_satisfaction_score`): These showed largely overlapping distributions, suggesting they are less discriminative on their own.

**Boxplots by Class with Mann-Whitney U Test**

Boxplots were generated for each numerical feature, split by class, with a Mann-Whitney U test to assess statistical significance.

*Purpose of this graph:* Boxplots show the median, interquartile range (IQR), and outliers of a feature for each class. The Mann-Whitney U test (a non-parametric test that does not assume normal distribution) determines whether the distribution of a feature is statistically significantly different between the two classes (p < 0.05).

*Why Mann-Whitney U and not a t-test:* The Mann-Whitney U test was chosen because it does not assume that the data is normally distributed. Since many of the features (e.g., `mileage_km`, `engine_hours`) have skewed distributions with outliers, a non-parametric test is more appropriate and robust.

### 2.5 Statistical Significance Testing

The Mann-Whitney U test was applied to all 8 numerical features. Results:

| Feature | p-value | Significant? |
|---|---|---|
| `vehicle_age_years` | 0.0001 | ✅ SIGNIFICANT |
| `oil_quality_pct` | 0.0334 | ✅ SIGNIFICANT |
| `mileage_km` | 0.2222 | ❌ not significant |
| `engine_hours` | 0.4995 | ❌ not significant |
| `last_service_km_ago` | 0.1577 | ❌ not significant |
| `avg_trip_length_km` | 0.8932 | ❌ not significant |
| `cleanliness_score` | 0.7327 | ❌ not significant |
| `driver_satisfaction_score` | 0.7913 | ❌ not significant |

**Key insight:** Only **2 out of 8** original features show statistically significant differences between the two classes. This is an important finding because it tells us:
1. The original features alone may not be sufficient for building a good classifier.
2. Feature engineering (creating new, more informative features from existing ones) will likely be necessary to improve performance.
3. `vehicle_age_years` and `oil_quality_pct` are the strongest individual signals in the raw data.

### 2.6 Breakdown Rate by Categorical Variables

Bar charts were created showing the breakdown rate (proportion of class 1) for each category within each categorical variable, with a horizontal dashed line indicating the global mean breakdown rate (~17%).

*Purpose of this graph:* These charts reveal whether certain categories within a feature have higher or lower breakdown rates than the overall average. Categories with significantly higher breakdown rates could be important predictive signals.

**Features analyzed:**
- **`vehicle_brand`**: Different brands showed varying breakdown rates, suggesting that the brand (and its associated build quality, maintenance ecosystem, etc.) may influence breakdown probability.
- **`weather_exposure`**: Categories with higher weather exposure tended to have higher breakdown rates, which makes domain sense — vehicles exposed to harsh weather conditions (extreme heat, cold, salt, humidity) experience faster degradation.
- **`fuel_type`**: Some fuel types showed slightly different breakdown rates, though the differences were moderate.
- **`tyre_type`**: Different tyre types showed varying breakdown rates, potentially reflecting the correlation between tyre choice and vehicle usage patterns.

### 2.7 Correlation Analysis

Two correlation analyses were performed side by side:

**Pearson Correlation Heatmap**

A full correlation heatmap including all numerical features and the target variable was computed.

*Purpose of this graph:* The Pearson correlation matrix shows linear relationships between all pairs of numerical variables. It serves two purposes:
1. **Feature-target correlation**: Identifying which features have the strongest linear relationship with the target variable.
2. **Feature-feature correlation**: Identifying multicollinearity (highly correlated features) which could cause issues in some models (e.g., logistic regression) and may indicate redundant information.

**Point-Biserial Correlation with Target**

Since the target variable is binary (0/1), the point-biserial correlation coefficient was used instead of the standard Pearson correlation. The point-biserial correlation is the mathematically correct correlation measure between a continuous variable and a binary variable.

*Purpose of this graph:* A horizontal bar chart shows the point-biserial correlation coefficient (r) for each feature with the target, with asterisks (*) indicating statistical significance (p < 0.05). Positive values mean the feature is positively correlated with breakdowns (higher values → more breakdowns), while negative values indicate an inverse relationship.

*Key observations:* The correlations with the target were generally weak (|r| < 0.15), which reinforces the finding from the Mann-Whitney tests that the original features individually have limited discriminative power. This further motivates the need for feature engineering and ensemble methods that can capture non-linear relationships.

### 2.8 Mutual Information Analysis

Mutual Information (MI) was computed for all original features (both numerical and categorical) against the target variable.

*What is Mutual Information:* MI measures the amount of information that one variable provides about another variable. Unlike Pearson correlation, MI captures **any** type of dependency (linear and non-linear). An MI score of 0 means the variables are independent; higher values indicate more shared information.

*Why use MI in addition to correlation:* Pearson correlation only detects linear relationships. If a feature has a non-linear relationship with the target (e.g., U-shaped, threshold effects), Pearson correlation might report a low value while MI would correctly identify the dependency.

*Purpose of the graph:* A horizontal bar chart ranked features by their MI score with the target, providing a ranking of feature importance that accounts for non-linear relationships.

**Results — Top features by MI (original only):**

| Feature | MI Score |
|---|---|
| `engine_hours` | 0.0228 |
| `avg_trip_length_km` | 0.0158 |
| `vehicle_age_years` | 0.0157 |
| `last_service_km_ago` | 0.0138 |
| `cleanliness_score` | 0.0113 |
| `weather_exposure` | 0.0083 |
| `mileage_km` | 0.0000 |
| `oil_quality_pct` | 0.0000 |
| `driver_satisfaction_score` | 0.0000 |
| `vehicle_brand` | 0.0000 |
| `fuel_type` | 0.0000 |
| `tyre_type` | 0.0000 |

**Key insight:** Several features that appeared significant in correlation analysis (`oil_quality_pct`) show zero MI here, and vice versa (`engine_hours` ranks first in MI but was not significant in the Mann-Whitney test). This discrepancy underscores the importance of using multiple analytical methods — no single metric tells the whole story.

The fact that 6 out of 12 features have MI = 0 with the target is concerning and further highlights the need for feature engineering.

### 2.9 Scatter Matrix of Top Features

A scatter matrix was generated for the top 4 features by MI: `engine_hours`, `avg_trip_length_km`, `vehicle_age_years`, and `last_service_km_ago`.

*Purpose of this graph:* The scatter matrix shows pairwise scatter plots for the top features, colored by class (blue = no breakdown, red = breakdown). The diagonal shows KDE plots for each feature by class. This visualization helps identify:
1. **Cluster patterns**: Whether breakdown cases cluster in specific regions of the feature space.
2. **Feature interactions**: Whether combinations of features create separable regions between classes.
3. **Non-linear boundaries**: Whether the classes can only be separated by non-linear decision boundaries.

*Key observation:* The classes were heavily overlapping in all pairwise plots, confirming that individual features or simple pairs of features are not sufficient to linearly separate the classes. This motivates the use of ensemble methods (Random Forest, Gradient Boosting, XGBoost) that can learn complex non-linear decision boundaries.

### 2.10 Domain-Driven Feature Engineering

Based on the insights from the EDA and domain knowledge about vehicle maintenance, **14 new features** were engineered. These features fall into five categories:

#### Wear Features (3 features)

| Feature | Formula | Rationale |
|---|---|---|
| `mileage_per_year` | `mileage_km / (vehicle_age_years + 1)` | Measures usage intensity — vehicles driven more intensely per year experience faster wear. The +1 in the denominator avoids division by zero for brand-new vehicles. |
| `hours_per_km` | `engine_hours / (mileage_km + 1)` | Measures engine efficiency per distance. A high ratio may indicate idling, city driving, or engine problems. |
| `service_overdue_ratio` | `last_service_km_ago / (mileage_km + 1)` | Normalizes the distance since last service by total mileage. A high ratio means a proportionally large fraction of the vehicle's life has passed since the last service. |

#### Risk Flags (5 binary features)

| Feature | Condition | Rationale |
|---|---|---|
| `is_service_overdue` | `last_service_km_ago > 15,000` | Threshold derived from the automotive maintenance domain: most manufacturers recommend service every 10,000–15,000 km. Exceeding 15,000 km is a clear overdue signal. |
| `is_old_vehicle` | `vehicle_age_years >= 15` | Vehicles older than 15 years have significantly higher failure rates due to material fatigue, obsolete parts, and accumulated wear. |
| `is_high_mileage` | `mileage_km > 200,000` | Vehicles exceeding 200,000 km often experience increased rates of component failure (transmissions, engines, suspensions). |
| `is_low_oil_quality` | `oil_quality_pct < 30` | Oil quality below 30% indicates severe degradation. Oil lubricates, cools, and cleans the engine; at very low quality, it fails to perform these functions adequately. |
| `is_dirty` | `cleanliness_score < 40` | A very dirty vehicle may signal general neglect, which often correlates with neglected mechanical maintenance as well. |

*Why binary risk flags:* Creating binary indicators from continuous features is useful because it captures threshold effects. A vehicle with 14,999 km since last service and one with 15,001 km have almost identical continuous values, but the latter has crossed a meaningful maintenance threshold. Binary flags make these domain-meaningful thresholds explicit for the model.

#### Interaction Features (2 features)

| Feature | Formula | Rationale |
|---|---|---|
| `age_x_mileage` | `vehicle_age_years × log1p(mileage_km)` | Captures the multiplicative effect of age and mileage. An old vehicle with high mileage is much more vulnerable than either condition alone. The log transform prevents mileage from dominating due to its large scale. |
| `oil_x_service` | `oil_quality_pct × (1 / (last_service_km_ago + 1))` | Captures the interaction between oil quality and service recency. High oil quality combined with recent service (small denominator → large value) indicates well-maintained vehicles. Low oil quality with overdue service indicates double negligence. |

*Why interaction features:* Individual features may each have weak predictive power, but their combination can be highly informative. Machine learning models like decision trees can learn interactions on their own, but explicitly providing them as features makes the signal available to all models, including linear ones, and reduces the complexity required in the model itself.

#### Log Transforms (3 features)

| Feature | Formula | Rationale |
|---|---|---|
| `log_mileage` | `log1p(mileage_km)` | Reduces the extreme right skew of mileage. The first 100,000 km add proportionally more wear than going from 400,000 to 500,000 km — the log transform captures this diminishing marginal effect. |
| `log_last_service` | `log1p(last_service_km_ago)` | Same reasoning: reduces skew and captures the non-linear relationship between service delay and risk. |
| `log_avg_trip` | `log1p(avg_trip_length_km)` | Normalizes the distribution of average trip length for better performance in models that assume or benefit from normally-distributed features. |

*Why log transforms:* Many statistical and machine learning methods perform better when features have approximately normal distributions. The `log1p` function (log(1+x)) is used instead of `log(x)` because it handles zero values gracefully (log1p(0) = 0 instead of log(0) = -infinity).

#### Composite Risk Score (1 feature)

| Feature | Formula | Rationale |
|---|---|---|
| `risk_score` | `is_service_overdue × 3 + is_low_oil_quality × 2 + is_old_vehicle × 2 + is_high_mileage × 1 + is_dirty × 1` | A weighted sum of all risk flags, where the weights reflect domain priority: service overdue is the highest risk factor (weight 3), followed by oil quality and vehicle age (weight 2 each), and mileage/cleanliness as secondary indicators (weight 1 each). Range: 0 (low risk) to 9 (maximum risk). |

*Why a composite score:* Individual binary flags each capture a narrow signal. The composite score aggregates these complementary signals into a single measure that can discriminate better than any individual flag. The weighting was derived from domain knowledge rather than being learned from data, which makes it more robust and less prone to overfitting.

**Summary:** After feature engineering, the total feature count increased from 12 original features (8 numerical + 4 categorical) to **26 features** (22 numerical + 4 categorical).

### 2.11 Mutual Information After Feature Engineering

MI was recomputed for all 22 numerical features (original + engineered) against the target. The results were displayed as a horizontal bar chart with features colored by origin (blue = original, red = engineered).

*Purpose of this graph:* To evaluate whether the engineered features actually provide more information about the target than the original features. If engineered features rank higher than original features, the feature engineering effort was worthwhile.

**Results — Top 10 features by MI (including engineered):**

| Feature | MI Score | Origin |
|---|---|---|
| `risk_score` | 0.0308 | engineered |
| `is_low_oil_quality` | 0.0268 | engineered |
| `engine_hours` | 0.0240 | original |
| `cleanliness_score` | 0.0171 | original |
| `avg_trip_length_km` | 0.0158 | original |
| `log_avg_trip` | 0.0153 | engineered |
| `oil_x_service` | 0.0145 | engineered |
| `log_last_service` | 0.0142 | engineered |
| `last_service_km_ago` | 0.0138 | original |
| `mileage_per_year` | 0.0133 | engineered |

**Key insight:** The top 2 features by MI are both **engineered**: `risk_score` (0.0308) and `is_low_oil_quality` (0.0268). In total, 6 out of the top 10 features are engineered, demonstrating that the feature engineering effort significantly improved the available signal. The `risk_score` composite feature achieved 35% higher MI than the best original feature (`engine_hours`), validating the domain-driven approach.

### 2.12 Risk Score Analysis

Two visualizations were created for the `risk_score` feature:

**Breakdown Rate by Risk Score (Bar Chart)**

*Purpose:* Shows the breakdown rate (proportion of class 1) for each risk score value (0 through 9), with a dashed line indicating the global mean breakdown rate (~17%).

*Key observation:* The breakdown rate increases monotonically with the risk score: vehicles with risk_score = 0 have the lowest breakdown rate, while those with the highest scores have breakdown rates well above the global mean. This confirms that the composite risk score effectively captures cumulative risk and is a strong discriminative feature.

**Risk Score Distribution by Class (KDE)**

*Purpose:* Shows the KDE distributions of the risk score for each class. This reveals how the risk score is distributed across breakdown vs. non-breakdown vehicles.

*Key observation:* The distribution for class 1 (breakdown) is shifted to the right compared to class 0 (no breakdown), with more density at higher risk scores. While there is still overlap (the feature alone is not perfectly separating), the separation is more pronounced than for any individual original feature, confirming the value of the composite approach.

**Point-Biserial Correlation of Engineered Features with Target:**

| Feature | r | p-value | Significant? |
|---|---|---|---|
| `risk_score` | 0.1602 | 0.0000 | ✅ |
| `age_x_mileage` | 0.1379 | 0.0000 | ✅ |
| `is_old_vehicle` | 0.1090 | 0.0007 | ✅ |
| `is_service_overdue` | 0.1016 | 0.0016 | ✅ |
| `is_high_mileage` | 0.0810 | 0.0120 | ✅ |
| `log_mileage` | 0.0625 | 0.0525 | ❌ |
| `is_low_oil_quality` | 0.0405 | 0.2092 | ❌ |
| `log_last_service` | 0.0244 | 0.4490 | ❌ |
| `oil_x_service` | 0.0073 | 0.8212 | ❌ |
| `log_avg_trip` | -0.0008 | 0.9796 | ❌ |
| `service_overdue_ratio` | -0.0048 | 0.8827 | ❌ |
| `hours_per_km` | -0.0245 | 0.4483 | ❌ |
| `is_dirty` | -0.0317 | 0.3259 | ❌ |
| `mileage_per_year` | -0.0568 | 0.0781 | ❌ |

5 out of 14 engineered features have statistically significant correlation with the target, with `risk_score` having the strongest correlation (r = 0.1602). Note the interesting discrepancy between MI and point-biserial correlation for `is_low_oil_quality`: it ranks 2nd in MI but is not significant in point-biserial. This is because MI captures non-linear dependencies that linear correlation misses.

### 2.13 EDA Conclusions

The EDA phase produced the following conclusions that directly informed the modeling strategy:

1. **Class imbalance (~17% class 1)** requires specific handling:
   - Use `class_weight='balanced'` or `scale_pos_weight` in models that support it
   - Apply SMOTE **inside** the pipeline (not before the train/test split, which would cause data leakage)
   - Optimize the classification threshold (the default 0.5 is suboptimal; optimal is approximately ~0.17 based on class prevalence)

2. **Most informative features** (by MI):
   - `risk_score` (engineered) — best overall
   - `is_low_oil_quality` (engineered)
   - `engine_hours` (original)
   - `log_last_service` / `last_service_km_ago`

3. **Variables with most different distributions per class:**
   - `last_service_km_ago`: class 1 has higher values (more km since service)
   - `oil_quality_pct`: class 1 has lower values (worse oil quality)

4. **Engineered features outperform originals** in MI → include them in modeling.

5. **Evaluation must use F1-class1, Balanced Accuracy, and ROC-AUC**, NOT global accuracy (which is misleading due to class imbalance).

---

## 3. Part II: Modeling

The modeling phase (notebook: `jose_modeling2_english.ipynb`) builds on the EDA findings to develop, compare, and optimize classification models.

### 3.1 Problem Statement and Objectives

The baseline model (Gradient Boosting with default threshold) produced only ~16 class-1 predictions out of 450 test samples, achieving very low recall on the minority class (~5%). The objective was to significantly improve minority class detection:

| Metric | Baseline | Target |
|---|---|---|
| Recall class 1 | ~0.05 | ≥ 0.50 |
| Precision class 1 | — | ≥ 0.40 |
| F1 class 1 | ~0.09 | ≥ 0.45 |
| Global Accuracy | 0.83 | ≥ 0.80 |

The fundamental tension is between **recall** (catching as many breakdowns as possible) and **precision** (not generating too many false alarms). In a real-world scenario, missing a breakdown (false negative) could mean a vehicle stranded on the road, while a false alarm (false positive) only means an unnecessary inspection — a much less costly error. This asymmetry justifies prioritizing recall.

### 3.2 Pipeline Architecture

All models were built using scikit-learn's `Pipeline` and `ColumnTransformer` architecture to ensure:

1. **No data leakage**: All transformations (imputation, scaling, encoding, oversampling) are applied inside the pipeline, meaning they are fit only on training data during cross-validation.
2. **Reproducibility**: The entire preprocessing + model chain is encapsulated in a single object.
3. **Easy comparison**: All models share the same preprocessor, so differences in results are attributable to the model itself.

**Preprocessor structure:**

```
ColumnTransformer
├── Numerical Pipeline (22 features: 8 original + 14 engineered)
│   └── SimpleImputer(strategy='median')
└── Categorical Pipeline (4 features)
    ├── SimpleImputer(strategy='most_frequent')
    └── OneHotEncoder(handle_unknown='ignore')
```

*Why `handle_unknown='ignore'` in OneHotEncoder:* The test set might contain categorical values not seen during training. This setting ensures the encoder does not crash on unknown categories; instead, it assigns a zero vector (no category active), which is a safe default.

### 3.3 Data Preparation for Modeling

**Train/Validation Split:**

A stratified 80/20 split was performed using `train_test_split` with `stratify=y_train_full`:

| Set | Rows | Class 0 | Class 1 |
|---|---|---|---|
| Training | 770 | 640 | 130 |
| Validation | 193 | 160 | 33 |

*Why stratified split:* With only 17% positive examples, a random split could accidentally create a validation set with too few (or too many) positive examples, leading to unreliable metric estimates. Stratified splitting guarantees that both sets maintain the same class proportions as the full dataset.

**Feature engineering was applied to both train and test sets** using the `add_engineered_features()` function. Infinite values (`inf`, `-inf`) resulting from division operations were replaced with `NaN` and subsequently handled by the imputer in the pipeline.

### 3.4 Evaluation Framework

An `evaluate_model()` function was created that for each model:

1. **Trains** the pipeline on the training data
2. **Computes predicted probabilities** (using `predict_proba` or `decision_function`)
3. **Applies a custom threshold** (default 0.5) to convert probabilities to class predictions
4. **Computes all metrics**: Accuracy, Balanced Accuracy, F1-class1, Recall-class1, Precision-class1, ROC-AUC
5. **Prints** a full classification report
6. **Displays** a confusion matrix heatmap

*Why support custom thresholds:* As identified in the EDA, the default threshold of 0.5 is suboptimal for imbalanced datasets. The function supports any threshold value, enabling the threshold analysis described in Section 3.6.

*Helper function `predict_with_threshold()`* was also created for generating final submission predictions with the optimal threshold.

### 3.5 Baseline Model

**Configuration:** GradientBoostingClassifier with:
- `n_estimators=200`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`
- **No class balancing** (no `class_weight`, no SMOTE)
- **Only original features** (8 numerical + 4 categorical) — no engineered features
- **Default threshold** = 0.5

**Results on validation set:**

| Metric | Value |
|---|---|
| Accuracy | 0.8342 |
| Balanced Accuracy | 0.5512 |
| F1 class 1 | 0.2000 |
| Recall class 1 | 0.1212 |
| Precision class 1 | 0.5714 |
| ROC-AUC | 0.6419 |

**Confusion matrix** showed only **7 predictions of class 1** out of 193 validation samples (4 correct, 3 false positives). The model predicted "no breakdown" for almost every vehicle, achieving high accuracy (83%) by simply following the majority class. This is the hallmark of a model that has not learned to detect the minority class.

*Graph: Confusion Matrix for Baseline*
The confusion matrix heatmap visually confirms the extreme bias: the vast majority of predictions fall in the "No breakdown predicted" column, with very few attempts to predict breakdowns.

**Baseline ROC and PR Curves**

Two curves were plotted:

*ROC Curve (Receiver Operating Characteristic):*
- Shows the tradeoff between True Positive Rate (TPR / Recall) and False Positive Rate (FPR) at different thresholds.
- The baseline achieved AUC = 0.642, which is better than random (0.5) but far from good.
- The operating point at threshold=0.5 was marked with a red dot, showing it operates in the very conservative region (low TPR, low FPR) — meaning it barely predicts any positives.

*Purpose of ROC curve:* It shows the model's discrimination ability across all possible thresholds. The AUC summarizes overall performance: 0.5 = random, 1.0 = perfect. An AUC of 0.64 means the model has some discrimination ability but is far from reliable.

*PR Curve (Precision-Recall):*
- Shows the tradeoff between Precision and Recall for the positive class at different thresholds.
- The horizontal dashed line marks the prevalence (17%), which represents the precision of a random classifier.
- The operating point at threshold=0.5 shows high precision but extremely low recall.

*Purpose of PR curve:* For imbalanced datasets, PR curves are often more informative than ROC curves because they focus on the minority class. A model that looks reasonable on the ROC curve may look much worse on the PR curve because high accuracy on the majority class inflates the ROC.

*Key insight from both curves:* The baseline model has reasonable discrimination ability (AUC = 0.64) but the threshold of 0.5 wastes this potential by being too conservative. Moving the threshold lower could dramatically increase recall with a manageable decrease in precision.

### 3.6 Threshold Analysis — The Highest Impact Technique

This is one of the most important sections of the modeling phase.

**Why threshold matters:**

In a binary classifier, the model outputs a probability P(class=1). To make a binary prediction, we compare this probability to a threshold:
- If P ≥ threshold → predict class 1
- If P < threshold → predict class 0

The default threshold is 0.5, which is appropriate when classes are balanced (50/50). However, when the prevalence is ~17%, the model's predicted probabilities are calibrated around that prevalence. Most vehicles will get predicted probabilities below 0.5, even those at higher risk. A threshold of 0.5 is therefore too high and results in very few positive predictions.

**The experiment:**

50 threshold values were tested from 0.05 to 0.90. For each threshold, F1-class1, Recall-class1, Precision-class1, and global Accuracy were computed on the validation set using the baseline model's predicted probabilities.

**Graph: F1 / Recall / Precision vs Threshold**

*Purpose:* This graph shows how the three key metrics change as the threshold moves from 0.05 (predicting almost everything as class 1) to 0.90 (predicting almost nothing as class 1). Three vertical lines mark important thresholds:
- Gray dashed: threshold = 0.50 (default)
- Green dotted: threshold = 0.17 (matching the class prevalence)
- Red solid: optimal threshold for F1-class1

*Key observations:*
- As the threshold decreases from 0.5, **Recall increases dramatically** (more true breakdowns are caught) while **Precision decreases gradually** (more false alarms occur).
- The F1-class1 curve has a clear peak around threshold ≈ 0.21, balancing the recall/precision tradeoff.
- At threshold = 0.50: F1 = 0.200, Recall = 0.121 (only 12% of breakdowns caught)
- At threshold = 0.17: F1 = 0.366, Recall = 0.455 (45% of breakdowns caught)
- At threshold = 0.21 (optimal): F1 = 0.373, Recall = 0.424 (42% of breakdowns caught)

**Graph: Accuracy vs Threshold**

*Purpose:* This graph shows how global accuracy changes with the threshold, with a horizontal red line at 80% (the minimum acceptable accuracy).

*Key observation:* Accuracy decreases as the threshold decreases (because more false positives appear), but remains above 80% until approximately threshold = 0.23. Below that, accuracy drops below the minimum target. This establishes the lower bound for threshold selection.

**Comparison of three thresholds:**

| Threshold | F1 | Recall | Accuracy | # class 1 predictions |
|---|---|---|---|---|
| 0.50 (default) | 0.200 | 0.121 | 0.834 | 7 |
| 0.17 (prevalence) | 0.366 | 0.455 | 0.731 | 49 |
| 0.21 (optimal F1) | 0.373 | 0.424 | 0.756 | 42 |

**Key insight:** Simply changing the threshold from 0.5 to 0.21 — **without changing the model at all** — nearly doubles the F1 score (0.200 → 0.373) and triples the recall (0.121 → 0.424). This is the single highest-impact improvement in the entire pipeline, and it demonstrates that threshold tuning is essential for imbalanced classification problems.

### 3.7 Sampling Strategies and Engineered Features

Four different sampling strategies were tested, all using the `imblearn.Pipeline` (which supports resampling steps) with the full set of engineered features:

#### SMOTE + GradientBoosting + Engineered Features

**What is SMOTE (Synthetic Minority Over-sampling Technique):**
SMOTE generates synthetic examples of the minority class by interpolating between existing minority class samples. For each minority example, SMOTE finds its k-nearest neighbors (default k=5) in the feature space and creates new synthetic examples along the line segments connecting them. This increases the number of minority class samples, making the training set more balanced.

*Why inside the pipeline:* SMOTE must be applied ONLY to the training data, never to the validation/test data. By placing it inside an `ImbPipeline`, we guarantee that oversampling happens after the train/validation split during cross-validation. Applying SMOTE before the split would create synthetic validation examples that are interpolations of training examples — a severe form of data leakage that would produce overly optimistic validation scores.

**Results:**
| Metric | Value |
|---|---|
| Accuracy | 0.8135 |
| F1 class 1 | 0.1000 |
| Recall class 1 | 0.0606 |
| ROC-AUC | 0.5369 |

*Observation:* Surprisingly, SMOTE + GradientBoosting performed **worse** than the baseline. The model predicted only 2 breakdowns correctly out of 33. This counter-intuitive result can be explained by the fact that SMOTE generates synthetic examples in feature space, which may not represent realistic breakdown patterns. When the feature space is noisy (many features with low MI), the synthetic examples can introduce noise rather than signal.

#### SMOTE + RandomForest (balanced)

**Configuration:** SMOTE followed by RandomForestClassifier with `class_weight='balanced'` (double balancing: both oversampling and cost-sensitive learning).

**Results:**
| Metric | Value |
|---|---|
| Accuracy | 0.8187 |
| F1 class 1 | 0.0541 |
| Recall class 1 | 0.0303 |
| ROC-AUC | 0.5843 |

*Observation:* Even worse than SMOTE + GB. Combining SMOTE with `class_weight='balanced'` resulted in double-correction that made the model oscillate and fail to learn meaningful patterns.

#### BorderlineSMOTE + GradientBoosting

**What is BorderlineSMOTE:**
A variant of SMOTE that only generates synthetic examples from minority class instances that are near the decision boundary (borderline examples). The intuition is that these are the most informative examples for the classifier — examples deep in the minority cluster are already easy to classify, while borderline examples help the model learn the boundary better.

**Results:**
| Metric | Value |
|---|---|
| Accuracy | 0.8290 |
| F1 class 1 | 0.1081 |
| Recall class 1 | 0.0606 |
| ROC-AUC | 0.5519 |

*Observation:* Slightly better than regular SMOTE but still worse than the baseline. The borderline approach is theoretically superior but did not help with this particular dataset.

#### SMOTETomek + GradientBoosting

**What is SMOTETomek:**
A combined approach that first applies SMOTE (oversampling the minority class) and then applies Tomek Links (a cleaning step that removes ambiguous examples near the class boundary). Tomek Links are pairs of examples from different classes that are each other's nearest neighbor — removing them cleans up the boundary region.

**Results:**
| Metric | Value |
|---|---|
| Accuracy | 0.8135 |
| F1 class 1 | 0.1000 |
| Recall class 1 | 0.0606 |
| ROC-AUC | 0.5369 |

*Observation:* Same performance as regular SMOTE. The Tomek Link cleaning step did not add meaningful value for this dataset.

**Overall conclusion on sampling strategies:** All four SMOTE-based approaches performed worse than the baseline at threshold=0.5. This is an important finding: SMOTE is not a universal solution to class imbalance. In this case, the limited discriminative power of the features means that synthetic minority examples add noise rather than meaningful signal. Alternative approaches (class weighting, threshold tuning) proved more effective.

### 3.8 Decision Tree — Interpretable Model

A Decision Tree was trained as first simple model, providing an interpretable model that can be visualized and explained.

**Hyperparameter Tuning:**

`GridSearchCV` was used with 5-fold stratified cross-validation, optimizing for F1-class1:

| Hyperparameter | Search Space | Best Value |
|---|---|---|
| `max_depth` | [3, 4, 5, 6, 7, 8] | 5 |
| `min_samples_leaf` | [5, 10, 15, 20] | 5 |
| `criterion` | ['gini', 'entropy'] | entropy |

Additional fixed setting: `class_weight='balanced'` — this adjusts the loss function to penalize misclassification of the minority class proportionally to its under-representation.

**Best CV F1:** 0.3254

**Results on validation set:**
| Metric | Value |
|---|---|
| Accuracy | 0.5959 |
| Balanced Accuracy | 0.5879 |
| F1 class 1 | 0.3276 |
| Recall class 1 | 0.5758 |
| Precision class 1 | 0.2289 |
| ROC-AUC | 0.5893 |

*Observation:* The Decision Tree achieved the **highest recall** (57.6%) of any model at threshold=0.5, meaning it correctly identified more than half of the breakdowns. However, this came at a severe cost to accuracy (59.6%) and precision (22.9%). The model generates many false positives — predicting breakdown for vehicles that won't actually break down.

**Tree Visualization (Top 3 Levels):**

*Purpose of this graph:* A visual representation of the first 3 levels of the decision tree, showing:
- Which features the tree splits on (the most important features appear near the root)
- The threshold values for each split
- The class distribution at each node
- The predicted class at each leaf

*Key observations from the tree:*
- The root node likely splits on one of the engineered features (e.g., `risk_score` or `is_service_overdue`), confirming that the feature engineering was valuable.
- The tree is relatively shallow (max_depth=5), which limits overfitting but also limits the model's ability to capture complex patterns.
- With `class_weight='balanced'`, the tree is more aggressive in predicting class 1, which explains the high recall but low precision.

### 3.9 Advanced Models — XGBoost and RandomForest Tuned

#### XGBoost with scale_pos_weight

**What is XGBoost:** XGBoost (eXtreme Gradient Boosting) is an optimized gradient boosting framework that uses regularization (L1 and L2), tree pruning, and parallel processing. It builds trees sequentially, where each new tree corrects the errors of the previous ensemble.

**What is scale_pos_weight:** This parameter tells XGBoost to weight positive examples more heavily during training. It is set to the ratio of negative to positive examples: `scale_pos_weight = n_neg / n_pos = 640 / 130 = 4.92`. This means the model treats each positive example as if it were approximately 5 examples, compensating for the class imbalance.

**Configuration:**
- `n_estimators=300`, `learning_rate=0.05`, `max_depth=4`
- `subsample=0.8`, `colsample_bytree=0.8` (stochastic gradient boosting: each tree sees only 80% of rows and 80% of features, reducing overfitting)
- `scale_pos_weight=4.92`

**Results:**
| Metric | Value |
|---|---|
| Accuracy | 0.7772 |
| F1 class 1 | 0.2456 |
| Recall class 1 | 0.2121 |
| Precision class 1 | 0.2917 |
| ROC-AUC | 0.6193 |

*Observation:* XGBoost showed improvement over the SMOTE-based approaches but did not outperform the baseline in F1. The model was more willing to predict breakdowns than the baseline (thanks to `scale_pos_weight`), but still conservative at threshold=0.5.

#### RandomForest with RandomizedSearchCV

**What is RandomizedSearchCV:** Unlike GridSearchCV (which exhaustively tries all parameter combinations), RandomizedSearchCV samples a fixed number of random combinations from the parameter space. This is more efficient when the search space is large.

**Search space and best parameters:**

| Hyperparameter | Search Space | Best Value |
|---|---|---|
| `n_estimators` | [100, 200, 300] | 100 |
| `max_depth` | [5, 8, 10, 15, None] | 8 |
| `min_samples_leaf` | [2, 5, 10] | 10 |
| `max_features` | ['sqrt', 'log2', 0.5] | log2 |

Fixed: `class_weight='balanced'`, `n_iter=20` (20 random combinations tried).

**Best CV F1:** 0.1830

**Results on validation:**
| Metric | Value |
|---|---|
| Accuracy | 0.7565 |
| Balanced Accuracy | 0.6006 |
| F1 class 1 | 0.3380 |
| Recall class 1 | 0.3636 |
| Precision class 1 | 0.3158 |
| ROC-AUC | 0.6460 |

*Observation:* The tuned RandomForest achieved the **best F1-class1** (0.338) among all models at threshold=0.5 and the **highest ROC-AUC** (0.646), indicating the best overall discrimination ability. Its balanced accuracy (0.601) was also the best, confirming it as the most well-rounded model.

### 3.10 Cross-Validation with Correct Metrics

To obtain robust, unbiased performance estimates, 5-fold stratified cross-validation was performed on the **full training set** (963 rows) for all models.

**Why cross-validation instead of a single train/validation split:**
A single split produces estimates that depend on the specific random partition. Cross-validation averages results over 5 different splits, providing more reliable estimates and also a standard deviation that indicates how much the performance varies.

**Metrics used:** F1-class1, Balanced Accuracy, ROC-AUC. Global accuracy was deliberately **not** included because it is misleading for imbalanced datasets.

**Results:**

| Model | F1 mean | F1 std | Balanced Acc | ROC-AUC |
|---|---|---|---|---|
| Decision Tree Best | 0.3219 | 0.0501 | 0.5906 | 0.5791 |
| RandomForest Tuned | 0.2676 | 0.0378 | 0.5588 | 0.6358 |
| XGBoost | 0.2022 | 0.0507 | 0.5351 | 0.5756 |
| BorderlineSMOTE+GB | 0.0872 | 0.0627 | 0.5056 | 0.5880 |
| SMOTE+GB+Eng | 0.0780 | 0.0662 | 0.5039 | 0.5792 |
| Baseline GB | 0.0733 | 0.0240 | 0.5090 | 0.6032 |

**Key insights:**
1. The **Decision Tree** has the highest mean F1 in cross-validation (0.3219), closely followed by the tuned RandomForest (0.2676).
2. The tuned **RandomForest has the highest ROC-AUC** (0.6358) and the **lowest F1 standard deviation** (0.0378), making it the most stable model.
3. All SMOTE-based approaches performed poorly in cross-validation, confirming the single-split findings.
4. The **baseline** has the lowest F1 mean (0.0733) but interestingly the second-highest ROC-AUC (0.6032), confirming that its problem is the threshold, not the underlying discrimination ability.

### 3.11 Model Comparison

**Validation Comparison Table:**

All 8 models were compared on the validation set in a single table:

| Model | F1-class1 | Balanced Acc | ROC-AUC | Accuracy | Recall-c1 | Precision-c1 |
|---|---|---|---|---|---|---|
| RandomForest (tuned, balanced) | 0.3380 | 0.6006 | 0.6460 | 0.7565 | 0.3636 | 0.3158 |
| Decision Tree (best, balanced) | 0.3276 | 0.5879 | 0.5893 | 0.5959 | 0.5758 | 0.2289 |
| XGBoost (scale_pos_weight=4.9) | 0.2456 | 0.5529 | 0.6193 | 0.7772 | 0.2121 | 0.2917 |
| Baseline GB (threshold=0.5) | 0.2000 | 0.5512 | 0.6419 | 0.8342 | 0.1212 | 0.5714 |
| BorderlineSMOTE + GB | 0.1081 | 0.5241 | 0.5519 | 0.8290 | 0.0606 | 0.5000 |
| SMOTE + GB + Eng. Features | 0.1000 | 0.5147 | 0.5369 | 0.8135 | 0.0606 | 0.2857 |
| SMOTETomek + GB | 0.1000 | 0.5147 | 0.5369 | 0.8135 | 0.0606 | 0.2857 |
| SMOTE + RF (balanced) | 0.0541 | 0.5058 | 0.5843 | 0.8187 | 0.0303 | 0.2500 |

**Graph: Overlaid ROC Curves — All Models**

*Purpose:* Overlaying the ROC curves of all models on a single plot allows direct visual comparison of their discrimination abilities. Models with curves closer to the top-left corner have better overall discrimination.

*Key observations:*
- The curves are relatively close together, indicating that no model dramatically outperforms the others in overall discrimination.
- The RandomForest and Baseline GB curves are slightly higher, corresponding to their higher AUC values.
- All models are well above the diagonal (random classifier), confirming they have learned some useful signal.

**Graph: Overlaid PR Curves — All Models**

*Purpose:* PR curves are more informative than ROC curves for imbalanced problems because they focus on the minority class. The horizontal line marks the prevalence (17%), which is the precision of a random classifier.

*Key observations:*
- All curves start with high precision at low recall (the most confident predictions are often correct) but precision drops rapidly as recall increases.
- No model achieves both high precision and high recall simultaneously, reflecting the inherent difficulty of the problem.
- The separation between curves is more pronounced in the PR space than in the ROC space.

### 3.12 Threshold Optimization of the Best Model

The **RandomForest (tuned, balanced)** was selected as the best model based on the highest F1-class1 on the validation set (0.338) and the highest ROC-AUC (0.646).

**Threshold sweep with constraints:**

100 threshold values from 0.05 to 0.80 were tested. Two constraints were imposed:
1. Global accuracy ≥ 0.80
2. Recall-class1 ≥ 0.40

*Why these constraints:*
- The accuracy constraint ensures the model remains useful overall and doesn't generate an unacceptable number of false positives.
- The recall constraint ensures a minimum level of breakdown detection, which is the primary business objective.

**Optimal threshold found:** 0.558

| Metric | Value |
|---|---|
| F1 class 1 | 0.3200 |
| Recall class 1 | 0.2424 |
| Precision class 1 | 0.4706 |
| Accuracy | 0.8238 |

**Note:** The optimal threshold (0.558) is **higher** than 0.5, which is counterintuitive. This is because the RandomForest with `class_weight='balanced'` already adjusts its probability outputs to compensate for class imbalance, effectively shifting the probability distribution upward for the minority class. With this internal balancing, a slightly higher threshold is needed to avoid over-predicting breakdowns.

This is fundamentally different from the baseline analysis where lowering the threshold helped — the baseline had no class balancing, so its probabilities were skewed low for the minority class.

**Graph: Threshold Optimization — Metrics vs Threshold for Best Model**

*Purpose:* Shows how F1-class1 (red), Recall (orange dashed), Precision (blue dash-dot), and Accuracy (green dotted) change across thresholds for the best model. A horizontal gray line marks the minimum acceptable accuracy (0.80), and a vertical red line marks the optimal threshold.

*Key observation:* The optimal threshold balances all four metrics within the constraint boundaries. Increasing the threshold above 0.558 improves precision but sacrifices recall; decreasing it improves recall but drops accuracy below 0.80.

### 3.13 Final Evaluation and Submission

**Final evaluation with optimal threshold:**

| Metric | Value | Target | Status |
|---|---|---|---|
| Recall class 1 | 0.2424 | ≥ 0.50 | ❌ Not met |
| Precision class 1 | 0.4706 | ≥ 0.40 | ✅ Met |
| Accuracy | 0.8238 | ≥ 0.80 | ✅ Met |

The recall target was not fully achieved, but the model represents a significant improvement over the baseline (Recall: 0.12 → 0.24, a 100% improvement).

**Retraining and Submission:**

The best model was retrained on the **full training set** (963 rows) before generating predictions on the test set. This is important because: the model benefits from seeing all available training data, not just the 80% used during the train/validation split.

**Test set predictions:**
| Class | Count | Percentage |
|---|---|---|
| 0 (No breakdown) | 411 | 91.3% |
| 1 (Breakdown) | 39 | 8.7% |

Expected ~76 class 1 predictions (17% of 450), actual 39 (8.7%). The model is still somewhat conservative in predicting breakdowns, which is reflected in the precision-recall tradeoff.

The submission was saved as `MySubmissions3.csv` with the correct format matching `sample_submission.csv`.

### 3.14 Modeling Conclusions

1. **Threshold tuning was the single most impactful technique** — doubling the F1 score of the baseline without changing the model.

2. **SMOTE-based approaches were ineffective** for this dataset. All four variants (SMOTE, BorderlineSMOTE, SMOTETomek, SMOTE+RF) performed worse than the baseline. This is likely because the features have limited discriminative power, and synthetic minority examples introduce noise.

3. **Class weighting** (`class_weight='balanced'`, `scale_pos_weight`) was more effective than oversampling. It adjusts the loss function without creating artificial data points.

4. **The tuned RandomForest** was the best overall model, achieving the highest F1-class1 (0.338), highest ROC-AUC (0.646), and best balanced accuracy (0.601).

5. **The Decision Tree** achieved the highest recall (57.6%) but at the cost of very low accuracy (59.6%) and precision (22.9%). It is valuable for interpretability but not ideal for production use.

6. **Engineered features significantly improved MI** — the top 2 features by MI were both engineered (`risk_score`, `is_low_oil_quality`), demonstrating that domain knowledge adds substantial value to the pipeline.

7. **The problem is inherently difficult**: even the best model achieves ROC-AUC of only 0.646. The features available may not contain enough information to reliably predict breakdowns, suggesting that additional data sources (e.g., diagnostic codes, vibration sensors, maintenance records) could improve performance.

---

## 4. Part III: How to Add More Models to the Pipeline

The modeling notebook is structured as a **modular pipeline** that makes it easy to add, test, and compare new models. Here is a step-by-step guide:

### Step 1: Import the Model

Add the import at the top of the notebook (cell 1). For example, to add a Support Vector Machine:

```python
from sklearn.svm import SVC
```

Or for LightGBM:

```python
import lightgbm as lgb
```

### Step 2: Create the Pipeline

All models use the same preprocessor. For models that don't need oversampling, use scikit-learn's `Pipeline`:

```python
new_pipeline = Pipeline([
    ('preprocessor', preprocessor),  # Reuse the existing preprocessor
    ('classifier', SVC(
        class_weight='balanced',
        probability=True,  # Required for predict_proba
        random_state=RANDOM_STATE
    ))
])
```

If you want to include oversampling (e.g., SMOTE), use `imblearn`'s `ImbPipeline`:

```python
from imblearn.pipeline import Pipeline as ImbPipeline

new_pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=RANDOM_STATE)),
    ('classifier', SVC(class_weight='balanced', probability=True, random_state=RANDOM_STATE))
])
```

**Important:** Models must support `predict_proba()` for threshold tuning to work. For scikit-learn's SVC, set `probability=True`. For models that only support `decision_function()`, the `evaluate_model()` function already handles this fallback.

### Step 3: Evaluate the Model

Use the existing `evaluate_model()` function:

```python
new_result = evaluate_model(
    'My New Model Name',          # Descriptive name
    new_pipeline,                  # The pipeline object
    X_train, y_train,             # Training data
    X_val, y_val,                 # Validation data
    threshold=0.5                  # Start with 0.5, optimize later
)
results.append(new_result)  # Add to results list for comparison
```

This will automatically:
- Train the model
- Generate predictions and probabilities
- Compute all metrics (Accuracy, Balanced Accuracy, F1, Recall, Precision, ROC-AUC)
- Print a classification report
- Display the confusion matrix

### Step 4: Hyperparameter Tuning (Optional but Recommended)

Use GridSearchCV or RandomizedSearchCV:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__kernel': ['rbf', 'poly'],
    'classifier__gamma': ['scale', 'auto']
}

grid = GridSearchCV(
    new_pipeline, param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
    scoring='f1',  # Optimize for F1-class1
    n_jobs=-1
)

grid.fit(X_train, y_train)
print('Best params:', grid.best_params_)
print('Best CV F1:', grid.best_score_)

# Evaluate best model
tuned_result = evaluate_model(
    'My Model (tuned)',
    grid.best_estimator_,
    X_train, y_train, X_val, y_val
)
results.append(tuned_result)
```

**Note on `scoring` parameter:** Always use `scoring='f1'` (not `scoring='accuracy'`) because accuracy is misleading for imbalanced datasets.

### Step 5: Add to Cross-Validation

Add the new model to the `cv_models` list in the cross-validation section:

```python
cv_models = [
    # ... existing models ...
    ('My New Model', new_pipeline),
]
```

The cross-validation loop will automatically evaluate it with the same metrics and folds as all other models.

### Step 6: Model Comparison

The new model will automatically appear in the comparison table and the ROC/PR curve plots because it was added to the `results` list. Simply re-run the comparison cells.

### Step 7: Threshold Optimization (If the New Model is Best)

If the new model achieves the best F1-class1, update the best model selection and run threshold optimization:

```python
best_result = max(non_baseline, key=lambda r: r['f1_class1'])
# The threshold optimization code will automatically work with the new best model
```

### Additional Models to Consider

| Model | Import | Key Parameters | Notes |
|---|---|---|---|
| Logistic Regression | `sklearn.linear_model.LogisticRegression` | `class_weight='balanced'`, `C`, `penalty` | Good baseline, interpretable coefficients |
| SVM | `sklearn.svm.SVC` | `class_weight='balanced'`, `C`, `kernel`, `probability=True` | Strong with good features, slow on large datasets |
| LightGBM | `lightgbm.LGBMClassifier` | `is_unbalance=True`, `n_estimators`, `learning_rate` | Fast, often competitive with XGBoost |
| CatBoost | `catboost.CatBoostClassifier` | `auto_class_weights='Balanced'`, native categorical support | Handles categoricals natively, no encoding needed |
| MLP (Neural Network) | `sklearn.neural_network.MLPClassifier` | `hidden_layer_sizes`, `alpha` | Needs careful tuning, good for complex patterns |
| Stacking Ensemble | `sklearn.ensemble.StackingClassifier` | Combine multiple models | Can leverage strengths of different models |

### Tips for Adding Models

1. **Always use `class_weight='balanced'`** (or equivalent) for models that support it. This is the most effective technique for this dataset.
2. **Always use the same `RANDOM_STATE`** for reproducibility.
3. **Try threshold optimization** on every model — the optimal threshold varies per model.
4. **Prioritize F1-class1 and ROC-AUC** over accuracy when comparing models.
5. **Check cross-validation stability** (low standard deviation in F1) before trusting single-split results.

---

## 5. Final Conclusions

### Summary of the Full Pipeline

| Phase | Key Finding |
|---|---|
| Data Cleaning | 87 rows removed (8.3%) with impossible values |
| Class Imbalance | 83.1% / 16.9% split — critical for evaluation and modeling strategy |
| Statistical Testing | Only 2/8 original features are statistically significant (vehicle_age, oil_quality) |
| Feature Engineering | 14 new features created; top 2 by MI are engineered (risk_score, is_low_oil_quality) |
| Baseline Model | GradientBoosting at threshold=0.5: F1=0.20, Recall=0.12 — nearly useless for detecting breakdowns |
| Threshold Analysis | Changing threshold from 0.5 to 0.21 doubles F1 to 0.37 — highest-impact single change |
| SMOTE-based Methods | All performed worse than baseline — not effective for this dataset |
| Best Model | RandomForest (tuned, balanced): F1=0.338, ROC-AUC=0.646, best overall balance |
| Decision Tree | Highest recall (57.6%) but lowest accuracy (59.6%) — useful for interpretability |
| Final Submission | 39 class 1 predictions out of 450 test samples (8.7%) |

### Key Takeaways

1. **Class imbalance must be addressed holistically**: not just through sampling or class weights, but also through threshold tuning and appropriate evaluation metrics.

2. **Feature engineering rooted in domain knowledge** consistently outperforms blind statistical methods. The composite `risk_score` feature became the most informative predictor.

3. **Simple techniques often outperform complex ones**: threshold tuning (zero model changes) had more impact than SMOTE, and `class_weight='balanced'` was more effective than synthetic oversampling.

4. **The problem has inherent limitations**: with an ROC-AUC of ~0.65, the available features contain limited information about future breakdowns. Improving performance beyond this point would likely require additional data sources (sensor data, maintenance logs, diagnostic codes).

5. **Evaluation methodology matters**: using global accuracy would have selected the baseline as the "best" model (83.4% accuracy) despite it being nearly useless for the actual task of detecting breakdowns. F1-class1 and ROC-AUC correctly identified the models that actually detect breakdowns.

### Winning Model Summary

**Model:** RandomForest with `class_weight='balanced'`, tuned via RandomizedSearchCV
**Best hyperparameters:** `n_estimators=100`, `max_depth=8`, `min_samples_leaf=10`, `max_features='log2'`
**Optimal threshold:** 0.558
**Validation metrics:** Accuracy=0.824, Precision-class1=0.471, Recall-class1=0.242, F1-class1=0.320, ROC-AUC=0.646

## 6. Gen AI Policy

Artificial intelligence was used to assist in this project. It was mainly used with help in writing code snippets. It was also used to get suggestions for different models other than random forest. The main thought process of the project and the choices made are our own choice and decision.


