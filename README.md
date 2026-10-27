#  Probabilistic CRM, Survival & Causal Uplift Engine

## 📌 Executive Summary
A comprehensive statistical engine designed to solve the three fundamental problems of non-contractual retail CRM: **Who is alive?**, **When will they churn?**, and **Who should we target?** By moving beyond deterministic RFM models, this project leverages stochastic processes, survival analysis, and causal inference to maximize Incremental ROI.

## Methodology & Statistical Architecture

### Phase 1: Data Generating Process (DGP) & Cohort Engineering
- **Objective:** Simulate a realistic, right-skewed, zero-inflated retail transaction dataset.
- **Statistical Tests Used:** Shapiro-Wilk (Normality), Mann-Whitney U (Friction impact), Chi-Square.

### Phase 2: Buy 'Til You Die (BTYD) Modeling
- **Frameworks:** BG/NBD (Beta-Geometric / Negative Binomial Distribution) and Gamma-Gamma.
- **Insight:** Separated the customer journey into two latent stochastic processes: the purchase process (Poisson/Gamma) and the dropout process (Geometric/Beta), forecasting 180-day Expected CLV.

### Phase 3: Survival Analysis (Time-to-Event Modeling)
- **Frameworks:** Kaplan-Meier Estimator and Cox Proportional Hazards (Cox PH) Model.
- **Insight:** Isolated the exact hazard ratios of operational metrics (e.g., delivery delays multiply churn hazard by 1.89x) handling right-censored data efficiently.

### Phase 4: Causal Uplift Modeling (T-Learner)
- **Framework:** Two-Model Machine Learning Approach (T-Learner) using Random Forests.
- **Insight:** Calculated Conditional Average Treatment Effect (CATE) to identify "Persuadables" and avoid wasting budget on "Sure Things" or triggering "Sleeping Dogs".

## 🛠 Tech Stack
`Python` | `Lifetimes` | `Lifelines` | `Scikit-Learn` | `SciPy` | `Pandas` | `Seaborn`
