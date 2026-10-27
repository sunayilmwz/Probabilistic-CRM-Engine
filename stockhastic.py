import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix
from lifelines import KaplanMeierFitter, CoxPHFitter
from sklearn.ensemble import RandomForestClassifier

# Set styling and random seed
np.random.seed(42)
sns.set_theme(style="whitegrid", palette="muted")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================================================
# PHASE 1: DATA GENERATING PROCESS (DGP) & COHORT ENGINEERING
# =============================================================================

def generate_retail_crm_data(n_customers=5000, observation_end="2026-10-01"):
    obs_end_date = pd.to_datetime(observation_end)
    customer_ids = [f"CUST_{i:05d}" for i in range(1, n_customers + 1)]
    days_since_acq = np.random.randint(30, 365, size=n_customers)
    acq_dates = [obs_end_date - pd.Timedelta(days=int(d)) for d in days_since_acq]
    
    acq_channel = np.random.choice(["Organic", "Paid_Ads", "Referral"], size=n_customers, p=[0.45, 0.35, 0.20])
    delivery_delay_rate = np.round(np.random.beta(a=1.5, b=8.0, size=n_customers), 3)
    support_tickets = np.random.poisson(lam=0.8 + 2.5 * delivery_delay_rate, size=n_customers)
    
    true_lambda = np.random.gamma(shape=2.0, scale=0.03, size=n_customers) 
    base_dropout_logit = -2.2 + 1.8 * delivery_delay_rate + 0.35 * support_tickets
    true_dropout_p = 1 / (1 + np.exp(-base_dropout_logit))
    
    customers_df = pd.DataFrame({
        "customer_id": customer_ids,
        "acquisition_date": acq_dates,
        "acquisition_channel": acq_channel,
        "delivery_delay_rate": delivery_delay_rate,
        "support_tickets": support_tickets,
        "true_lambda": true_lambda,
        "true_dropout_p": true_dropout_p
    })
    
    transactions = []
    is_churned_list = []
    
    for idx, row in customers_df.iterrows():
        curr_date = row["acquisition_date"]
        lam = row["true_lambda"]
        p_drop = row["true_dropout_p"]
        customer_mean_aov = np.random.gamma(shape=4.0, scale=120.0)
        
        order_val = np.random.lognormal(mean=np.log(customer_mean_aov), sigma=0.35)
        transactions.append((row["customer_id"], curr_date, round(order_val, 2)))
        
        alive = True
        while alive:
            wait_days = np.random.exponential(scale=1.0 / lam)
            curr_date += pd.Timedelta(days= max(1, int(round(wait_days))))
            if curr_date > obs_end_date:
                break
            if np.random.rand() < p_drop:
                alive = False
                break
            order_val = np.random.lognormal(mean=np.log(customer_mean_aov), sigma=0.35)
            transactions.append((row["customer_id"], curr_date, round(order_val, 2)))
            
        is_churned_list.append(0 if alive else 1)
        
    customers_df["latent_churned"] = is_churned_list
    transactions_df = pd.DataFrame(transactions, columns=["customer_id", "order_date", "order_amount"])
    
    return customers_df, transactions_df

def build_customer_summary(customers_df, transactions_df, observation_end="2026-10-01"):
    obs_end_date = pd.to_datetime(observation_end)
    tx = transactions_df.sort_values(["customer_id", "order_date"]).copy()
    
    tx["prev_order_date"] = tx.groupby("customer_id")["order_date"].shift(1)
    tx["ipt_days"] = (tx["order_date"] - tx["prev_order_date"]).dt.days
    
    agg_tx = tx.groupby("customer_id").agg(
        first_order=("order_date", "min"),
        last_order=("order_date", "max"),
        total_orders=("order_date", "count"),
        total_spend=("order_amount", "sum"),
        aov=("order_amount", "mean"),
        mean_ipt=("ipt_days", "mean")
    ).reset_index()
    
    agg_tx["frequency_repeat"] = agg_tx["total_orders"] - 1
    agg_tx["tenure_days"] = (obs_end_date - agg_tx["first_order"]).dt.days
    agg_tx["recency_clv_days"] = (agg_tx["last_order"] - agg_tx["first_order"]).dt.days
    
    crm_df = pd.merge(customers_df, agg_tx, on="customer_id", how="left")
    return crm_df


# =============================================================================
# PHASE 2: STOCHASTIC CLV & BUY 'TIL YOU DIE MODELING
# =============================================================================

def run_stochastic_clv_models(crm_df):
    clv_df = crm_df[crm_df["frequency_repeat"] > 0].copy()
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(clv_df["frequency_repeat"], clv_df["recency_clv_days"], clv_df["tenure_days"])
    
    clv_df["prob_alive"] = bgf.conditional_probability_alive(
        clv_df["frequency_repeat"], clv_df["recency_clv_days"], clv_df["tenure_days"]
    )
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(clv_df["frequency_repeat"], clv_df["aov"])
    
    clv_df["clv_180d"] = ggf.customer_lifetime_value(
        bgf, clv_df["frequency_repeat"], clv_df["recency_clv_days"],
        clv_df["tenure_days"], clv_df["aov"], time=6, discount_rate=0.01 
    )
    return clv_df


# =============================================================================
# PHASE 3: SURVIVAL ANALYSIS & CHURN HAZARD MODELING
# =============================================================================

def run_survival_analysis(crm_df):
    survival_df = crm_df[["tenure_days", "latent_churned", "delivery_delay_rate", "support_tickets", "acquisition_channel"]].copy()
    survival_df = pd.get_dummies(survival_df, columns=["acquisition_channel"], drop_first=True)
    
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(survival_df, duration_col="tenure_days", event_col="latent_churned")
    return cph


# =============================================================================
# PHASE 4: CAUSAL UPLIFT MODELING (T-LEARNER) & OPTIMIZATION
# =============================================================================

def run_causal_uplift_modeling(crm_df):
    print("\n" + "=" * 65)
    print("PHASE 4: CAUSAL UPLIFT MODELING (CATE) & OPTIMIZATION")
    print("=" * 65)
    
    # 1. Simulate A/B Test Campaign Results
    # 50% received a 20% discount coupon (Treatment=1), 50% didn't (Control=0)
    # Conversion probability depends on recency, delivery friction, and the treatment itself.
    np.random.seed(42)
    uplift_df = crm_df.dropna().copy()
    uplift_df["campaign_treatment"] = np.random.binomial(1, 0.5, size=len(uplift_df))
    
    # Latent true probability of converting during the campaign window
    base_prob = 0.1 + (uplift_df["frequency_repeat"] * 0.01) - (uplift_df["delivery_delay_rate"] * 0.2)
    # Treatment adds a positive bump, but less so for heavily delayed customers (Sleeping Dogs)
    treatment_effect = 0.15 - (uplift_df["delivery_delay_rate"] * 0.25)
    
    final_prob = base_prob + (uplift_df["campaign_treatment"] * treatment_effect)
    final_prob = np.clip(final_prob, 0.01, 0.99)
    uplift_df["campaign_conversion"] = np.random.binomial(1, final_prob)
    
    # 2. T-Learner Implementation (Two-Model Approach)
    features = ["frequency_repeat", "recency_clv_days", "tenure_days", "delivery_delay_rate", "support_tickets", "aov"]
    
    # Split data into Control (T=0) and Treatment (T=1)
    df_control = uplift_df[uplift_df["campaign_treatment"] == 0]
    df_treatment = uplift_df[uplift_df["campaign_treatment"] == 1]
    
    # Train Model 0 (Control)
    rf_control = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_control.fit(df_control[features], df_control["campaign_conversion"])
    
    # Train Model 1 (Treatment)
    rf_treatment = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_treatment.fit(df_treatment[features], df_treatment["campaign_conversion"])
    
    # 3. Predict Conditional Average Treatment Effect (CATE) for EVERY customer
    prob_if_treated = rf_treatment.predict_proba(uplift_df[features])[:, 1]
    prob_if_control = rf_control.predict_proba(uplift_df[features])[:, 1]
    
    # Uplift Score = P(Buy | Treatment) - P(Buy | Control)
    uplift_df["uplift_score"] = prob_if_treated - prob_if_control
    
    # 4. Budget Optimization & Target Group Formulation
    # Categorizing Customers based on Causal Inference logic
    def categorize_customer(row):
        if row["uplift_score"] > 0.05:
            return "Persuadable (TARGET)"
        elif row["uplift_score"] < -0.01:
            return "Sleeping Dog (DO NOT TOUCH)"
        elif prob_if_control[row.name] > 0.5:
            return "Sure Thing (WASTE OF BUDGET)"
        else:
            return "Lost Cause (IGNORE)"
            
    # Resetting index to align row names for the lambda function
    uplift_df = uplift_df.reset_index(drop=True)
    uplift_df["causal_segment"] = uplift_df.apply(categorize_customer, axis=1)
    
    print("\n[1] Customer Distribution by Causal Segments:")
    print(uplift_df["causal_segment"].value_counts())
    
    print("\n[2] Top 5 'Persuadable' Customers (Highest Incremental ROI):")
    # We target high uplift customers who ALSO have high historical AOV
    uplift_df["expected_incremental_revenue"] = uplift_df["uplift_score"] * uplift_df["aov"]
    top_targets = uplift_df[uplift_df["causal_segment"] == "Persuadable (TARGET)"].sort_values(
        "expected_incremental_revenue", ascending=False
    ).head()
    print(top_targets[["customer_id", "aov", "uplift_score", "expected_incremental_revenue"]].to_string(index=False))
    
    return uplift_df

# =============================================================================
# EXECUTION PIPELINE
# =============================================================================
if __name__ == "__main__":
    customers_df, transactions_df = generate_retail_crm_data(n_customers=5000)
    crm_df = build_customer_summary(customers_df, transactions_df)
    
    clv_df = run_stochastic_clv_models(crm_df)
    cph_model = run_survival_analysis(crm_df)
    
    # Run the final Causal Engine
    final_crm_df = run_causal_uplift_modeling(clv_df)
    print("\n=================================================================")
    print("PROBABILISTIC CRM ENGINE COMPLETED SUCCESSFULLY.")
    print("=================================================================")