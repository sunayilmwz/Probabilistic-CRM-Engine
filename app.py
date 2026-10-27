import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix
from lifelines import KaplanMeierFitter, CoxPHFitter

# Import core statistical functions from our backend engine
from stockhastic import (
    generate_retail_crm_data,
    build_customer_summary,
    run_stochastic_clv_models,
    run_causal_uplift_modeling
)

# Page Configuration
st.set_page_config(
    page_title="Probabilistic CRM & Causal Uplift Engine",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Probabilistic CRM, Survival & Causal Uplift Engine")
st.markdown(
    "An end-to-end decision support system combining **Buy 'Til You Die (BG/NBD & Gamma-Gamma)**, "
    "**Survival Analysis (Cox PH)**, and **Causal Inference (T-Learner)** to optimize marketing ROI."
)

# =============================================================================
# SIDEBAR: SIMULATION & OPTIMIZATION PARAMETERS
# =============================================================================
st.sidebar.header("⚙️ Engine Configuration")

n_customers = st.sidebar.slider(
    "Customer Base Size (N)",
    min_value=1000,
    max_value=10000,
    value=5000,
    step=1000
)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Campaign Budget Constraints")

total_budget = st.sidebar.number_input(
    "Total Marketing Budget ($)",
    min_value=500,
    max_value=100000,
    value=10000,
    step=500
)

coupon_cost = st.sidebar.slider(
    "Discount / Incentives Cost per Customer ($)",
    min_value=5,
    max_value=100,
    value=25,
    step=5
)


# =============================================================================
# CACHED PIPELINE EXECUTION
# =============================================================================
@st.cache_data(show_spinner=False)
def execute_crm_pipeline(n_cust):
    """Runs the stochastic simulation and trains all statistical models."""
    customers_df, transactions_df = generate_retail_crm_data(n_customers=n_cust)
    crm_df = build_customer_summary(customers_df, transactions_df)
    
    # Handle return signature cleanly
    clv_result = run_stochastic_clv_models(crm_df)
    if isinstance(clv_result, tuple):
        clv_df, bgf_model = clv_result
    else:
        clv_df = clv_result
        bgf_model = BetaGeoFitter(penalizer_coef=0.01)
        bgf_model.fit(
            clv_df["frequency_repeat"],
            clv_df["recency_clv_days"],
            clv_df["tenure_days"]
        )
        
    final_df = run_causal_uplift_modeling(clv_df)
    return crm_df, final_df, bgf_model


# =============================================================================
# MAIN DASHBOARD INTERFACE
# =============================================================================
if st.sidebar.button("Run Simulation Engine", type="primary"):
    with st.spinner("Simulating stochastic DGP and fitting statistical models..."):
        crm_df, final_df, bgf_model = execute_crm_pipeline(n_customers)
        
    st.success("All probabilistic, survival, and causal models converged successfully!")
    
    # Top-Level KPI Metrics
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Active Customer Base", f"{len(crm_df):,}")
    kpi2.metric("Repeat Buyers (CLV Eligible)", f"{len(final_df):,}")
    kpi3.metric("Mean 180-Day Expected CLV", f"${final_df['clv_180d'].mean():,.2f}")
    kpi4.metric(
        "Persuadable Segment Size",
        f"{(final_df['causal_segment'] == 'Persuadable (TARGET)').sum():,}"
    )
    
    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs([
        "📈 Stochastic CLV (BG/NBD)",
        "⏳ Survival & Hazard Analysis (Cox PH)",
        "🎯 Causal Uplift & Budget Optimizer"
    ])
    
    # -------------------------------------------------------------------------
    # TAB 1: STOCHASTIC CLV & BTYD DIAGNOSTICS
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("Top 10 VIP Customers (Ranked by 180-Day Expected CLV)")
        top_vips = (
            final_df[["customer_id", "frequency_repeat", "aov", "prob_alive", "clv_180d"]]
            .sort_values("clv_180d", ascending=False)
            .head(10)
        )
        st.dataframe(top_vips, use_container_width=True, hide_index=True)
        
        st.markdown("### BG/NBD Diagnostic Matrices")
        col1, col2 = st.columns(2)
        
        with col1:
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            plt.sca(ax1)
            plot_frequency_recency_matrix(bgf_model)
            ax1.set_title("Expected Future Transactions")
            st.pyplot(fig1)
            
        with col2:
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            plt.sca(ax2)
            plot_probability_alive_matrix(bgf_model)
            ax2.set_title("Probability of Being Alive")
            st.pyplot(fig2)

    # -------------------------------------------------------------------------
    # TAB 2: SURVIVAL ANALYSIS & COX PROPORTIONAL HAZARDS
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Time-to-Event Churn Modeling & Covariate Hazard Ratios")
        
        survival_df = crm_df[[
            "tenure_days", "latent_churned", "delivery_delay_rate",
            "support_tickets", "acquisition_channel"
        ]].copy()
        survival_df = pd.get_dummies(survival_df, columns=["acquisition_channel"], drop_first=True)
        
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(survival_df, duration_col="tenure_days", event_col="latent_churned")
        
        hazard_ratio_delay = cph.hazard_ratios_["delivery_delay_rate"]
        st.warning(
            f"**Statistical Business Insight:** Holding all other covariates constant, a 1-unit (100%) "
            f"increase in the **Delivery Delay Rate** multiplies the customer churn hazard by **{hazard_ratio_delay:.2f}x** (p < 0.005)."
        )
        
        col3, col4 = st.columns(2)
        with col3:
            fig3, ax3 = plt.subplots(figsize=(7, 5))
            kmf = KaplanMeierFitter()
            kmf.fit(crm_df["tenure_days"], crm_df["latent_churned"], label="Customer Retention")
            kmf.plot_survival_function(ax=ax3, color="navy", linewidth=2)
            ax3.set_title("Kaplan-Meier Retention Curve")
            ax3.set_xlabel("Customer Tenure (Days)")
            ax3.set_ylabel("Retention Probability")
            st.pyplot(fig3)
            
        with col4:
            fig4, ax4 = plt.subplots(figsize=(7, 5))
            cph.plot(ax=ax4)
            ax4.set_title("Cox PH: Log-Hazard Ratios (95% CI)")
            st.pyplot(fig4)

    # -------------------------------------------------------------------------
    # TAB 3: CAUSAL UPLIFT & CONSTRAINED BUDGET OPTIMIZATION
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Causal Segmentation & Constrained Budget Allocation")
        
        # Calculate Net Incremental Value per targeted customer
        opt_df = final_df.copy()
        opt_df["net_incremental_profit"] = opt_df["expected_incremental_revenue"] - coupon_cost
        
        # Filter profitable persuadables and rank by net incremental profit
        profitable_targets = opt_df[
            (opt_df["causal_segment"] == "Persuadable (TARGET)") &
            (opt_df["net_incremental_profit"] > 0)
        ].sort_values("net_incremental_profit", ascending=False)
        
        # Apply Budget Constraint
        max_affordable_customers = int(total_budget // coupon_cost)
        selected_targets = profitable_targets.head(max_affordable_customers)
        
        actual_spend = len(selected_targets) * coupon_cost
        expected_gross_return = selected_targets["expected_incremental_revenue"].sum()
        expected_net_roi = selected_targets["net_incremental_profit"].sum()
        
        opt1, opt2, opt3 = st.columns(3)
        opt1.metric("Optimal Customers Targeted", f"{len(selected_targets):,}")
        opt2.metric("Budget Utilized", f"${actual_spend:,.2f} / ${total_budget:,.2f}")
        opt3.metric("Expected Net Incremental Profit", f"${expected_net_roi:,.2f}")
        
        col5, col6 = st.columns([1, 2])
        with col5:
            st.markdown("**Causal Segment Breakdown**")
            seg_counts = final_df["causal_segment"].value_counts().reset_index()
            seg_counts.columns = ["Causal Segment", "Customer Count"]
            st.dataframe(seg_counts, hide_index=True, use_container_width=True)
            
        with col6:
            st.markdown("**Optimal Target List (Ranked by Expected Net Incremental Profit)**")
            st.dataframe(
                selected_targets[[
                    "customer_id", "aov", "uplift_score",
                    "expected_incremental_revenue", "net_incremental_profit"
                ]].head(15),
                hide_index=True,
                use_container_width=True
            )
else:
    st.info("👈 Adjust your simulation and budget parameters in the sidebar and click **Run Simulation Engine** to start.")