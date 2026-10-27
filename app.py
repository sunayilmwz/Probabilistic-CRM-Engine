import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from cycler import cycler
from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_frequency_recency_matrix, plot_probability_alive_matrix
from lifelines import KaplanMeierFitter

# Import core statistical functions from the backend engine
from stockhastic import (
    generate_retail_crm_data,
    build_customer_summary,
    run_stochastic_clv_models,
    run_survival_analysis,
    run_causal_uplift_modeling,
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="CRM & Causal Analytics Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# THEME-AWARE STYLING (works in both light and dark mode)
# Rule: never hard-code text or surface colors. Inherit text color from the
# active Streamlit theme and use translucent surfaces so they adapt.
# =============================================================================
ACCENT = "#6366f1"       # indigo
ACCENT_SOFT = "rgba(99, 102, 241, 0.12)"
SURFACE = "rgba(128, 128, 128, 0.07)"
BORDER = "rgba(128, 128, 128, 0.22)"
MUTED = "rgba(128, 128, 128, 0.95)"

st.markdown(f"""
<style>
    /* Layout */
    .block-container {{
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }}

    /* Typography: inherit color from the active theme */
    h1, h2, h3, h4 {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Inter", sans-serif;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: inherit;
    }}
    h1 {{
        font-size: 2rem;
        margin-bottom: 0.25rem;
    }}
    h3 {{
        font-size: 1.15rem;
    }}
    h4 {{
        font-size: 1rem;
        margin-top: 1.25rem;
    }}

    /* Hero block */
    .hero {{
        padding: 1.5rem 1.75rem;
        border: 1px solid {BORDER};
        border-radius: 12px;
        background: linear-gradient(135deg, {ACCENT_SOFT} 0%, {SURFACE} 60%);
        margin-bottom: 1.5rem;
    }}
    .hero-eyebrow {{
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {ACCENT};
        margin-bottom: 0.4rem;
    }}
    .hero-title {{
        font-size: 1.9rem;
        font-weight: 650;
        letter-spacing: -0.025em;
        line-height: 1.2;
        margin: 0 0 0.5rem 0;
        color: inherit;
    }}
    .hero-sub {{
        font-size: 0.95rem;
        line-height: 1.55;
        color: {MUTED};
        margin: 0;
        max-width: 60rem;
    }}

    /* Metric cards: translucent surface + inherited text color */
    [data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 3px solid {ACCENT};
        padding: 16px 20px;
        border-radius: 10px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {MUTED};
    }}
    [data-testid="stMetricLabel"] p {{
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }}
    [data-testid="stMetricValue"] {{
        color: inherit;
        font-weight: 650;
        letter-spacing: -0.02em;
    }}

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 1px solid {BORDER};
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 44px;
        padding: 0 18px;
        background: transparent;
        border-radius: 8px 8px 0 0;
        font-weight: 500;
        color: {MUTED};
    }}
    .stTabs [aria-selected="true"] {{
        color: inherit;
        font-weight: 600;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {ACCENT};
        height: 2px;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] {{
        border-right: 1px solid {BORDER};
    }}
    [data-testid="stSidebar"] h2 {{
        font-size: 1.05rem;
    }}
    [data-testid="stSidebar"] h3 {{
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {MUTED};
    }}

    /* Primary button */
    .stButton > button[kind="primary"] {{
        background: {ACCENT};
        border: 1px solid {ACCENT};
        color: #ffffff;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.55rem 1rem;
        transition: filter 0.15s ease, transform 0.05s ease;
    }}
    .stButton > button[kind="primary"]:hover {{
        filter: brightness(1.1);
        border-color: {ACCENT};
        color: #ffffff;
    }}
    .stButton > button[kind="primary"]:active {{
        transform: translateY(1px);
    }}

    /* Dataframes */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 10px;
        overflow: hidden;
    }}

    /* Section caption */
    .section-note {{
        font-size: 0.88rem;
        color: {MUTED};
        margin: -0.25rem 0 1rem 0;
    }}

    /* Horizontal rule */
    hr {{
        border-color: {BORDER};
    }}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# MATPLOTLIB THEME: transparent canvas + neutral mid-gray text so charts
# remain readable on both light and dark backgrounds
# =============================================================================
NEUTRAL = "#8b93a1"
plt.rcParams.update({
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.facecolor": "none",
    "savefig.transparent": True,
    "text.color": NEUTRAL,
    "axes.labelcolor": NEUTRAL,
    "axes.titlecolor": NEUTRAL,
    "axes.edgecolor": NEUTRAL,
    "xtick.color": NEUTRAL,
    "ytick.color": NEUTRAL,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": NEUTRAL,
    "grid.alpha": 0.18,
    "grid.linewidth": 0.8,
    "axes.titlesize": 11,
    "axes.titleweight": "600",
    "axes.labelsize": 10,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "axes.prop_cycle": cycler(color=["#6366f1", "#22c55e", "#f59e0b", "#ef4444", "#06b6d4"]),
})


def section_note(text):
    st.markdown(f"<p class='section-note'>{text}</p>", unsafe_allow_html=True)


# =============================================================================
# HERO HEADER
# =============================================================================
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Decision Support</div>
    <div class="hero-title">Customer Analytics and Causal Uplift Engine</div>
    <p class="hero-sub">
        Integrates Buy 'Til You Die (BG/NBD) stochastic modeling, Cox proportional
        hazards survival analysis, and T-Learner causal inference to optimize
        enterprise marketing spend.
    </p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR: SIMULATION & OPTIMIZATION PARAMETERS
# =============================================================================
st.sidebar.header("Configuration")

n_customers = st.sidebar.slider(
    "Customer Sample Size (N)",
    min_value=1000,
    max_value=10000,
    value=5000,
    step=1000,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Campaign Constraints")

total_budget = st.sidebar.number_input(
    "Total Marketing Budget ($)",
    min_value=500,
    max_value=100000,
    value=10000,
    step=500,
)

coupon_cost = st.sidebar.slider(
    "Incentive Cost per Customer ($)",
    min_value=5,
    max_value=100,
    value=25,
    step=5,
)

st.sidebar.markdown("---")
run_clicked = st.sidebar.button("Run Model Pipeline", type="primary", use_container_width=True)


# =============================================================================
# CACHED PIPELINE (DataFrames ONLY - fitted model objects are NOT picklable)
# =============================================================================
@st.cache_data(show_spinner=False)
def execute_crm_pipeline(n_cust):
    """Runs the simulation and returns pickle-serializable DataFrames only."""
    customers_df, transactions_df = generate_retail_crm_data(n_customers=n_cust)
    crm_df = build_customer_summary(customers_df, transactions_df)

    clv_result = run_stochastic_clv_models(crm_df)
    clv_df = clv_result[0] if isinstance(clv_result, tuple) else clv_result

    final_df = run_causal_uplift_modeling(clv_df)
    return crm_df, final_df


def fit_bgf(final_df):
    """Re-fits the BG/NBD model outside the cache (takes well under a second)."""
    subset = final_df[final_df["frequency_repeat"] > 0]
    bgf = BetaGeoFitter(penalizer_coef=0.01)
    bgf.fit(
        subset["frequency_repeat"],
        subset["recency_clv_days"],
        subset["tenure_days"],
    )
    return bgf


# =============================================================================
# RUN PIPELINE (results kept in session_state so widgets don't reset the view)
# =============================================================================
if run_clicked:
    with st.spinner("Executing stochastic simulation and fitting econometric models..."):
        crm_df, final_df = execute_crm_pipeline(n_customers)
        bgf_model = fit_bgf(final_df)
        cph_model = run_survival_analysis(crm_df)

    st.session_state["results"] = {
        "crm_df": crm_df,
        "final_df": final_df,
        "bgf_model": bgf_model,
        "cph_model": cph_model,
    }
    st.success("Pipeline executed successfully. All models converged.")

# =============================================================================
# DASHBOARD
# =============================================================================
if "results" in st.session_state:
    res = st.session_state["results"]
    crm_df = res["crm_df"]
    final_df = res["final_df"]
    bgf_model = res["bgf_model"]
    cph_model = res["cph_model"]

    # Top-level KPIs
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Active Customer Base", f"{len(crm_df):,}")
    kpi2.metric("CLV Eligible Base", f"{len(final_df):,}")
    kpi3.metric("Mean 180-Day Expected CLV", f"${final_df['clv_180d'].mean():,.2f}")
    kpi4.metric(
        "Persuadable Segment",
        f"{(final_df['causal_segment'] == 'Persuadable (TARGET)').sum():,}",
    )

    st.markdown("<div style='height: 1rem'></div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs([
        "Stochastic CLV (BG/NBD)",
        "Survival and Hazard Analysis",
        "Causal Uplift and Budget Optimizer",
    ])

    # -------------------------------------------------------------------------
    # TAB 1: STOCHASTIC CLV & BTYD DIAGNOSTICS
    # -------------------------------------------------------------------------
    with tab1:
        st.markdown("### Top Customers by Expected CLV")
        section_note("Ten customers with the highest projected 180-day lifetime value.")

        top_vips = (
            final_df[["customer_id", "frequency_repeat", "aov", "prob_alive", "clv_180d"]]
            .sort_values("clv_180d", ascending=False)
            .head(10)
        )
        st.dataframe(
            top_vips,
            use_container_width=True,
            hide_index=True,
            column_config={
                "customer_id": st.column_config.TextColumn("Customer"),
                "frequency_repeat": st.column_config.NumberColumn("Repeat Orders", format="%d"),
                "aov": st.column_config.NumberColumn("Avg Order Value", format="$%.2f"),
                "prob_alive": st.column_config.ProgressColumn(
                    "Probability Alive", min_value=0.0, max_value=1.0, format="%.2f"
                ),
                "clv_180d": st.column_config.NumberColumn("180-Day CLV", format="$%.2f"),
            },
        )

        st.markdown("#### Diagnostic Matrices")
        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots(figsize=(7, 5))
            plt.sca(ax1)
            plot_frequency_recency_matrix(bgf_model)
            ax1.set_title("Expected Future Transactions")
            ax1.grid(False)
            st.pyplot(fig1)
            plt.close(fig1)

        with col2:
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            plt.sca(ax2)
            plot_probability_alive_matrix(bgf_model)
            ax2.set_title("Probability of Being Alive")
            ax2.grid(False)
            st.pyplot(fig2)
            plt.close(fig2)

    # -------------------------------------------------------------------------
    # TAB 2: SURVIVAL & HAZARD ANALYSIS
    # -------------------------------------------------------------------------
    with tab2:
        st.markdown("### Survival Curves by Acquisition Channel")
        section_note("Kaplan-Meier estimates of the probability a customer is still active over time.")

        fig3, ax3 = plt.subplots(figsize=(10, 4.5))
        for channel, grp in crm_df.groupby("acquisition_channel"):
            kmf = KaplanMeierFitter()
            kmf.fit(grp["tenure_days"], event_observed=grp["latent_churned"], label=channel)
            kmf.plot_survival_function(ax=ax3, ci_show=True, ci_alpha=0.12)
        ax3.set_xlabel("Tenure (days)")
        ax3.set_ylabel("Survival probability")
        ax3.set_title("Kaplan-Meier Estimates")
        ax3.legend(title="Channel")
        st.pyplot(fig3)
        plt.close(fig3)

        st.markdown("#### Churn Drivers (Cox Proportional Hazards)")
        section_note("A hazard ratio above 1 increases churn risk; below 1 reduces it.")

        col3, col4 = st.columns(2)

        with col3:
            summary = cph_model.summary[["exp(coef)", "p"]].rename(
                columns={"exp(coef)": "Hazard Ratio", "p": "p-value"}
            )
            st.dataframe(summary.round(4), use_container_width=True)

        with col4:
            fig4, ax4 = plt.subplots(figsize=(6, 4))
            cph_model.plot(hazard_ratios=True, ax=ax4)
            ax4.set_title("Hazard Ratios (95% CI)")
            st.pyplot(fig4)
            plt.close(fig4)

    # -------------------------------------------------------------------------
    # TAB 3: CAUSAL UPLIFT & CONSTRAINED BUDGET OPTIMIZATION
    # -------------------------------------------------------------------------
    with tab3:
        st.markdown("### Targeted Campaign Optimization")
        section_note(
            "Selects Persuadable customers with positive net incremental profit, "
            "ranked by profit and capped by the available budget."
        )

        opt_df = final_df.copy()
        opt_df["net_incremental_profit"] = opt_df["expected_incremental_revenue"] - coupon_cost

        profitable_targets = opt_df[
            (opt_df["causal_segment"] == "Persuadable (TARGET)")
            & (opt_df["net_incremental_profit"] > 0)
        ].sort_values("net_incremental_profit", ascending=False)

        max_affordable_customers = int(total_budget // coupon_cost)
        selected_targets = profitable_targets.head(max_affordable_customers)

        actual_spend = len(selected_targets) * coupon_cost
        expected_net_roi = selected_targets["net_incremental_profit"].sum()

        opt1, opt2, opt3 = st.columns(3)
        opt1.metric("Targeted Segment Count", f"{len(selected_targets):,}")
        opt2.metric("Allocated Budget", f"${actual_spend:,.0f} / ${total_budget:,.0f}")
        opt3.metric("Projected Net Profit", f"${expected_net_roi:,.2f}")

        st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

        col5, col6 = st.columns([1, 2])
        with col5:
            st.markdown("#### Causal Profiling")
            seg_counts = final_df["causal_segment"].value_counts().reset_index()
            seg_counts.columns = ["Segment", "Count"]
            st.dataframe(seg_counts, hide_index=True, use_container_width=True)

        with col6:
            st.markdown("#### Optimized Target List")
            st.dataframe(
                selected_targets[[
                    "customer_id", "aov", "uplift_score",
                    "expected_incremental_revenue", "net_incremental_profit",
                ]].head(15),
                hide_index=True,
                use_container_width=True,
                column_config={
                    "customer_id": st.column_config.TextColumn("Customer"),
                    "aov": st.column_config.NumberColumn("Avg Order Value", format="$%.2f"),
                    "uplift_score": st.column_config.NumberColumn("Uplift Score", format="%.3f"),
                    "expected_incremental_revenue": st.column_config.NumberColumn(
                        "Expected Incremental Revenue", format="$%.2f"
                    ),
                    "net_incremental_profit": st.column_config.NumberColumn(
                        "Net Incremental Profit", format="$%.2f"
                    ),
                },
            )
else:
    st.markdown(
        f"<p style='color: {MUTED}; text-align: center; margin-top: 4rem;'>"
        "Configure parameters in the sidebar and run the pipeline to initialize the dashboard."
        "</p>",
        unsafe_allow_html=True,
    )