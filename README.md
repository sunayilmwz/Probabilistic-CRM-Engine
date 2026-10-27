# Probabilistic CRM, Survival & Causal Uplift Engine

Welcome! If you've ever worked with retail or e-commerce data, you know the classic CRM struggle: **Customers don't tell you when they leave.** They just silently stop buying. 

**[Try the Live Web App](https://probabilistic-crm-engine.streamlit.app/)**

Standard RFM (Recency, Frequency, Monetary) models only take a snapshot of the *past*. In this project, I built a statistical decision engine to predict the *future*. Instead of just asking "What did the customer do?", this engine answers three critical business questions:
1. **Are they still alive?** (Stochastic CLV)
2. **When will they churn, and why?** (Survival Analysis)
3. **Who actually needs a discount to stay?** (Causal Inference)

---

## The Journey & Methodology

### 1. Building a Realistic Retail World (Data Generating Process)
I didn't want to use a perfectly clean, unrealistic Kaggle dataset. Instead, I engineered a custom Data Generating Process (DGP) using stochastic distributions. 
* Purchase frequencies are modeled with a Poisson process.
* Monetary values are heavily right-skewed (Log-Normal).
* I validated this synthetic world using non-parametric tests (Mann-Whitney U) to ensure it mirrors real-life retail chaos.

### 2. The "Buy 'Til You Die" Framework (BG/NBD & Gamma-Gamma)
How do you separate a "sleeping" customer from a "dead" one? 
Using the `lifetimes` library, I modeled the customer journey as two latent processes: the urge to buy (Gamma) and the probability of dropping out (Beta). This allowed me to predict the exact **Probability Alive** and calculate the **180-day Expected CLV** using Discounted Cash Flow.

### 3. Survival Analysis (Because Time Matters)
Knowing a customer might churn isn't enough; we need to know *what* accelerates their "death." 
I brought biostatistics into CRM by applying the **Kaplan-Meier Estimator** and a **Cox Proportional Hazards (Cox PH)** model. 
* **Business Insight:** The model statistically proved that a 1-unit increase in delivery delays multiplies the customer's churn hazard by **1.89x**. 

### 4. Causal Uplift Modeling (Smart Discounts)
Marketing teams often want to send a 20% discount to everyone who is at risk of churning. *This is a trap.*
Using a **T-Learner (Two-Model Approach with Random Forests)**, I calculated the Conditional Average Treatment Effect (CATE) for every single customer. This segments the customer base into four causal groups:
* **Sure Things:** Will buy anyway (Don't waste budget).
* **Lost Causes:** Won't buy regardless (Don't waste budget).
* **Sleeping Dogs:** Will churn *because* you bothered them.
* **Persuadables:** Will only buy IF you send the discount. 👉 **(Our Target!)**

## Tech Stack
`Python` | `Lifetimes` | `Lifelines` | `Scikit-Learn` | `SciPy` | `Pandas` 

---
*Feel free to explore the code to see how mathematical theory meets retail strategy!*
