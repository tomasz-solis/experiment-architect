# 🏗️ The Experiment Architect

**Plan. Execute. Analyze.** The Decision Scientist's Companion.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B)
![OpenAI](https://img.shields.io/badge/AI-GPT--4o-green)
![SciPy](https://img.shields.io/badge/Stats-SciPy-orange)

## 🚀 Overview

**The Experiment Architect** is a hybrid intelligence tool designed to solve the "Blank Page Problem" in Experimentation. It bridges the gap between **Statistical Rigor** (SciPy/NumPy) and **Operational Speed** (OpenAI GPT-4).

Unlike standard A/B testing calculators, this tool acts as a **Data Scientist in a box**:
1.  **Prevents Bad Tests:** Calculates "Reverse MDE" to tell stakeholders what is actually detectable.
2.  **Solves the "Can't Randomize" Problem:** Guides users to the correct Causal Inference method (DiD, RDD, PSM).
3.  **Eliminates Hallucinations:** Uses LLMs for semantic mapping, but executes all statistical tests using deterministic Python code.

---

## 🧠 Core Philosophy: The "Hybrid" Architecture

This tool was built to answer a critical interview question: *"How do you use AI without trusting it blindly?"*

### 1. The "Air Gap" Strategy
We never ask the LLM to do math. LLMs are excellent at **Semantics** (reading column names, writing boilerplate) but terrible at **Arithmetic** (calculating p-values).
* **❌ Wrong:** Asking ChatGPT "Is this result significant?"
* **✅ Right:** Asking ChatGPT "Map these columns to 'Variant' and 'Outcome'", then passing those columns to `scipy.stats.ttest_ind`.

### 2. The "Reverse" MDE Wizard
Most PMs guess their Minimum Detectable Effect (MDE). This tool inverts the formula:
> *"Tell me how long you are willing to wait, and I will tell you the smallest effect size you can detect."*

---

## 🛠️ Features

### Tab 1: Design Strategy
* **A/B Test Calculator:**
    * Sample size & Duration estimation.
    * **Unequal Split Warning:** Calculates the exact "Time Tax" (efficiency loss) of running 80/20 splits vs 50/50.
    * **🚦 AI Sanity Check:** An agent analyzes the parameters (Baseline vs. Target Lift) and flags unrealistic goals based on industry benchmarks.
* **Causal Inference Selector:**
    * An Expert System (Logic Tree) that selects the right Quasi-Experiment (RDD, DiD, PSM) based on user constraints.
    * **Code Generator Agent:** Writes the implementation code (e.g., `CausalImpact` or `DoWhy`) customized to the user's specific SQL dialect (Snowflake/BigQuery).

### Tab 2: Analysis & Decision
* **Hallucination-Free CSV Analyzer:**
    * Upload raw results.
    * Agent maps semantic columns (e.g., `total_eur` -> `Metric`, `grp` -> `Variant`).
    * **Python Engine** runs the appropriate test (Chi-Squared for binary, Welch's T-Test for continuous).
* **SQL-to-Notebook Generator:**
    * Converts raw SQL queries into a ready-to-run Jupyter Notebook snippet with Bayesian analysis built-in.

---

## ⚙️ Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/tomasz-solis/experiment-architect.git](https://github.com/tomasz-solis/experiment-architect.git)
    cd experiment-architect
    ```

2.  **Install requirements**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Configure API Key**
    Create a file named `.env` in the root directory:
    ```bash
    OPENAI_API_KEY=sk-your-openai-key-here
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

---

## 📸 Usage Scenarios

### Scenario A: The "Aggressive" PM
* **User Input:** Wants to detect a 1% lift on a Checkout page with 1,000 daily visitors in 1 week.
* **Tool Output:** * *Math:* "Impossible. Minimum detectable lift is 15%."
    * *AI Sanity Check:* 🛑 **Unrealistic.** "A 15% lift on a mature checkout page is highly unlikely."

### Scenario B: The "Messy Data" Analysis
* **User Input:** Uploads a CSV with headers `user_uuid`, `bucket_id`, `has_purchased`.
* **Tool Action:** 1. AI identifies `bucket_id` as Variant and `has_purchased` as Outcome (Binary).
    2. Python executes a Chi-Squared test.
    3. Tool declares: "Variant B is winning (p=0.03)."

---

## 📜 License

MIT License. Built for the Open Source Data Science community.

---

## Contact

Tomasz Solis
- Email: tomasz.solis@gmail.com
- [LinkedIn](https://linkedin.com/in/tomaszsolis)
- [GitHub](https://github.com/tomasz-solis)