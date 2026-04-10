# The Experiment Architect

**Plan. Execute. Analyze.** The Decision Scientist's Companion.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B)
![OpenAI](https://img.shields.io/badge/AI-GPT--4o-green)
![SciPy](https://img.shields.io/badge/Stats-SciPy-orange)

## Overview

**The Experiment Architect** combines statistical rigor (SciPy/NumPy) with LLM-powered semantic mapping (OpenAI, Anthropic, or Gemini).

Key capabilities:
1.  **Reverse MDE Calculator:** Tells you what's actually detectable given your traffic and time constraints
2.  **Causal Inference Selector:** Guides you to the right quasi-experimental method (DiD, RDD, PSM)
3.  **Hybrid Analysis:** LLMs handle semantic tasks (column mapping), Python handles all math (no hallucinated statistics)

---

## Core Philosophy

### 1. LLMs for Semantics, Python for Math
LLMs excel at understanding column names and context. Python handles all statistical calculations.
* **❌ Wrong:** Asking LLM "Is this result significant?"
* **✅ Right:** LLM maps columns, then `scipy.stats.chi2_contingency` calculates the p-value

### 2. Reverse MDE Calculator
Instead of guessing effect sizes, input your constraints (traffic, time) and calculate what's actually detectable.

---

## Features

### Tab 1: Design Experiment
* **A/B Test Calculator:**
    * Sample size & duration estimation
    * Unequal split penalty calculator (efficiency loss of 80/20 vs 50/50)
    * Sanity check: Flags unrealistic effect sizes based on industry benchmarks
* **Causal Inference Selector:**
    * Expert system selects the right quasi-experimental method (DiD, RDD, PSM)
    * Full DiD and RDD implementations with file upload
    * Code generator for PSM and CausalImpact

### Tab 2: Analyze Results
* **Manual Stats Input:**
    * Frequentist analysis (p-values, confidence intervals, effect sizes)
    * Bayesian analysis (probability, expected loss)
    * Sample Ratio Mismatch detection
* **CSV Analyzer:**
    * Automatic column mapping
    * Chi-squared test for binary outcomes
    * Welch's T-test for continuous outcomes
* **SQL Code Generator:**
    * Generates Jupyter notebooks for BigQuery, Snowflake, Redshift

---

## Installation & Setup

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
    # Choose your LLM provider (openai, anthropic, or gemini)
    LLM_PROVIDER=openai
    OPENAI_API_KEY=sk-your-openai-key-here

    # OR use Anthropic/Claude
    # LLM_PROVIDER=anthropic
    # ANTHROPIC_API_KEY=sk-your-key-here

    # OR use Google Gemini
    # LLM_PROVIDER=gemini
    # GEMINI_API_KEY=your-key-here
    ```

4.  **Run the App**
    ```bash
    streamlit run app.py
    ```

5.  **Run Tests (Optional)**
    ```bash
    pytest tests/ -v
    # 26 tests, 100% pass rate
    ```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for details.

---

## Usage Examples

### Example 1: Unrealistic Target Detection
**Input:** 1% lift, 1,000 daily visitors, 1 week wait
**Output:**
- Calculated MDE: 15%
- Sanity check: "Unrealistic - a 15% lift on checkout is extremely rare"

### Example 2: CSV Analysis
**Input:** Upload CSV with columns `user_uuid`, `bucket_id`, `has_purchased`
**Output:**
- Maps: variant=`bucket_id`, outcome=`has_purchased` (binary)
- Runs Chi-squared test
- Result: "Variant B wins (p=0.03)"

---

## License

MIT License. Built for the Open Source Data Science community.

---

## Contact

Tomasz Solis
- [LinkedIn](https://linkedin.com/in/tomaszsolis)
- [GitHub](https://github.com/tomasz-solis)
