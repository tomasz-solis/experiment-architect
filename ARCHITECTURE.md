# Architecture Documentation

## Overview
The Experiment Architect is a production-ready Streamlit application with a clean modular architecture.

---

## Current Structure

```
experimentation-architect/
├── app.py (669 lines)              # Main Streamlit app - UI orchestration only
├── config.py (15 lines)            # All configuration constants
├── llm/                            # LLM abstraction layer
│   ├── __init__.py
│   ├── client.py (98 lines)        # Factory, initialization, API wrapper
│   └── providers.py (67 lines)     # OpenAI, Anthropic, Gemini providers
├── stats/                          # Statistical computation layer
│   ├── __init__.py
│   ├── frequentist.py (175 lines)  # Tests, CI, effect sizes, sample size
│   ├── bayesian.py (64 lines)      # Beta-Binomial analysis
│   └── causal.py (105 lines)       # DiD, RDD, method selector
└── ui/                             # UI helper components
    ├── __init__.py
    └── components.py (66 lines)    # Shared display functions
```

**Total**: 1,253 lines of production code

---

## Design Principles

### 1. DRY (Don't Repeat Yourself)
- **config.py**: Single source of truth for all constants
- **No duplication**: app.py imports from modules, doesn't redefine logic
- **Reusable functions**: All stats/LLM code can be imported elsewhere

### 2. Separation of Concerns
- **app.py**: UI orchestration and Streamlit-specific code only
- **stats/**: Pure Python statistical functions (no Streamlit dependencies)
- **llm/**: LLM client abstraction (provider-agnostic)
- **config.py**: Configuration constants only

### 3. Lazy Loading
- Optional dependencies (statsmodels, anthropic, google-generativeai) load only when used
- Clear error messages if packages are missing
- Graceful degradation (app works without AI features if no API key)

### 4. Production Ready
- Proper error handling with try/except
- Clear, concise error messages
- No magic numbers (all constants in config.py)

---

## Module Details

### config.py
**Purpose**: Central configuration

```python
# Statistical constants
Z_ALPHA = 1.96  # 95% confidence
Z_BETA = 0.84   # 80% power
ALPHA = 0.05    # Significance threshold

# Bayesian sampling
BAYESIAN_SAMPLES = 100000

# Streamlit config
PAGE_TITLE = "Experiment Architect"
PAGE_ICON = "🏗️"
PAGE_LAYOUT = "centered"
```

**Why**: Single source of truth. Change Z_ALPHA once, it updates everywhere.

---

### llm/ (LLM Abstraction Layer)

#### client.py
**Purpose**: LLM client factory and API wrapper

**Key Functions**:
- `get_llm_provider()` - Read from .env or Streamlit secrets
- `get_api_key(provider)` - Fetch API key for provider
- `create_llm_client()` - Factory that returns (client, enabled, provider)
- `ask_agent(...)` - Unified API for all providers

**Example**:
```python
from llm.client import create_llm_client, ask_agent

client, ai_enabled, provider = create_llm_client()
response = ask_agent(client, provider, ai_enabled,
                    "You are a statistician",
                    "Explain p-values")
```

#### providers.py
**Purpose**: Provider implementations

**Classes**:
- `OpenAIProvider` - GPT-4o
- `AnthropicProvider` - Claude 3.5 Sonnet
- `GeminiProvider` - Gemini 1.5 Pro

**Pattern**: Each provider has a `call(system_role, user_prompt, json_mode)` method.

---

### stats/ (Statistical Computation Layer)

#### frequentist.py
**Purpose**: Frequentist statistical tests

**Key Functions**:
- `chi_squared_test(...)` - For binary outcomes
- `welch_t_test(...)` - For continuous outcomes
- `confidence_interval_binary(...)` - CI on lift for conversion rates
- `confidence_interval_continuous(...)` - CI on lift for means
- `check_srm(...)` - Sample Ratio Mismatch detection
- `calculate_sample_size(...)` - Sample size & duration estimation
- `calculate_reverse_mde(...)` - Reverse MDE calculator
- `calculate_lift(...)` - Relative lift calculation
- `is_significant(p_value)` - Significance check

**Example**:
```python
from stats.frequentist import chi_squared_test

result = chi_squared_test(
    successes_a=100, failures_a=900,
    successes_b=115, failures_b=885
)
# Returns: {"p_value": 0.03, "effect_size": 0.048, ...}
```

#### bayesian.py
**Purpose**: Bayesian analysis

**Key Functions**:
- `beta_binomial_analysis(...)` - Beta-Binomial conjugate prior
- `get_decision_recommendation(prob)` - Decision logic

**Returns**:
- `prob_b_wins` - P(Variant > Control)
- `expected_loss` - Expected loss if you ship variant B
- Posterior parameters (alpha_a, beta_a, alpha_b, beta_b)

**Example**:
```python
from stats.bayesian import beta_binomial_analysis

result = beta_binomial_analysis(
    successes_a=100, failures_a=900,
    successes_b=115, failures_b=885
)
# Returns: {"prob_b_wins": 0.93, "expected_loss": 0.002, ...}
```

#### causal.py
**Purpose**: Causal inference methods

**Key Functions**:
- `difference_in_differences(df, ...)` - DiD regression
- `regression_discontinuity(df, ...)` - RDD regression
- `select_causal_method(...)` - Expert system for method selection

**Pattern**: Lazy statsmodels imports with helpful error messages.

**Example**:
```python
from stats.causal import difference_in_differences

result = difference_in_differences(
    df=panel_data,
    time_col='date',
    treatment_col='treated',
    outcome_col='revenue',
    intervention_point='2024-01-01'
)
# Returns: {"coefficient": 5.2, "p_value": 0.01, "ci_lower": 1.3, "ci_upper": 9.1, ...}
```

---

### ui/ (UI Helper Components)

#### components.py
**Purpose**: Reusable UI display functions

**Key Functions**:
- `show_data_quality(df)` - Display metrics: rows, missing values, duplicates
- `show_data_preview(df, n_rows)` - Show first N rows
- `show_srm_warning(ratio)` - Display SRM warning if needed
- `show_frequentist_results(...)` - Display frequentist test results
- `show_bayesian_results(...)` - Display Bayesian analysis results
- `show_bayesian_decision(prob, group_name)` - Decision recommendation

**Example**:
```python
from ui.components import show_data_quality

show_data_quality(df)  # Displays Streamlit metrics
```

---

### app.py (Main Application)

**Purpose**: UI orchestration only

**Structure**:
```python
# Import all modules
from config import Z_ALPHA, Z_BETA, ALPHA, BAYESIAN_SAMPLES
from llm.client import create_llm_client, ask_agent as llm_ask_agent
from scipy.stats import ttest_ind, chi2_contingency, beta

# Initialize LLM client
client, ai_enabled, llm_provider = create_llm_client()

# Thin wrapper for consistency
def ask_agent(system_role, user_prompt, json_mode=False):
    return llm_ask_agent(client, llm_provider, ai_enabled, system_role, user_prompt, json_mode)

# UI Layout
st.title("🏗️ The Experiment Architect")
st.markdown("**Plan. Execute. Analyze.** The Decision Scientist's Companion.")

tab_design, tab_analyze = st.tabs(["Design Experiment", "Analyze Results"])

# Tab 1: Design
# - A/B test calculator
# - Reverse MDE wizard
# - Causal method selector
# - DiD/RDD implementations

# Tab 2: Analyze
# - Manual stats input (frequentist + Bayesian)
# - CSV analyzer (automatic column mapping)
# - SQL notebook generator
```

**No business logic** - app.py is pure UI orchestration. All calculations happen in stats/ modules.

---

## Testing & Verification

### Import Test
```bash
source venv/bin/activate
python -c "
from config import Z_ALPHA, BAYESIAN_SAMPLES
from llm.client import create_llm_client
from stats.frequentist import chi_squared_test
from stats.bayesian import beta_binomial_analysis
from stats.causal import difference_in_differences
print('✓ All imports successful')
"
```

### Run App
```bash
streamlit run app.py
```

---

## Dependencies

### Core (Required)
- streamlit
- pandas
- numpy
- scipy
- python-dotenv
- openai (if using OpenAI)

### Optional
- **anthropic** - Only if LLM_PROVIDER=anthropic
- **google-generativeai** - Only if LLM_PROVIDER=gemini
- **statsmodels** - Only for DiD/RDD analysis (lazy loaded)

All optional dependencies have graceful error messages if missing.

---

## Architecture Benefits

### 1. Testable
Each stats function can be unit tested independently:
```python
def test_chi_squared():
    result = chi_squared_test(100, 900, 115, 885)
    assert result['p_value'] < 0.05
    assert result['effect_size'] > 0
```

### 2. Reusable
Import stats functions in Jupyter notebooks:
```python
from stats.frequentist import calculate_sample_size

sample_size = calculate_sample_size(
    baseline=0.10,
    mde=0.10,
    daily_traffic=5000
)
```

### 3. Maintainable
Change Chi-squared implementation? Edit one file (stats/frequentist.py). All uses update automatically.

### 4. Flexible
Swap LLM providers via .env without touching code:
```bash
# .env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-...
```