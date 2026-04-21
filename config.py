"""Configuration constants for the Experiment Architect."""

# Statistical constants
Z_ALPHA = 1.96  # 95% confidence level
Z_BETA = 0.84   # 80% power
ALPHA = 0.05    # Significance threshold

# Bayesian sampling
BAYESIAN_SAMPLES = 100000

# Streamlit config
PAGE_TITLE = "Test Architect"
PAGE_ICON = "🏗️"
PAGE_LAYOUT = "wide"

# LLM Models
MODEL_OPENAI = "gpt-4o"
MODEL_ANTHROPIC = "claude-3-5-sonnet-20241022"
MODEL_GEMINI = "gemini-1.5-pro"

# LLM Config
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048
