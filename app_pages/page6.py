import streamlit as st

def page6_body():
    """Display project hypothesis and conclusions page."""

    st.write("## Project Hypothesis and Conclusions")

    st.info(
        "### Hypothesis 1: House Size and Price Correlation\n"        
        "**Hypothesis:** Larger houses drive higher sale prices.\n\n"
        "**Validation:**\n"
        "* Spearman and Pearson correlation analysis to measure the "
        "strength of relationship\n"
        "* Scatter plots showing the relationship between square footage "
        "and sale price\n"
        "* Parallel plot visualization to show how size categories relate "
        "to price ranges\n\n"
        "**Result:** Validated - GrLivArea (living area) shows strong "
        "positive correlation (>0.70) with sale price\n\n")
        
    st.info("### Hypothesis 2: Quality Ratings Impact Price\n"
        "**Hypothesis:** Houses with higher overall quality ratings have "
        "higher sale prices.\n\n"
        "**Validation:**\n"
        "* Correlation analysis between OverallQual and SalePrice\n"
        "* Visualization of price distributions across different quality "
        "ratings\n"
        "* Feature importance analysis from the ML model\n\n"
        "**Result:** Validated - OverallQual is the most important feature"
        " with the strongest correlation to price\n\n")

    st.info("### Hypothesis 3: Year Built Affects Price\n"
        "**Hypothesis:** Newer houses generally sell for higher prices than"
        " older properties.\n\n"
        "**Validation:**\n"
        "* Correlation analysis between YearBuilt and SalePrice\n"    
        "* Categorical analysis showing price trends across different "
        "construction periods\n"
        "* Inclusion in top features identified by the ML model\n\n"
        "**Result:** Validated - YearBuilt shows positive correlation and "
        "appears in top 5 important features\n\n")