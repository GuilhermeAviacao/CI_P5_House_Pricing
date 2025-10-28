import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from feature_engine.encoding import OneHotEncoder
from feature_engine.discretisation import ArbitraryDiscretiser


# Data Load
def page2_body():
    """Page 2 - Data Exploration and Correlation Study"""

    st.write("## Data Exploration and Correlation Study")
    st.info(
        "This page shows the dataset inspection, sale price distribution, "
        "and correlation analysis to identify the most important features "
        "affecting house prices."
    )

    # Load data
    df = load_house_data()

    # Data Inspection
    st.write("---")
    st.write("### Dataset Inspection")
    data_inspection(df)

    # Sale Price Distribution
    st.write("---")
    st.write("### Sale Price Distribution")
    if st.checkbox("Show Sale Price Distribution"):
        sale_price_distribution(df)

    # Correlation Study
    st.write("---")
    st.write("### Correlation Study")
    if st.checkbox("Show Correlation Analysis"):
        correlation_study(df)


def load_house_data():
    """Load the house prices dataset - copied from notebook 01"""
    df = pd.read_csv("outputs/datasets/collection/HouseFeaturesPrices.csv")
    return df


def data_inspection(df):

    """Display dataset information - copied from notebook 01"""
    st.write(f"**Dataset Shape:** {df.shape[0]} rows, {df.shape[1]} columns")

    st.write("**Dataset Columns:**")
    st.write(list(df.columns))

    st.write("**First 5 rows:**")
    st.dataframe(df.head())

    st.write("**Dataset Information:**")
    st.text(f"Total entries: {df.shape[0]}")
    st.text(f"Total features: {df.shape[1]}")


def sale_price_distribution(df):
    """Show SalePrice distribution - copied from notebook 02"""
    st.write(
            "Price distribution skewed to the left, with a median lower "
            "than the mean. Long right tail with some outliers on the "
            "higher end / luxury houses. Box-plot depicts that most homes"
            "are around 100-200K.")
    
    if 'SalePrice' in df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram
        axes[0].hist(df['SalePrice'], bins=50, edgecolor='black')
        axes[0].set_xlabel('Sale Price')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Distribution of Sale Price')

        # Box plot
        axes[1].boxplot(df['SalePrice'])
        axes[1].set_ylabel('Sale Price')
        axes[1].set_title('Sale Price Box Plot')

        plt.tight_layout()
        st.pyplot(fig)

        st.write("**Sale Price Statistics:**")
        st.write(f"Mean: ${df['SalePrice'].mean():,.2f}")
        st.write(f"Median: ${df['SalePrice'].median():,.2f}")
        st.write(f"Std Dev: ${df['SalePrice'].std():,.2f}")
        st.write(f"Min: ${df['SalePrice'].min():,.2f}")
        st.write(f"Max: ${df['SalePrice'].max():,.2f}")


# Correlation Study

def correlation_study(df):
    """Correlation analysis - copied from notebook 03"""
    
    # Missing values treatment - copied from notebook 03
    df_filled = df.copy()
    object_cols = df.columns[df.dtypes == 'object'].to_list()

    # Fill missing values in object columns
    for col in object_cols:
        df_filled[col] = df_filled[col].fillna('Unknown')

    # For numeric columns, filling with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())

    # Encoding categorical variables - copied from notebook 03
    encoder = OneHotEncoder(variables=df_filled.columns[
        df_filled.dtypes == 'object'].to_list(), drop_last=False)
    df_ohe = encoder.fit_transform(df_filled)

    # Spearman correlation - copied from notebook 03
    corr_spearman = df_ohe.corr(method='spearman')['SalePrice'].sort_values(
                                        key=abs, ascending=False)[1:].head(10)

    st.write("**Top 10 Spearman Correlations with SalePrice:**")
    st.dataframe(corr_spearman)

    # Pearson correlation - copied from notebook 03
    corr_pearson = df_ohe.corr(method='pearson')['SalePrice'].sort_values(
                                        key=abs, ascending=False)[1:].head(10)

    st.write("**Top 10 Pearson Correlations with SalePrice:**")
    st.dataframe(corr_pearson)

    # Top features - copied from notebook 03
    top_n = 5
    vars_to_study = list(set(corr_pearson[:top_n].index.to_list() +
                             corr_spearman[:top_n].index.to_list()))

    st.write(f"**Top {len(vars_to_study)} Features to Study:**")
    st.write(vars_to_study)

    # EDA on Selected Variables - copied from notebook 03
    df_eda = df_ohe.filter(vars_to_study + ['SalePrice'])

    # Correlation heatmap - copied from notebook 03
    st.write("**Correlation Heatmap - Top Selected Features:**")
    features_for_heatmap = vars_to_study + ['SalePrice']

    plt.figure(figsize=(10, 8))
    correlation_matrix = df[features_for_heatmap].corr()
    sns.heatmap(correlation_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": .8})

    plt.title('Correlation Heatmap - Top Selected House Features',
              fontsize=16, pad=20)
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Correlation bar chart - copied from notebook 03
    st.write("**Correlation with SalePrice - Bar Chart:**")
    correlations_with_price = correlation_matrix['SalePrice'].drop(
                                    'SalePrice').sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(correlations_with_price)),
                   correlations_with_price.values,
                   color=['darkgreen' if x > 0 else 'darkred' for x
                          in correlations_with_price.values])

    plt.title('Correlation with SalePrice - Selected House Features',
              fontsize=16, pad=20)
    plt.xlabel('House Features', fontsize=12)
    plt.ylabel('Correlation Coefficient', fontsize=12)

    plt.xticks(range(len(correlations_with_price)),
               correlations_with_price.index,
               rotation=45,
               ha='right')

    for i, (feature, value) in enumerate(correlations_with_price.items()):
        plt.text(i, value + 0.01 if value > 0 else value - 0.03,
                 f'{value:.3f}',
                 ha='center',
                 va='bottom' if value > 0 else 'top',
                 fontweight='bold',
                 fontsize=10)

    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Parallel plot - copied from notebook 03
    if st.checkbox("Show Parallel Plot"):
        parallel_plot(df_eda)

# Parallel Plot


def parallel_plot(df_eda):
    """Creation of parallel plot - copied from notebook 03"""

    st.write("Dynamic chart, hover over to see the relationship of variables")

    # Discretize SalePrice - copied from notebook 03
    price_map = [-np.inf, 50000, 100000, 150000, 200000, 250000, 300000,
                 350000, np.inf]
    disc = ArbitraryDiscretiser(binning_dict={'SalePrice': price_map})
    df_parallel = disc.fit_transform(df_eda)

    # Create labels - copied from notebook 03
    n_classes = len(price_map) - 1
    classes_ranges = disc.binner_dict_['SalePrice'][1:-1]

    labels_map = {}
    for n in range(0, n_classes):
        if n == 0:
            labels_map[n] = f"<{classes_ranges[0]}"
        elif n == n_classes-1:
            labels_map[n] = f"+{classes_ranges[-1]}"
        else:
            labels_map[n] = f"{classes_ranges[n-1]} to {classes_ranges[n]}"

    df_parallel['SalePrice'] = df_parallel['SalePrice'].replace(labels_map)

    # Create categorization functions - copied from notebook 03
    df_plot = df_parallel.copy()

    # Categorize features - copied from notebook 03
    if '1stFlrSF' in df_plot.columns:
        df_plot['1stFlrSF_cat'] = df_plot['1stFlrSF'].apply(
                                                            categorize_1stflr)
    if 'GarageArea' in df_plot.columns:
        df_plot['GarageArea_cat'] = df_plot['GarageArea'].apply(
                                                            categorize_garage)
    if 'GrLivArea' in df_plot.columns:
        df_plot['GrLivArea_cat'] = df_plot['GrLivArea'].apply(
                                                            categorize_living)
    if 'TotalBsmtSF' in df_plot.columns:
        df_plot['TotalBsmtSF_cat'] = df_plot['TotalBsmtSF'].apply(
                                                          categorize_basement)
    if 'OverallQual' in df_plot.columns:
        df_plot['OverallQual_cat'] = df_plot['OverallQual'].apply(
                                                            categorize_quality)
    if 'YearBuilt' in df_plot.columns:
        df_plot['YearBuilt_cat'] = df_plot['YearBuilt'].apply(categorize_year)

    # Price mapping for coloring - copied from notebook 03
    price_mapping = {
        '<50000': 25000,
        '50000 to 100000': 75000,
        '100000 to 150000': 125000,
        '150000 to 200000': 175000,
        '200000 to 250000': 225000,
        '250000 to 300000': 275000,
        '300000 to 350000': 325000,
        '+350000': 400000
    }

    df_plot['SalePrice_numeric'] = df_plot['SalePrice'].map(price_mapping)

    # Plot order - copied from notebook 03
    categorical_columns = ['OverallQual_cat', 'YearBuilt_cat', 'GrLivArea_cat',
                           '1stFlrSF_cat', 'GarageArea_cat', 'TotalBsmtSF_cat']

    fig = px.parallel_categories(df_plot,
                                 dimensions=categorical_columns,
                                 color='SalePrice_numeric',
                                 color_continuous_scale='viridis',
                                 title="Housing Top Features - "
                                       "Colored by Sale Price",
                                 labels={col: col.replace('_cat', '').replace(
                                                            'SF', ' (sq ft)')
                                         for col in categorical_columns})

    fig.update_layout(
        font_size=10,
        height=700,
        width=1200,
        margin=dict(l=200, r=80, t=100, b=50),
        title_font_size=16,
        coloraxis_colorbar=dict(
            title=dict(text="Sale Price ($)", font=dict(size=14)),
            x=-0.15,
            thickness=40,
            len=0.9,
            tickformat="$,.0f",
            tickfont=dict(size=12)
        )
    )

    st.plotly_chart(fig)


# Categorization functions - copied from notebook 03
def categorize_1stflr(value):
    if value < 700:
        return 'Small (<700 sq ft)'
    elif value < 1000:
        return 'Medium (700-1000)'
    elif value < 1400:
        return 'Large (1000-1400)'
    else:
        return 'Very Large (1400+ sq ft)'


def categorize_garage(value):
    if value == 0:
        return 'No Garage'
    elif value < 400:
        return 'Small (1-400 sq ft)'
    elif value < 600:
        return 'Medium (400-600)'
    elif value < 800:
        return 'Large (600-800)'
    else:
        return 'Very Large (800+ sq ft)'


def categorize_living(value):
    if value < 1000:
        return 'Compact (<1000 sq ft)'
    elif value < 1500:
        return 'Average (1000-1500)'
    elif value < 2000:
        return 'Spacious (1500-2000)'
    elif value < 2500:
        return 'Large (2000-2500)'
    else:
        return 'Very Large (2500+ sq ft)'


def categorize_basement(value):
    if value == 0:
        return 'No Basement'
    elif value < 800:
        return 'Small (1-800 sq ft)'
    elif value < 1200:
        return 'Medium (800-1200)'
    elif value < 1600:
        return 'Large (1200-1600)'
    else:
        return 'Very Large (1600+ sq ft)'


def categorize_quality(value):
    if value <= 3:
        return 'Poor (1-3)'
    elif value <= 5:
        return 'Below Average (4-5)'
    elif value <= 7:
        return 'Average (6-7)'
    elif value <= 8:
        return 'Good (8)'
    else:
        return 'Excellent (9-10)'


def categorize_year(value):
    if value < 1950:
        return 'Historic (Pre-1950)'
    elif value < 1970:
        return 'Mid-Century (1950-1970)'
    elif value < 1990:
        return 'Modern (1970-1990)'
    elif value < 2000:
        return 'Contemporary (1990-2000)'
    else:
        return 'New (2000+)'
