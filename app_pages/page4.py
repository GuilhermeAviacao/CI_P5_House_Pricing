import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


def page4_body():
    """
    House Price Prediction Page
    
    """

    st.write("### House Price Predictor")

    st.info(
        "**Price Forecasts** Usage of the forecast model to predict the inheritance house values"
    )

    # Section 1: Predict Inherited Houses
    st.write("---")
    st.write("#### Inherited Houses Price Prediction")

    # Load inherited houses data
    inherited_houses = pd.read_csv("Data Input/inherited_houses.csv")

    st.write("**The 4 Inherited Houses:**")
    st.dataframe(inherited_houses)
    st.write("---")
    
    if st.checkbox("Predict Prices for Inherited Houses"):
        predictions = make_predictions(inherited_houses)

        if predictions is not None:
            # Load dataset to get median and mean
            df = pd.read_csv("outputs/datasets/collection/HouseFeaturesPrices.csv")
            median_price = df['SalePrice'].median()
            mean_price = df['SalePrice'].mean()

            # Display price distribution plot
            st.write("#### Price Distribution Analysis")
            plot_price_distribution(predictions)

            # Display reference values
            st.write("**Dataset Reference Values:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Median Price", f"${median_price:,.0f}")
            with col2:
                st.metric("Mean Price", f"${mean_price:,.0f}")

            st.write("---")

            # Create summary table with colors matching the plot
            colors = ['red', 'orange', 'green', 'blue']

            # Display table header
            col1, col2, col3, col4 = st.columns([1.5, 2, 1.5, 1.5])
            with col1:
                st.write("**House**")
            with col2:
                st.write("**Predicted Price**")
            with col3:
                st.write("**% from Median**")
            with col4:
                st.write("**% from Mean**")

            # Display each house row with colors
            for i, price in enumerate(predictions, 1):
                pct_from_median = ((price - median_price) / median_price) * 100
                pct_from_mean = ((price - mean_price) / mean_price) * 100
                color = colors[i-1]

                col1, col2, col3, col4 = st.columns([1.5, 2, 1.5, 1.5])
                with col1:
                    st.markdown(f":{color}[House {i}]")
                with col2:
                    st.markdown(f":{color}[${price:,.0f}]")
                with col3:
                    st.markdown(f":{color}[{pct_from_median:+.1f}%]")
                with col4:
                    st.markdown(f":{color}[{pct_from_mean:+.1f}%]")

            # Total row
            col1, col2, col3, col4 = st.columns([1.5, 2, 1.5, 1.5])
            with col1:
                st.write("**Total Sum**")
            with col2:
                st.write(f"**${predictions.sum():,.0f}**")
            with col3:
                st.write("")
            with col4:
                st.write("")

            # Display total appraisal in green success box
            st.success(f"The four houses appraisal prediction is **${predictions.sum():,.0f}**")

            st.write("---")

def plot_price_distribution(predictions):

    # Load the full dataset to get all prices
    df = pd.read_csv("outputs/datasets/collection/HouseFeaturesPrices.csv")

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot histogram of all prices (same as page2)
    ax.hist(df['SalePrice'], bins=50, alpha=0.6, label='All Houses (Dataset)',
            color='skyblue', edgecolor='black')

    # Add median line from dataset (before mean for legend order)
    median_price = df['SalePrice'].median()
    ax.axvline(median_price, color='gray', linestyle='-', linewidth=2,
               label=f'Dataset Median: ${median_price:,.0f}')

    # Add mean line from dataset
    mean_price = df['SalePrice'].mean()
    ax.axvline(mean_price, color='black', linestyle='-', linewidth=2,
               label=f'Dataset Mean: ${mean_price:,.0f}')

    # Plot inherited house predictions as vertical lines
    colors = ['red', 'orange', 'green', 'blue']
    for i, (price, color) in enumerate(zip(predictions, colors), 1):
        ax.axvline(price, color=color, linestyle='--', linewidth=2,
                   label=f'Inherited House {i}: ${price:,.0f}')

    ax.set_xlabel('Sale Price ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Inherited Houses vs Market Price Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


def make_predictions(input_df):

    # Path to saved model
    model_path = "outputs/ml_pipeline/price_prediction_pipeline.pkl"

    try:
        # Load the model
        with open(model_path, 'rb') as file:
            pipeline = pickle.load(file)

        # Load training data to get the expected column order and values for missing data
        training_df = pd.read_csv("outputs/datasets/collection/HouseFeaturesPrices.csv")
        # Removal of target variable
        training_df = training_df.drop(['SalePrice'], axis=1)  

        # Make a copy of input data
        input_df_clean = input_df.copy()

        # Ensure all training columns are present in input data
        for col in training_df.columns:
            if col not in input_df_clean.columns:
                st.warning(f"Column '{col}' missing from input data. Adding with median/mode value.")
                if training_df[col].dtype == 'object':
                    input_df_clean[col] = training_df[col].mode()[0]
                else:
                    input_df_clean[col] = training_df[col].median()

        # Ensure columns are in the same order as training data
        input_df_clean = input_df_clean[training_df.columns]

        # Handle missing values (same as in notebook)
        # Fill missing categorical values with 'Unknown'
        for col in input_df_clean.select_dtypes(include=['object']).columns:
            input_df_clean[col] = input_df_clean[col].fillna('Unknown')

        # Fill missing numerical values with training data median
        for col in input_df_clean.select_dtypes(include=[np.number]).columns:
            if input_df_clean[col].isnull().any():
                input_df_clean[col] = input_df_clean[col].fillna(training_df[col].median())

        # Make predictions
        predictions = pipeline.predict(input_df_clean)

        return predictions

    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        st.error(f"Details: {type(e).__name__}")
        import traceback
        st.error(traceback.format_exc())
        return None
