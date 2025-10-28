import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


def page5_body():

    st.write("### Predict Price for any House in Ames, Iowa")

    st.info(
        "**Custom Price Prediction:** Our model finds that the five most "
        "important features are sufficient to predict the house sale price:"
    )

    st.write("Use the controllers below to input the house attributes:")

    # Get user inputs for the 5 most important features
    house_data = get_house_inputs()

    # Predict house price
    if st.checkbox("Predict Sale Price"):
        # Convert to dataframe
        input_df = pd.DataFrame([house_data])

        # Make prediction
        predictions = make_predictions(input_df)

        if predictions is not None:
            predicted_price = predictions[0]
            st.success(f"### Predicted Sale Price: ${predicted_price:,.0f}")

            # Show price distribution plot for the forecasted house
            st.write("---")
            st.write("#### Forecast comparison with the whole market")
            plot_single_house_distribution(predicted_price)


def get_house_inputs():
    """
    Input widget with the 5 most important price features
    """

    # Load the full dataset to get ranges
    df = pd.read_csv("outputs/datasets/collection/HouseFeaturesPrices.csv")

    house_data = {}

    st.write("**Most Important Features:**")

    # Feature 1: OverallQual
    house_data['OverallQual'] = st.slider(
        "Overall Quality (1=Very Poor, 10=Very Excellent)",
        min_value=1,
        max_value=10,
        value=5,
    )

    # Feature 2: GrLivArea
    house_data['GrLivArea'] = st.number_input(
        "Above Ground Living Area (Square Feet)",
        min_value=int(df['GrLivArea'].min()),
        max_value=int(df['GrLivArea'].max()),
        value=int(df['GrLivArea'].median()),
    )

    # Feature 3: YearBuilt
    house_data['YearBuilt'] = st.number_input(
        "Year Built",
        min_value=int(df['YearBuilt'].min()),
        max_value=int(df['YearBuilt'].max()),
        value=int(df['YearBuilt'].median()),
    )

    # Feature 4: TotalBsmtSF
    house_data['TotalBsmtSF'] = st.number_input(
        "Total Basement Area (Square Feet)",
        min_value=int(df['TotalBsmtSF'].min()),
        max_value=int(df['TotalBsmtSF'].max()),
        value=int(df['TotalBsmtSF'].median()),
    )

    # Feature 5: GarageArea
    house_data['GarageArea'] = st.number_input(
        "Garage Area (Square Feet)",
        min_value=int(df['GarageArea'].min()),
        max_value=int(df['GarageArea'].max()),
        value=int(df['GarageArea'].median()),
    )

    # Add remaining features with median values
    # Needed for the model but not shown to user
    remaining_features = [
        '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr', 'BsmtFinSF1', 'BsmtUnfSF',
        'EnclosedPorch', 'GarageYrBlt', 'LotArea', 'LotFrontage',
        'MasVnrArea', 'OpenPorchSF', 'OverallCond', 'WoodDeckSF',
        'YearRemodAdd'
    ]

    for feature in remaining_features:
        if feature in df.columns:
            house_data[feature] = df[feature].median()

    # Add categorical features with most common values
    categorical_features = ['BsmtExposure', 'BsmtFinType1', 'GarageFinish',
                            'KitchenQual']

    for feature in categorical_features:
        if feature in df.columns:
            house_data[feature] = df[feature].mode()[0]

    return house_data


def plot_single_house_distribution(predicted_price):

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

    # Plot predicted house price as vertical line
    ax.axvline(predicted_price, color='red', linestyle='--', linewidth=3,
               label=f'Predicted House: ${predicted_price:,.0f}')

    ax.set_xlabel('Sale Price ($)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Predicted House vs Market Price Distribution', fontsize=14,
                 fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)

    # Show comparison statistics
    st.write("**Market Comparison:**")
    col1, col2, col3 = st.columns(3)

    mean_price = df['SalePrice'].mean()
    median_price = df['SalePrice'].median()

    with col1:
        st.metric("Dataset Mean", f"${mean_price:,.0f}")
    with col2:
        st.metric("Dataset Median", f"${median_price:,.0f}")
    with col3:
        diff_from_mean = ((predicted_price - mean_price) / mean_price) * 100
        st.metric("vs Market Mean", f"{diff_from_mean:+.1f}%")

    # Show percentile
    percentile = (df['SalePrice'] < predicted_price).mean() * 100
    st.info(f"This house is priced higher than **{percentile:.1f}%** "
            "of houses in the dataset.")


def make_predictions(input_df):

    # Path to saved model
    model_path = "outputs/ml_pipeline/price_prediction_pipeline.pkl"

    try:
        # Load the model
        with open(model_path, 'rb') as file:
            pipeline = pickle.load(file)

        # Load training to get the expected col order and vals for missing data
        training_df = pd.read_csv(
            "outputs/datasets/collection/HouseFeaturesPrices.csv")
        training_df = training_df.drop(['SalePrice'], axis=1)

        # Make a copy of input data
        input_df_clean = input_df.copy()

        # Ensure all training columns are present in input data
        for col in training_df.columns:
            if col not in input_df_clean.columns:
                st.warning(f"Column '{col}' missing from input data."
                           "Adding with median/mode value.")
                if training_df[col].dtype == 'object':
                    input_df_clean[col] = training_df[col].mode()[0]
                else:
                    input_df_clean[col] = training_df[col].median()

        # Ensure columns are in the SAME ORDER as training data
        input_df_clean = input_df_clean[training_df.columns]

        # Handle missing values (same as in notebook)
        # Fill missing categorical values with 'Unknown'
        for col in input_df_clean.select_dtypes(include=['object']).columns:
            input_df_clean[col] = input_df_clean[col].fillna('Unknown')

        # Fill missing numerical values with training data median
        for col in input_df_clean.select_dtypes(include=[np.number]).columns:
            if input_df_clean[col].isnull().any():
                input_df_clean[col] = input_df_clean[col].fillna(
                                                    training_df[col].median())

        # Make predictions
        predictions = pipeline.predict(input_df_clean)

        return predictions

    except Exception as e:
        st.error(f"Error making predictions: {str(e)}")
        st.error(f"Details: {type(e).__name__}")
        import traceback
        st.error(traceback.format_exc())
        return None