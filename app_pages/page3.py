import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

# ML Pipeline components - copied from notebook 04
from feature_engine.encoding import OrdinalEncoder
from feature_engine.selection import SmartCorrelatedSelection
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesRegressor

sns.set_style('whitegrid')


def page3_body():
    """Page 3 - ML Model: Price Prediction"""

    st.write("## ML Model: Price Prediction")
    st.info(
        "**Machine Learning** pipeline development, "
        "model performance evaluation, and feature importance analysis."
    )

    # Load data
    df = load_house_data()

    # Model Development Process
    st.write("---")
    st.write("### Model Development Process")
    if st.checkbox("Show Model Training Details"):
        show_model_training_process(df)

    # Load the trained model
    st.write("---")
    st.write("### Model Performance Evaluation")
    if st.checkbox("Show Model Performance"):
        show_model_performance(df)

    # Feature Importance
    st.write("---")
    st.write("### Feature Importance Analysis")
    if st.checkbox("Show Feature Importance"):
        show_feature_importance(df)


def load_house_data():
    """House prices dataset - copied from notebook 04"""
    path = r"outputs/datasets/collection/HouseFeaturesPrices.csv"
    csv_file = os.path.join(path)
    df = pd.read_csv(csv_file)
    return df


def handle_missing_values(df):
    """Missing values treatment - copied from notebook 04"""
    df_filled = df.copy()
    object_cols = df.columns[df.dtypes == 'object'].to_list()

    # Fill missing values in object columns
    for col in object_cols:
        df_filled[col] = df_filled[col].fillna('Unknown')

    # For numeric columns, filling with median
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df_filled[col] = df_filled[col].fillna(df_filled[col].median())

    return df_filled


def show_model_training_process(df):
    """Display model training process - copied from notebook 04"""

    st.write("#### Data Preparation")
    st.write(f"**Original Dataset Shape:** {df.shape}")

    # Handle missing values
    df_filled = handle_missing_values(df)
    st.write(f"** After handling missing values: ** {df_filled.
                                                     isnull().sum().sum(
                                                     )} missing values")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df_filled.drop(['SalePrice'], axis=1),
        df_filled['SalePrice'],
        test_size=0.2,
        random_state=0
    )

    st.write(f"**Train set:** {X_train.shape[0]} samples")
    st.write(f"**Test set:** {X_test.shape[0]} samples")

    st.write("#### Model Selection Results")
    st.write(
        "Several regression algorithms were evaluated using Grid Search "
        "CV with default hyperparameters. "
        "The models were compared based on R2 score with 5-fold """
        "cross-validation."
    )

    # Quick search results - copied from notebook 04
    quick_search_results = {
        'estimator': ['ExtraTreesRegressor', 'RandomForestRegressor',
                      'LinearRegression',
                      'GradientBoostingRegressor', 'AdaBoostRegressor',
                      'XGBRegressor',
                      'DecisionTreeRegressor'],
        'mean_score': [0.801154, 0.797278, 0.789251, 0.756799, 0.740974,
                       0.711484, 0.600532]
    }
    df_quick_search = pd.DataFrame(quick_search_results)

    st.write("**Algorithm Comparison (R2 Score):**")
    st.dataframe(df_quick_search)

    st.write(
        "**Best Algorithm:** ExtraTreesRegressor with mean R2 score of 0.801"
    )

    st.write("#### Hyperparameter Optimization")
    st.write(
        "The ExtraTreesRegressor was further optimized by testing different \
            hyperparameter combinations."
    )

    # Hyperparameter search results - copied from notebook 04
    hyperparam_results = {
        'n_estimators': [300, 100, 300, 100, 100, 300],
        'max_depth': [10, 10, None, None, 3, 3],
        'mean_score': [0.807148, 0.806978, 0.804857, 0.801154, 0.728399,
                       0.727763]
    }
    df_hyperparam = pd.DataFrame(hyperparam_results)

    st.write("**Hyperparameter Tuning Results:**")
    st.dataframe(df_hyperparam)

    st.write(
        "**Best Hyperparameters:** n_estimators=300, max_depth=10, \
            criterion='squared_error'"
    )


def show_model_performance(df):
    """Show model performance metrics and plots - copied from notebook 04"""

    # Load the trained model
    model_path = "outputs/ml_pipeline/price_prediction_pipeline.pkl"

    with open(model_path, 'rb') as file:
        best_regressor_pipeline = pickle.load(file)

    # Prepare data
    df_filled = handle_missing_values(df)
    X_train, X_test, y_train, y_test = train_test_split(
        df_filled.drop(['SalePrice'], axis=1),
        df_filled['SalePrice'],
        test_size=0.2,
        random_state=0
    )

    # Model evaluation - copied from notebook 04
    st.write("#### Model Performance Metrics")

    st.write(
        "Both Train and Test R2 scores are above the 0.80 score target"
        "proving the predictive quality of the model")
    # Train set evaluation
    st.write("**Train Set Performance:**")
    pred_train = best_regressor_pipeline.predict(X_train)
    st.write(f"R2 Score: \
    {r2_score(y_train, pred_train):.3f}")
    st.write(f"Mean Absolute Error: \
    {mean_absolute_error(y_train,pred_train):,.3f}")
    st.write(f"Mean Squared Error:\
    {mean_squared_error(y_train, pred_train):,.3f} ")
    st.write(f"Root Mean Squared Error:\
    {np.sqrt(mean_squared_error(y_train, pred_train)):,.3f}")

    # Test set evaluation
    st.write("**Test Set Performance:**")
    pred_test = best_regressor_pipeline.predict(X_test)
    st.write(f"R2 Score: \
    {r2_score(y_test, pred_test):.3f}")
    st.write(f"Mean Absolute Error: \
    {mean_absolute_error(y_test, pred_test):,.3f}")
    st.write(f"Mean Squared Error: \
    {mean_squared_error(y_test, pred_test):,.3f}")
    st.write(f"Root Mean Squared Error: \
    {np.sqrt(mean_squared_error(y_test, pred_test)):,.3f}")

    # Performance plots - copied from notebook 04
    st.write("#### Actual vs Predicted Plots")
    regression_evaluation_plots(X_train, y_train, X_test, y_test,
                                best_regressor_pipeline)


def regression_evaluation_plots(X_train, y_train, X_test, y_test, pipeline,
                                alpha_scatter=0.5):
    """Create actual vs predicted plots - copied from notebook 04"""
    pred_train = pipeline.predict(X_train)
    pred_test = pipeline.predict(X_test)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # Train set plot
    sns.scatterplot(x=y_train, y=pred_train, alpha=alpha_scatter, ax=axes[0])
    sns.lineplot(x=y_train, y=y_train, color='red', ax=axes[0])
    axes[0].set_xlabel("Actual")
    axes[0].set_ylabel("Predictions")
    axes[0].set_title("Train Set")

    # Test set plot
    sns.scatterplot(x=y_test, y=pred_test, alpha=alpha_scatter, ax=axes[1])
    sns.lineplot(x=y_test, y=y_test, color='red', ax=axes[1])
    axes[1].set_xlabel("Actual")
    axes[1].set_ylabel("Predictions")
    axes[1].set_title("Test Set")

    plt.tight_layout()
    st.pyplot(fig)


def show_feature_importance(df):
    """Show feature importance analysis - copied from notebook 04"""

    # Load the trained model
    model_path = "outputs/ml_pipeline/price_prediction_pipeline.pkl"

    with open(model_path, 'rb') as file:
        best_regressor_pipeline = pickle.load(file)

    # Prepare data
    df_filled = handle_missing_values(df)
    X_train, X_test, y_train, y_test = train_test_split(
        df_filled.drop(['SalePrice'], axis=1),
        df_filled['SalePrice'],
        test_size=0.2,
        random_state=0
    )

    # Feature importance - copied from notebook 04
    data_cleaning_feat_eng_steps = 2

    columns_after_data_cleaning_feat_eng = (Pipeline(best_regressor_pipeline.
    steps[:data_cleaning_feat_eng_steps])
    .transform(X_train)
    .columns)

    best_features = columns_after_data_cleaning_feat_eng[
            best_regressor_pipeline['feat_selection'].get_support()].to_list()

    df_feature_importance = (pd.DataFrame(data={
            'Feature': columns_after_data_cleaning_feat_eng[
                best_regressor_pipeline['feat_selection'].get_support()],
            'Importance': best_regressor_pipeline['model'].
            feature_importances_})
        .sort_values(by='Importance', ascending=False)
    )

    st.write(f"**These are the {len(best_features)} most important \
                                            features in descending order.**")
    st.dataframe(df_feature_importance)

    # Feature importance plot - copied from notebook 04
    st.write("#### Feature Importance Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    df_feature_importance.plot(kind='bar', x='Feature', y='Importance',
                               ax=ax, legend=False)
    ax.set_xlabel("Feature")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance - ExtraTreesRegressor")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
