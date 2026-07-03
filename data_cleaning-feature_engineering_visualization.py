"""
data_cleaning.py
Reusable data cleaning and preprocessing functions for the DataCo Supply Chain dataset.
"""


import pandas as pd
import numpy as np


def load_raw_data(filepath: str) -> pd.DataFrame:
    """Load the raw DataCo dataset from CSV."""
    df = pd.read_csv(filepath, encoding='latin-1')
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names: lowercase, replace spaces with underscores, remove special chars."""
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(' ', '_', regex=False)
        .str.replace('(', '', regex=False)
        .str.replace(')', '', regex=False)
        .str.replace('.', '_', regex=False)
    )
    return df


def drop_redundant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that are irrelevant or contain PII/placeholder data."""
    cols_to_drop = [
        'customer_email', 'customer_password', 'product_image',
        'customer_fname', 'customer_lname'
    ]
    existing_drops = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_drops)
    print(f"Dropped {len(existing_drops)} redundant columns: {existing_drops}")
    return df


def convert_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert date string columns to datetime objects."""
    date_cols = ['order_date_dateorders', 'shipping_date_dateorders']
    df = df.copy()
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            print(f"Converted '{col}' to datetime")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Report and handle missing values."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({
        'missing_count': missing,
        'missing_pct': missing_pct
    }).query('missing_count > 0').sort_values('missing_pct', ascending=False)

    if len(missing_report) > 0:
        print(f"\nColumns with missing values:\n{missing_report}")
    else:
        print("No missing values found.")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows."""
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed = initial_rows - len(df)
    print(f"Removed {removed:,} duplicate rows ({removed/initial_rows*100:.2f}%)")
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add useful calculated columns for analysis."""
    df = df.copy()

    # Shipping delay (actual - scheduled)
    if 'days_for_shipping_real' in df.columns and 'days_for_shipment_scheduled' in df.columns:
        df['shipping_delay_days'] = df['days_for_shipping_real'] - df['days_for_shipment_scheduled']

    # Order month and year for time series
    if 'order_date_dateorders' in df.columns:
        df['order_month'] = df['order_date_dateorders'].dt.to_period('M')
        df['order_year'] = df['order_date_dateorders'].dt.year
        df['order_day_of_week'] = df['order_date_dateorders'].dt.day_name()

    # Profit margin percentage
    if 'order_profit_per_order' in df.columns and 'sales' in df.columns:
        df['profit_margin_pct'] = np.where(
            df['sales'] != 0,
            (df['order_profit_per_order'] / df['sales'] * 100).round(2),
            0
        )

    print("Added derived features: shipping_delay_days, order_month, order_year, order_day_of_week, profit_margin_pct")
    return df


def run_full_cleaning_pipeline(filepath: str) -> pd.DataFrame:
    """Execute the complete cleaning pipeline and return a clean DataFrame."""
    print("=" * 60)
    print("RUNNING DATA CLEANING PIPELINE")
    print("=" * 60)

    df = load_raw_data(filepath)
    df = standardize_columns(df)
    df = drop_redundant_columns(df)
    df = convert_date_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = add_derived_features(df)

    print("=" * 60)
    print(f"PIPELINE COMPLETE — Final shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print("=" * 60)
    return df

#---------------------------------------------------------------------------------------------------------------------------------



"""
feature_engineering.py
Feature engineering functions for predictive modeling (Deliverable 3).
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def encode_categorical_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Label encode specified categorical columns."""
    df = df.copy()
    encoders = {}
    for col in columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders


def create_delivery_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features specifically for late delivery prediction model."""
    df = df.copy()

    # Order value buckets
    df['order_value_bucket'] = pd.qcut(df['sales'], q=4, labels=['Low', 'Medium', 'High', 'Premium'])

    # Is high discount
    if 'order_item_discount' in df.columns:
        df['is_high_discount'] = (df['order_item_discount'] > df['order_item_discount'].median()).astype(int)

    # Items per order indicator
    if 'order_item_quantity' in df.columns:
        df['is_bulk_order'] = (df['order_item_quantity'] > 3).astype(int)

    # Day of week encoded
    if 'order_day_of_week' in df.columns:
        df['is_weekend_order'] = df['order_day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

    return df


def create_fraud_detection_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features specifically for fraud detection model."""
    df = df.copy()

    # Unusually high profit margin flag
    if 'profit_margin_pct' in df.columns:
        q95 = df['profit_margin_pct'].quantile(0.95)
        df['extreme_margin'] = (df['profit_margin_pct'] > q95).astype(int)

    # Order-to-ship speed
    if 'days_for_shipping_real' in df.columns:
        df['fast_ship'] = (df['days_for_shipping_real'] <= 1).astype(int)

    # Negative profit flag
    if 'order_profit_per_order' in df.columns:
        df['negative_profit'] = (df['order_profit_per_order'] < 0).astype(int)

    return df


def prepare_model_data(df: pd.DataFrame, target_col: str, feature_cols: list,
                       scale: bool = True) -> tuple:
    """Prepare X and y arrays for model training."""
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle any remaining NaN
    X = X.fillna(X.median(numeric_only=True))

    if scale:
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        return X_scaled, y, scaler

    return X, y, None

#----------------------------------------------------------------------------------

"""
visualization.py
Reusable visualization functions for supply chain analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11


def plot_delivery_status_distribution(df: pd.DataFrame, save_path: str = None):
    """Bar chart of delivery status counts."""
    fig, ax = plt.subplots(figsize=(10, 6))
    counts = df['delivery_status'].value_counts()
    colors = sns.color_palette("Blues_r", len(counts))
    counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title('Order Distribution by Delivery Status', fontsize=14, fontweight='bold')
    ax.set_xlabel('Delivery Status')
    ax.set_ylabel('Number of Orders')
    ax.tick_params(axis='x', rotation=45)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 500, f'{v:,}', ha='center', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_late_delivery_by_shipping_mode(df: pd.DataFrame, save_path: str = None):
    """Late delivery rate by shipping mode."""
    fig, ax = plt.subplots(figsize=(10, 6))
    late_rate = df.groupby('shipping_mode')['late_delivery_risk'].mean().sort_values(ascending=False) * 100
    late_rate.plot(kind='barh', ax=ax, color=sns.color_palette("Reds_r", len(late_rate)), edgecolor='black')
    ax.set_title('Late Delivery Risk by Shipping Mode', fontsize=14, fontweight='bold')
    ax.set_xlabel('Late Delivery Rate (%)')
    for i, v in enumerate(late_rate.values):
        ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_monthly_order_trend(df: pd.DataFrame, save_path: str = None):
    """Line chart of monthly order volume over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    monthly = df.groupby('order_month').size()
    monthly.index = monthly.index.to_timestamp()
    ax.plot(monthly.index, monthly.values, marker='o', linewidth=2, markersize=4, color='#2196F3')
    ax.fill_between(monthly.index, monthly.values, alpha=0.1, color='#2196F3')
    ax.set_title('Monthly Order Volume Over Time', fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Number of Orders')
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_profit_by_category(df: pd.DataFrame, save_path: str = None):
    """Box plot of profit per order by product category."""
    fig, ax = plt.subplots(figsize=(14, 7))
    top_categories = df['category_name'].value_counts().head(10).index
    subset = df[df['category_name'].isin(top_categories)]
    sns.boxplot(data=subset, x='category_name', y='order_profit_per_order', ax=ax, palette='viridis')
    ax.set_title('Profit Distribution by Top 10 Product Categories', fontsize=14, fontweight='bold')
    ax.set_xlabel('Category')
    ax.set_ylabel('Profit per Order ($)')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_sales_by_region(df: pd.DataFrame, save_path: str = None):
    """Horizontal bar chart of total sales by market region."""
    fig, ax = plt.subplots(figsize=(10, 6))
    region_sales = df.groupby('market')['sales'].sum().sort_values()
    region_sales.plot(kind='barh', ax=ax, color=sns.color_palette("coolwarm", len(region_sales)), edgecolor='black')
    ax.set_title('Total Sales Revenue by Market Region', fontsize=14, fontweight='bold')
    ax.set_xlabel('Total Sales ($)')
    for i, v in enumerate(region_sales.values):
        ax.text(v + 1000, i, f'${v:,.0f}', va='center', fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """Heatmap of correlations between numeric columns."""
    fig, ax = plt.subplots(figsize=(14, 10))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                square=True, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Heatmap — Numeric Features', fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_shipping_delay_distribution(df: pd.DataFrame, save_path: str = None):
    """Histogram of shipping delay in days."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['shipping_delay_days'], bins=30, color='#FF7043', edgecolor='black', alpha=0.8)
    ax.axvline(x=0, color='green', linestyle='--', linewidth=2, label='On Time')
    ax.set_title('Distribution of Shipping Delay (Days)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Delay (days) — Negative = Early, Positive = Late')
    ax.set_ylabel('Frequency')
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
