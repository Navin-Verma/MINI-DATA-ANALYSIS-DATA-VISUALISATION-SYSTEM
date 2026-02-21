import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime

# CONFIGURATION
DATA_PATH = r"C:/Users/Robin/Desktop/PROJECT 3/'data/sample_data.csv"
OUTPUT_DIR = "outputs"
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")

os.makedirs(CHART_DIR, exist_ok=True)

# DATA LOADING
def load_data(path: str) -> pd.DataFrame:
    print("Loading dataset...")
    df = pd.read_csv(path)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

# DATA CLEANING
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Cleaning dataset...")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Handle missing numeric values
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Handle missing categorical values
    categorical_cols = df.select_dtypes(include='object').columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    print("Cleaning complete.")
    return df


# SUMMARY STATISTICS
def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    print("Generating summary statistics...")
    summary = df.describe(include='all')
    summary.to_csv(os.path.join(OUTPUT_DIR, "summary_statistics.csv"))
    print("Summary statistics saved.")
    return summary

# VISUALIZATION
def create_visualizations(df: pd.DataFrame):
    print("Creating visualizations...")

    sns.set(style="whitegrid")

    numeric_cols = df.select_dtypes(include=np.number).columns

    for col in numeric_cols:
        plt.figure()
        sns.histplot(df[col], kde=True)
        plt.title(f"Distribution of {col}")
        chart_path = os.path.join(CHART_DIR, f"{col}_distribution.png")
        plt.savefig(chart_path)
        plt.close()

    # Correlation heatmap
    if len(numeric_cols) > 1:
        plt.figure(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        heatmap_path = os.path.join(CHART_DIR, "correlation_heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()

    print("Visualizations saved.")

# PDF REPORT GENERATION
def generate_pdf_report(df: pd.DataFrame):
    print("Generating PDF report...")

    report_path = os.path.join(OUTPUT_DIR, "report.pdf")

    with PdfPages(report_path) as pdf:
        plt.figure(figsize=(10, 6))
        plt.axis("off")
        plt.text(
            0.1, 0.8,
            "Mini Data Analysis Report",
            fontsize=18
        )
        plt.text(
            0.1, 0.7,
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=12
        )
        plt.text(
            0.1, 0.6,
            f"Dataset Shape: {df.shape}",
            fontsize=12
        )
        pdf.savefig()
        plt.close()

        # Add summary statistics page
        summary = df.describe().round(2)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(
            cellText=summary.values,
            colLabels=summary.columns,
            rowLabels=summary.index,
            loc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        pdf.savefig()
        plt.close()

    print("PDF report generated.")

# MAIN EXECUTION PIPELINE
def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)

    # Save cleaned data
    df.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)

    generate_summary(df)
    create_visualizations(df)
    generate_pdf_report(df)

    print("Data Analysis Pipeline Completed Successfully.")


if __name__ == "__main__":
    main()
