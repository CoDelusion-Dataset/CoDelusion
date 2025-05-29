#!/usr/bin/env python3
"""
Command-line tool for evaluating hallucination detection results.
"""

import argparse
import sys
import re
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def extract_number(text: Any) -> Optional[float]:
    """
    Extract a numeric value from text.
    
    Args:
        text: The text to extract number from
        
    Returns:
        The extracted number as float, or None if no number found
    """
    if pd.isna(text) or text == "":
        return None
        
    # Use regular expression to match numeric values
    matches = re.findall(r"\d+\.?\d*", str(text))
    return float(matches[0]) if matches else None


def process_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process raw dataframe to prepare for evaluation.
    
    Args:
        df: Input DataFrame with raw detection results
        
    Returns:
        Tuple of (valid_data, full_results_df)
    """
    # Extract numerical values from labels
    df["label_clean"] = df["label"].apply(extract_number)
    df["check_clean"] = df["check"].apply(extract_number)
    
    # Filter valid samples
    valid_data = df.dropna(subset=["check_clean"]).copy()
    
    # Add correctness column
    valid_data["is_correct"] = valid_data["label_clean"] == valid_data["check_clean"]
    
    return valid_data, df


def calculate_metrics(valid_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate evaluation metrics from valid data.
    
    Args:
        valid_data: DataFrame with processed valid samples
        
    Returns:
        Dictionary containing evaluation metrics
    """
    y_true = valid_data["check_clean"].astype(str)
    y_pred = valid_data["label_clean"].astype(str)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Get full classification report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Compile metrics
    metrics = {
        "accuracy": accuracy,
        "report": report,
        "macro_precision": report['macro avg']['precision'],
        "macro_recall": report['macro avg']['recall'],
        "macro_f1": report['macro avg']['f1-score'],
    }
    
    return metrics


def generate_visualizations(results_df: pd.DataFrame, output_prefix: str) -> None:
    """
    Generate and save visualizations of evaluation results.
    
    Args:
        results_df: DataFrame with evaluation results per class
        output_prefix: Prefix for saving output files
    """
    # Filter out average rows, only keep specific error types
    class_results = results_df[~results_df["error_type"].isin(["accuracy", "macro avg", "weighted avg"])]
    
    # 1. Bar charts comparing performance metrics
    plt.figure(figsize=(14, 8))
    class_results_melted = pd.melt(
        class_results, 
        id_vars=["error_type"], 
        value_vars=["precision", "recall", "f1-score"],
        var_name="metric", 
        value_name="value"
    )

    sns.barplot(x="error_type", y="value", hue="metric", data=class_results_melted)
    plt.title("Code Defect Detection Performance by Type", fontsize=16)
    plt.xlabel("Error Type", fontsize=14)
    plt.ylabel("Metric Value", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title="Evaluation Metrics")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_error_type_performance.png", dpi=300)

    # 2. Heatmap showing overall performance
    plt.figure(figsize=(10, 8))
    heatmap_data = class_results.set_index("error_type")[["precision", "recall", "f1-score"]]
    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=0.5)
    plt.title("Code Defect Detection Performance Heatmap", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_performance_heatmap.png", dpi=300)

    # 3. Bar chart for support (sample count distribution)
    plt.figure(figsize=(12, 6))
    sns.barplot(x="error_type", y="support", data=class_results)
    plt.title("Error Type Sample Count Distribution", fontsize=16)
    plt.xlabel("Error Type", fontsize=14)
    plt.ylabel("Sample Count", fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_error_type_distribution.png", dpi=300)

    logger.info(f"Visualizations saved with prefix: {output_prefix}")


def evaluate_results(input_file: str, output_prefix: Optional[str] = None, 
                     generate_graphs: bool = False) -> Dict[str, Any]:
    """
    Evaluate code hallucination detection results.
    
    Args:
        input_file: Path to the CSV file with detection results
        output_prefix: Prefix for output files (default: input file name without extension)
        generate_graphs: Whether to generate visualizations (default: False)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Determine output prefix
    if output_prefix is None:
        output_prefix = Path(input_file).stem
    
    # Read CSV file
    try:
        df = pd.read_csv(input_file, encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to read input file: {str(e)}")
        return {}
    
    # Process data
    valid_data, full_df = process_dataframe(df)
    
    # Print label distribution
    label_dist = valid_data["label_clean"].value_counts().to_dict()
    logger.info(f"Label distribution: {label_dist}")
    
    # Calculate metrics
    metrics = calculate_metrics(valid_data)
    
    # Output results
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Valid samples: {len(valid_data)}")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info("Classification report (by category):")
    logger.info(classification_report(
        valid_data["check_clean"].astype(str), 
        valid_data["label_clean"].astype(str)
    ))
    logger.info("Macro average metrics:")
    logger.info(f"Macro Precision: {metrics['macro_precision']:.4f}")
    logger.info(f"Macro Recall: {metrics['macro_recall']:.4f}")
    logger.info(f"Macro F1: {metrics['macro_f1']:.4f}")
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(metrics["report"]).T
    results_df = results_df.reset_index().rename(columns={"index": "error_type"})
    
    # Save results to CSV
    results_csv_path = f"{output_prefix}_evaluation_results.csv"
    results_df.to_csv(results_csv_path, index=False)
    logger.info(f"Results saved to {results_csv_path}")
    
    # Generate visualizations if requested
    if generate_graphs:
        generate_visualizations(results_df, output_prefix)
    
    return metrics


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Evaluate hallucination detection results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_file", 
        help="Path to CSV file containing detection results"
    )
    
    parser.add_argument(
        "--prefix", "-p",
        help="Output file prefix",
        default=None
    )
    
    parser.add_argument(
        "--graphs", "-g",
        action="store_true",
        help="Generate performance graphs"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    try:
        evaluate_results(
            args.input_file,
            args.prefix,
            args.graphs
        )
        return 0
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main()) 