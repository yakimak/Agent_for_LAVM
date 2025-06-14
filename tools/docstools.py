from langchain_core.tools import tool
from typing import List, Dict, Any, Optional
import tempfile
from urllib.parse import urlparse
import os
import io
import uuid
import requests
from PIL import Image
import pytesseract
import pandas as pd


@tool
def save_and_read_file(content: str, filename: Optional[str] = None) -> str:
    """
    Save content to a file and return the path.
    Args:
        content (str): the content to save to the file
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    temp_dir = tempfile.gettempdir()
    if filename is None:
        temp_file = tempfile.NamedTemporaryFile(delete=False, dir=temp_dir)
        filepath = temp_file.name
    else:
        filepath = os.path.join(temp_dir, filename)

    with open(filepath, "w") as f:
        f.write(content)

    return f"File saved to {filepath}. You can read this file to process its contents."



@tool
def download_file_from_url(url: str, filename: Optional[str] = None) -> str:
    """
    Download a file from a URL and save it to a temporary location.
    Args:
        url (str): the URL of the file to download.
        filename (str, optional): the name of the file. If not provided, a random name file will be created.
    """
    try:
        # Parse URL to get filename if not provided
        if not filename:
            path = urlparse(url).path
            filename = os.path.basename(path)
            if not filename:
                filename = f"downloaded_{uuid.uuid4().hex[:8]}"

        # Create temporary file
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)

        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Save the file
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        return f"File downloaded to {filepath}. You can read this file to process its contents."
    except Exception as e:
        return f"Error downloading file: {str(e)}"


@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image using OCR library pytesseract (if available).
    Args:
        image_path (str): the path to the image file.
    """
    try:
        # Open the image
        image = Image.open(image_path) 

        # Extract text from the image
        text = pytesseract.image_to_string(image)

        return f"Extracted text from image:\n\n{text}"
    except Exception as e:
        return f"Error extracting text from image: {str(e)}"


@tool
def read_dataframe(file_path: str) -> tuple:
    """
    Reads a CSV or Excel file and returns the dataframe and file type.
    """
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path)
        file_type = "CSV"
    elif any(file_path.lower().endswith(ext) for ext in ['.xls', '.xlsx', '.xlsm']):
        df = pd.read_excel(file_path)
        file_type = "Excel"
    else:
        raise ValueError(f"Unsupported file format. Please provide a CSV or Excel file.")
    
    return df, file_type



@tool
def summary_dataframe(file_path: str, query: str = "summary") -> str:
    """
    Summarize a tabular file (CSV or Excel) using pandas based on specific query.
    
    Args:
        file_path (str): The path to the CSV or Excel file.
        query (str): Analysis query - can be "summary", "info", "shape", "nulls", 
                    "unique", "correlation between col1 and col2", "outliers in col", 
                    "sample [n]" etc.
    """
    try:
        df, file_type = read_dataframe(file_path)
        query = query.lower()
        
        # Basic summary (default)
        if "summary" in query:
            result = f"{file_type} file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
            result += f"Columns: {', '.join(df.columns)}\n\n"
            result += "Summary statistics:\n"
            result += str(df.describe())
            return result
        
        # Sample data rows
        elif "sample" in query:
            rows = 5  # Default sample size
            # Check if a specific number is mentioned
            num_match = re.search(r'sample\s+(\d+)', query)
            if num_match:
                rows = min(int(num_match.group(1)), len(df))
            return f"Sample of {rows} rows:\n\n{df.sample(rows).to_string()}"
        
        # DataFrame info
        elif "info" in query:
            buffer = io.StringIO()
            df.info(buf=buffer)
            return buffer.getvalue()
            
        # Shape of dataframe
        elif "shape" in query or "dimensions" in query:
            return f"DataFrame has {df.shape[0]} rows and {df.shape[1]} columns."
            
        # Check for null values
        elif "null" in query or "missing" in query:
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0]
            
            if len(null_cols) > 0:
                result = "Columns with missing values:\n"
                for col, count in null_cols.items():
                    result += f"- {col}: {count} missing values ({count/len(df):.2%} of data)\n"
                return result
            else:
                return "No missing values found in the dataset."
                
        # Unique values count
        elif "unique" in query:
            for col in df.columns:
                if col.lower() in query:
                    unique_count = df[col].nunique()
                    return f"Column '{col}' has {unique_count} unique values out of {len(df)} rows."
            
            # If no specific column mentioned, show for all columns
            result = "Unique value counts per column:\n"
            for col in df.columns:
                result += f"- {col}: {df[col].nunique()} unique values\n"
            return result
            
        # Correlation between columns
        elif "correlation" in query or "corr" in query:
            # Try to extract column names from query
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # If two specific columns are mentioned
            for col1 in numeric_cols:
                if col1.lower() in query:
                    for col2 in numeric_cols:
                        if col2.lower() in query and col1 != col2:
                            corr = df[col1].corr(df[col2])
                            return f"Correlation between '{col1}' and '{col2}': {corr:.4f}"
            
            # If no specific columns or only one column mentioned, show correlation matrix
            if len(numeric_cols) > 1:
                return f"Correlation matrix:\n{df[numeric_cols].corr().round(2)}"
            else:
                return "Not enough numeric columns for correlation analysis."
                
        # Outlier detection
        elif "outlier" in query:
            numeric_cols = df.select_dtypes(include=['number']).columns
            
            # Check if a specific column is mentioned
            target_col = None
            for col in numeric_cols:
                if col.lower() in query:
                    target_col = col
                    break
            
            # If no specific column mentioned but there are numeric columns
            if target_col is None and len(numeric_cols) > 0:
                target_col = numeric_cols[0]  # Use the first numeric column
                
            if target_col:
                # Using IQR method for outlier detection
                Q1 = df[target_col].quantile(0.25)
                Q3 = df[target_col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
                
                result = f"Outlier analysis for '{target_col}':\n"
                result += f"- IQR (Interquartile Range): {IQR:.4f}\n"
                result += f"- Lower bound: {lower_bound:.4f}\n"
                result += f"- Upper bound: {upper_bound:.4f}\n"
                result += f"- Found {len(outliers)} outliers out of {len(df)} values ({len(outliers)/len(df):.2%})\n"
                
                if len(outliers) > 0 and len(outliers) <= 10:
                    result += f"\nOutlier values:\n{outliers[target_col].tolist()}"
                
                return result
            else:
                return "No numeric columns found for outlier detection."
        
        # Default summary if query doesn't match specific analysis
        else:
            result = f"{file_type} file loaded with {len(df)} rows and {len(df.columns)} columns.\n"
            result += f"Columns: {', '.join(df.columns)}\n\n"
            result += f"First 5 rows of data:\n{df.head().to_string()}\n\n"
            result += "For specific analysis, try queries like:\n"
            result += "- 'summary' for basic statistics\n"
            result += "- 'sample [n]' for n random rows\n"
            result += "- 'info' for detailed dataframe info\n"
            result += "- 'shape' for dimensions\n"
            result += "- 'nulls' for missing value analysis\n"
            result += "- 'unique' for unique value counts\n"
            result += "- 'correlation between col1 and col2' for correlation\n"
            result += "- 'outliers in column_name' for outlier detection"
            return result

    except Exception as e:
        return f"Error analyzing file: {str(e)}"