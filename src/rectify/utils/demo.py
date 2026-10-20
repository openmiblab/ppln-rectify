import pandas as pd
import os


def append_string_to_file(text: str, file_path: str):
    """Appends a string to the end of a text file on a new line."""
    with open(file_path, 'a') as f:
        f.write(f"\n{text}")


def append_string_to_csv(text: str, file_path: str, column_name: str = "entry"):
    """Appends a string as a new row in a CSV file using pandas."""
    df = pd.DataFrame({column_name: [text]})
    
    # Check if file exists to determine if we need to write the header
    file_exists = os.path.isfile(file_path)
    
    df.to_csv(file_path, mode='a', index=False, header=not file_exists)



