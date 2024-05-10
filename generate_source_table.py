import os
import pandas as pd

def csvs_to_excel(folder_path, output_excel_file):
    """
    Convert all CSV files in a folder to an Excel file with each CSV as a separate sheet.

    Parameters:
    - folder_path: str, the path to the folder containing CSV files.
    - output_excel_file: str, the path for the output Excel file.
    """
    writer = pd.ExcelWriter(output_excel_file, engine='openpyxl')

    for file in sorted(os.listdir(folder_path)):
        if file.endswith('.csv'):
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            sheet_name = os.path.splitext(file)[0]  # Remove .csv extension
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.save()

# Example usage
folder_path = os.path.join('figures', 'source')
output_excel_file = 'source_data.xlsx'  # Desired output Excel file path
csvs_to_excel(folder_path, output_excel_file)
