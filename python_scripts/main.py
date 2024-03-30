from data_loader.load_data import load_data
from data_processing.feature_engineering import feature_engineering
from data_processing.feature_engineering_f4 import feature_engineering_f4
from model_builder.build_model import build_model
from model_evaluation.evaluate_model import evaluate_model
import os


def process_dataframe(sheet_name, feature_engineering_func):
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    # Construct the full path to the Excel file within the data_loader folder
    file_path = os.path.join(script_directory, "data_loader", "name.xlsx")
    df = load_data(file_path, sheet_name=sheet_name)
    df = feature_engineering_func(df)
    best_models, X_train, X_test, y_train, y_test, feature_names = build_model(df)
    evaluate_model(best_models, X_test, y_test, feature_names)

def main():
        process_dataframe("F1", feature_engineering)
        process_dataframe("F2", feature_engineering)
        process_dataframe("F3", feature_engineering)
        process_dataframe("F4", feature_engineering_f4)

if __name__ == "__main__":
    main()
