from data_loader import load_data
from data_processing import feature_engineering, feature_engineering_f4
from model_builder import build_model
from model_evaluation import evaluate_model
import os


def process_dataframe(sheet_name, feature_engineering_func):
    
    script_directory = os.path.dirname(os.path.realpath(__file__))
    
    file_path = os.path.join(script_directory, "data_loader", "WaitData.Published.xlsx")
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
