import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(best_models, X_test, y_test, feature_names):
    for model_name, best_model in best_models.items():
        print(f"Evaluating {model_name}")

        if hasattr(best_model, 'feature_importances_'):
            feature_importances = best_model.feature_importances_
            important_features_indices = np.argsort(feature_importances)[::-1]
            important_features_names = [feature_names[i] for i in important_features_indices]

            print("Important Features:")
            for feature_name in important_features_names:
                print(feature_name)
        else:
            print("Model does not support feature_importances_")

        if not hasattr(best_model, 'predict'):
            best_model.fit(X_test, y_test)

        y_pred = best_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R-squared:", r2)
        print("\n")

        # Save the plot to the PDF file
        plt.figure()
        plt.xlabel('Regression Models', fontsize=12)
        plt.ylabel('R-squared', fontsize=12)
        plt.title(f'R-squared Scores for {model_name}', fontsize=15)
        plt.xticks(fontsize=10, rotation=45, ha="right")
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.close()

    # Show the PDF file after the loop
    plt.show()
