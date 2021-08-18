import numpy as np
import pandas as pd
import argparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory


# Prepare data function
def prepare_data(data):
    # Drop missing values
    X_df = data.to_pandas_dataframe().dropna()

    # Features and target
    y_df = X_df.pop("DEATH_EVENT")

    return X_df, y_df


# Main function
def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help="Inverse of regularization strength. " \
             "Smaller values cause stronger regularization")

    parser.add_argument(
        '--max_iter',
        type=int,
        default=100,
        help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(
        C=args.C, max_iter=args.max_iter).fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    run.log("Accuracy", np.float(accuracy))



# Create TabularDataset using TabularDatasetFactory
data_path = "https://raw.githubusercontent.com/peppegili/" \
            "3_Capstone_Project_ML_Engineer/master/" \
            "data/heart_failure_clinical_records_dataset.csv"
ds = TabularDatasetFactory.from_delimited_files(path=data_path)

# Prepare data
X, y = prepare_data(ds)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

run = Run.get_context()

if __name__ == '__main__':
    main()
