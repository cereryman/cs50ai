import csv
import sys
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Additional imports
import datetime

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Read data in from file
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)

        data = []
        for row in reader:
            data.append({
                "evidence": [                                                           # In | Label
                    int(row[0]),                                                        # 0  | Administrative
                    float(row[1]),                                                      # 1  | Administrative_Duration
                    int(row[2]),                                                        # 2  | Informational
                    float(row[3]),                                                      # 3  | Informational_Duration
                    int(row[4]),                                                        # 4  | ProductRelated
                    float(row[5]),                                                      # 5  | ProductRelated_Duration
                    float(row[6]),                                                      # 6  | BounceRates
                    float(row[7]),                                                      # 7  | ExitRates
                    float(row[8]),                                                      # 8  | PageValues
                    float(row[9]),                                                      # 9  | SpecialDay
                    # Credit for this clean approach to month conversion:
                    # https://www.kite.com/python/answers/how-to-convert-between-month-name-and-month-number-in-python
                    int(datetime.datetime.strptime(str(row[10])[:3], "%b").month) - 1,  # 10 | Month
                    int(row[11]),                                                       # 11 | OperatingSystems
                    int(row[12]),                                                       # 12 | Browser
                    int(row[13]),                                                       # 13 | Region
                    int(row[14]),                                                       # 14 | TrafficType
                    1 if row[15] == "Returning_Visitor" else 0,                         # 15 | VisitorType
                    1 if row[16] == "TRUE" else 0                                       # 16 | Weekend
                ],
                "label": 1 if row[17] == "TRUE" else 0
            })

    evidence = [row["evidence"] for row in data]
    label = [row["label"] for row in data]
    return evidence, label


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """

    # X_training = [row["evidence"] for row in training]
    # y_training = [row["label"] for row in training]
    # model.fit(X_training, y_training)
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """

    actual_positives = 0
    predicted_positives = 0
    actual_negatives = 0
    predicted_negatives = 0

    for actual, predicted in zip(labels, predictions):
        if actual == 1:
            actual_positives += 1
            if predicted == 1:
                predicted_positives += 1
        else:
            actual_negatives += 1
            if predicted == 0:
                predicted_negatives += 1

    # Calculate Sensitivity
    # Sensitivity refers to the proportion of positive examples that were correctly identified: in other words,
    # the proportion of users who did go through with a purchase who were correctly identified.
    sensitivity = predicted_positives/actual_positives
    # Calculated Specificity
    # Specificity refers to the proportion of negative examples that were correctly identified: in this case,
    # the proportion of users who did not go through with a purchase who were correctly identified.
    specificity = predicted_negatives/actual_negatives

    return sensitivity, specificity


if __name__ == "__main__":
    main()
