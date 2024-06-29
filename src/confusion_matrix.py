import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from importlib_resources import files


def main():
    data = str(files("resources") / "excel_corpora" / "test_predictions.xlsx")
    evaluate_predictions(data)


def evaluate_predictions(excel_path):
    # Load the Excel file
    df = pd.read_excel(excel_path)

    # Extract the true labels and predicted labels
    y_true = df['GS_labels']
    y_pred = df['Predicted_labels']

    # Replace labels 1 with "original" and 0 with "rephrased"
    label_mapping = {1: 'original', 0: 'rephrased'}
    y_true = y_true.map(label_mapping)
    y_pred = y_pred.map(label_mapping)

    # Generate classification report
    report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(report)

    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    # Plot the normalized confusion matrix
    labels = ['original', 'rephrased']
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Normalized Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    main()
