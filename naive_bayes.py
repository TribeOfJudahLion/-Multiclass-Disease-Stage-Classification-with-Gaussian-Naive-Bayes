import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns

def load_data(file_path):
    """ Load data from a file."""
    data = np.loadtxt(file_path, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    return X, y

def plot_classifier(classifier, X, y):
    """ Plot the decision boundaries of the classifier."""
    # Define ranges to plot the figure
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
    step_size = 0.01

    # Define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # Compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray, shading='auto')

    # Overlay the training points on the plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())
    plt.xticks(np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0))
    plt.yticks(np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0))
    plt.title("Decision boundaries and training samples")
    plt.show()

def perform_cross_validation(classifier, X, y, num_validations=5):
    """ Perform cross-validation and print the evaluation metrics."""
    metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    for metric in metrics:
        scores = model_selection.cross_val_score(classifier, X, y, scoring=metric, cv=num_validations)
        print(f"{metric.capitalize()}: {round(100*scores.mean(), 2)}%")

def main():
    input_file = 'data_multivar.txt'
    X, y = load_data(input_file)

    classifier_gaussiannb = GaussianNB()
    classifier_gaussiannb.fit(X, y)

    plot_classifier(classifier_gaussiannb, X, y)
    perform_cross_validation(classifier_gaussiannb, X, y)

    # Additional: Confusion Matrix and Classification Report
    y_pred = classifier_gaussiannb.predict(X)
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:\n", cm)
    sns.heatmap(pd.DataFrame(cm), annot=True, cmap="YlGnBu", fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.show()

    print("\nClassification Report:\n", classification_report(y, y_pred))

if __name__ == "__main__":
    main()
