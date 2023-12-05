<br/>
<p align="center">
  <h3 align="center">Advance Diagnostic Accuracy with AI: Multiclass Disease Stage Classification</h3>

  <p align="center">
    Unlock the power of precision medicine — Classify with confidence using Gaussian Naive Bayes!
    <br/>
    <br/>
  </p>
</p>



## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Getting Started](#getting-started)
* [Contributing](#contributing)
* [License](#license)
* [Authors](#authors)
* [Acknowledgements](#acknowledgements)

## About The Project

### Problem Scenario:

A medical research institute is trying to improve its diagnostic process for a particular disease that presents itself in four distinct stages. Each stage of the disease requires a different treatment plan, so accurate stage classification is critical. The institute has gathered a comprehensive dataset of patient cases, each with various physiological measurements that correlate with the disease stages. The data has been anonymized and labeled with the correct stage of the disease for each case, ready for analysis.

The institute needs a machine learning model that can learn from this dataset and classify new cases into the correct disease stage with high accuracy. It is crucial for the model to be interpretable to some extent, as the clinicians need to understand the basis for its predictions. Additionally, they want to evaluate the model's performance thoroughly before deploying it in a clinical setting.

### Solution:

1. **Data Preparation:**
   - Use the `load_data` function to load the physiological measurement data from the `data_multivar.txt` file. The last column in this file is assumed to be the disease stage labels (0 to 3), and the preceding columns are the features (physiological measurements).

2. **Model Selection:**
   - Choose the Gaussian Naive Bayes classifier, which is suitable for classification when features are continuous and normally distributed. It also has the advantage of being relatively simple and providing probabilistic predictions that clinicians can interpret.

3. **Model Training:**
   - Fit the Gaussian Naive Bayes model to the dataset using the `fit` method. This will train the model on the features and their corresponding stages.

4. **Visualization:**
   - Visualize the decision boundaries of the trained classifier with the `plot_classifier` function. This will help clinicians to see how the model separates different stages based on the features.

5. **Model Evaluation:**
   - Conduct cross-validation using the `perform_cross_validation` function to assess the model's performance across different subsets of the data. This process will provide a robust estimate of the model's accuracy, precision, recall, and F1 score.

6. **Performance Metrics:**
   - After cross-validation, generate predictions for the dataset to construct a confusion matrix and calculate a classification report. This will give detailed insights into the model's performance, such as how often it confuses different disease stages.

7. **Interpretability:**
   - Use the confusion matrix and classification report to identify areas where the model performs well or needs improvement. For instance, if certain stages are consistently misclassified, further investigation into the features or additional data collection might be necessary.

8. **Deployment Considerations:**
   - Once the model demonstrates satisfactory performance, it can be considered for deployment in a clinical trial setting, where its predictions can assist clinicians in diagnosing new cases.

By following this solution, the medical research institute can develop a Naive Bayes classification model to accurately predict disease stages, which is critical for determining the appropriate treatment plan for patients.

The script is a comprehensive example of a machine learning application using Python. It demonstrates the use of Gaussian Naive Bayes classifier from the Scikit-learn library to classify a dataset. The script is divided into several key components which are explained below:

1. **Importing Required Libraries:**
   - `numpy`: For numerical operations and handling arrays.
   - `matplotlib.pyplot`: For plotting graphs and visualizations.
   - `sklearn.naive_bayes.GaussianNB`: The Gaussian Naive Bayes classifier.
   - `sklearn.model_selection`: For model evaluation, specifically cross-validation.
   - `sklearn.metrics`: For computing various classification metrics.
   - `pandas`: For data manipulation and analysis.
   - `seaborn`: For enhanced data visualization.

2. **Function `load_data(file_path)`:**
   - This function loads data from a given file path.
   - The data is assumed to be in CSV format with the last column as the target variable.
   - It splits the data into features (`X`) and labels (`y`).

3. **Function `plot_classifier(classifier, X, y)`:**
   - It plots the decision boundaries of the given classifier.
   - A mesh grid is created to visualize the decision surface.
   - Training points are overlaid on the plot, showing how the classifier divides the space.

4. **Function `perform_cross_validation(classifier, X, y, num_validations=5)`:**
   - Performs cross-validation on the classifier.
   - It evaluates the model using different metrics like accuracy, precision, recall, and F1-score.
   - The average score for each metric across the cross-validation folds is printed.

5. **The `main()` function:**
   - Loads the dataset using `load_data`.
   - Initializes the `GaussianNB` classifier and fits it to the data.
   - Plots the classifier's decision boundaries using `plot_classifier`.
   - Performs cross-validation using `perform_cross_validation`.
   - Additionally, it predicts the labels on the training data and prints the confusion matrix and classification report for further analysis.

6. **Additional Analysis:**
   - After model training and evaluation, the script predicts the class labels for the input data.
   - A confusion matrix is generated and visualized using Seaborn's heatmap.
   - A classification report, which includes precision, recall, F1-score, and support for each class, is printed.

7. **Execution:**
   - The script runs the `main()` function if it is the main program being executed (`if __name__ == "__main__":`).
   - This is a common Python idiom for scripts intended to be executed as the main program.

This script serves as a good example of a basic machine learning workflow including data loading, model training, evaluation, and visualization of results. It's particularly useful for understanding the Gaussian Naive Bayes algorithm and basic data visualization techniques in Python.

### Decision Boundaries Visualization (First Image)

The first image is a visualization of the decision boundaries created by the Gaussian Naive Bayes classifier along with the training samples. Here's what it tells us:

- **Different Colors**: Each color region represents a different class as predicted by the classifier. The boundaries between these colors are the decision boundaries where the classifier switches from predicting one class to another.
- **Data Points**: The points are the training samples, and they are colored based on their actual class labels. The position of the points with respect to the decision boundaries indicates how well the classifier has learned to separate the classes.
- **Correct Classification**: The majority of the points lie within the region of their corresponding color, which indicates correct classification.
- **Possible Errors**: There are some points that lie on the border or on the wrong side of the boundary, which might be misclassified or represent areas where the classifier is less certain.

### Confusion Matrix (Second Image)

The second image is a confusion matrix, a tabular representation of the classifier's performance:

- **Axis Labels**: The x-axis represents the predicted labels by the classifier, while the y-axis represents the actual labels.
- **Diagonal Values**: The cells along the diagonal from top-left to bottom-right show the number of correct predictions for each class. Values of 100 in most of these cells suggest very high accuracy.
- **Off-Diagonal Values**: The off-diagonal cells show the number of misclassifications. For example, one instance of class 1 was misclassified as class 2, and one instance of class 3 was misclassified as class 0.
- **Color Intensity**: The color intensity represents the magnitude of the values, with darker colors indicating higher numbers. The scale on the right side provides a reference for the colors.

### Performance Metrics

The text provides several performance metrics:

- **Accuracy**: 99.5% accuracy indicates that the classifier correctly predicted the class labels for 99.5% of the instances in the dataset.
- **Precision (Weighted)**: At 99.52%, this metric tells us that when the classifier predicts a class, it is correct 99.52% of the time, on average, weighted by the number of instances in each class.
- **Recall (Weighted)**: Also at 99.5%, this means that the classifier correctly identifies 99.5% of the instances of each class, on average, weighted by the number of instances in each class.
- **F1 Score (Weighted)**: The F1 score is the harmonic mean of precision and recall, and at 99.5%, it is also very high, indicating a balanced classifier with both high precision and recall.

The **Classification Report** provides these metrics for each individual class:

- The **precision** column tells us the accuracy of positive predictions for each class.
- The **recall** column shows the fraction of positives that were correctly identified.
- The **f1-score** combines precision and recall into a single metric for each class.
- The **support** column shows the number of actual occurrences of each class in the dataset.
- The report also gives macro averages (unweighted mean) and weighted averages (weighted by support) for each metric.

Overall, these results indicate that the classifier has performed extremely well on this dataset, with very high accuracy and other metrics indicating a strong match between predicted and actual class labels.

## Built With

This project leverages powerful libraries and tools within the Python ecosystem for machine learning, data processing, and visualization:

- [**NumPy**](https://numpy.org/) - A fundamental package for scientific computing with Python. It provides support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.

- [**pandas**](https://pandas.pydata.org/) - An open-source data analysis and manipulation tool, built on top of the Python programming language. It offers data structures and operations for manipulating numerical tables and time series.

- [**matplotlib**](https://matplotlib.org/) - A comprehensive library for creating static, animated, and interactive visualizations in Python. It is used here to plot the decision boundaries of the classifier and overlay the training samples.

- [**Seaborn**](https://seaborn.pydata.org/) - A Python visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics. In this project, it is used for plotting the confusion matrix.

- [**Scikit-learn**](https://scikit-learn.org/stable/) - A machine learning library for Python. It features various classification, regression and clustering algorithms including support vector machines, random forests, gradient boosting, k-means, and is designed to interoperate with the Python numerical and scientific libraries NumPy and SciPy. Here, it is used for implementing the Gaussian Naive Bayes classifier, model selection, and performance metrics.

These libraries are instrumental in performing the heavy lifting for data processing, model training, and visualization tasks, which are essential components of the machine learning workflow implemented in this project.
```

In your repository, this section would help users understand the dependencies and tools they need to have installed and familiarize themselves with in order to run your code successfully. It also gives credit to the developers of these libraries.

## Getting Started

This guide will walk you through the steps needed to get this machine learning classification project up and running on your local machine for development and testing purposes.

### Prerequisites

Before you begin, ensure you have the following installed:
- Python (3.6 or later is recommended)
- pip (Python package installer)

### Installation

Follow these steps to set up your development environment:

1. **Clone the repository**

   ```sh
   git clone https://github.com/your-repository.git
   cd your-repository
   ```

2. **Set up a virtual environment (Optional but recommended)**

   - For Unix/macOS:

     ```sh
     python3 -m venv venv
     source venv/bin/activate
     ```

   - For Windows:

     ```sh
     python -m venv venv
     .\venv\Scripts\activate
     ```

3. **Install the required packages**

   ```sh
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

   These commands install all the necessary libraries used in this project including NumPy for numerical computations, pandas for data manipulation, matplotlib and seaborn for visualization, and scikit-learn for machine learning algorithms.

4. **Prepare the Dataset**

   Make sure you have the dataset file `data_multivar.txt` located in the root directory of the project. This file should be in a comma-separated values format, with the last column being the label and the preceding columns being the features.

5. **Run the script**

   ```sh
   python main.py
   ```

   Replace `main.py` with the actual name of your script. This will execute the main function in the script, which includes loading the data, training the Gaussian Naive Bayes model, visualizing the decision boundaries, performing cross-validation, and printing the confusion matrix and classification report.

### Running the tests

To verify that everything is set up correctly, you can run the provided tests (if any). For example:

```sh
python -m unittest discover tests
```

Replace `tests` with the directory where your tests are located.

### Troubleshooting

- If you encounter any issues with package versions, you may want to check the package documentation or look up the error messages you're getting.
- For problems related to the virtual environment, ensure that it is activated before running the script or installing packages.
- If the script can't find the dataset file, double-check that the path in the `load_data` function corresponds to the actual location of your `data_multivar.txt` file.

By following these instructions, you should be able to get the project running on your local machine without any issues.
```

This template assumes the users have basic knowledge of the command line and Python environments. It should be tailored to include any specific steps or additional dependencies your project may have.

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.
* If you have suggestions for adding or removing projects, feel free to [open an issue](https://github.com/TribeOfJudahLion/ Multiclass-Disease-Stage-Classification-with-Gaussian-Naive-Bayes/issues/new) to discuss it, or directly create a pull request after you edit the *README.md* file with necessary changes.
* Please make sure you check your spelling and grammar.
* Create individual PR for each suggestion.
* Please also read through the [Code Of Conduct](https://github.com/TribeOfJudahLion/ Multiclass-Disease-Stage-Classification-with-Gaussian-Naive-Bayes/blob/main/CODE_OF_CONDUCT.md) before posting your first idea as well.

### Creating A Pull Request

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See [LICENSE](https://github.com/TribeOfJudahLion/ Multiclass-Disease-Stage-Classification-with-Gaussian-Naive-Bayes/blob/main/LICENSE.md) for more information.

## Authors

* **Robbie** - *PhD Computer Science Student* - [Robbie](https://github.com/TribeOfJudahLion/) - **

## Acknowledgements

* []()
* []()
* []()
