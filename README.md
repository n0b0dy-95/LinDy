# LinDy
Linear Discriminant Analysis in Python

## Version:
0.23

## Description

LinDy (Linear Discriminant Analysis Tool) is a Python application built using tkinter and matplotlib libraries. It provides a graphical user interface (GUI) for performing Linear Discriminant Analysis (LDA) on datasets (Training set and Test set as .csv file) while performing both Leave-One-Out cross-validation (LOO-CV) and external set validation using the test set. LDA is a dimensionality reduction technique used in machine learning and statistics for finding the linear combinations of features that best separate different classes in the dataset.

## 2. Features and Benefits

- Load training and test datasets in CSV format.
- Perform LDA analysis with different solver options (svd, eigen, lsqr).
- Set tolerance level for convergence of the algorithm.
- Display ROC curve and Confusion Matrix plots.
- Save LDA summary and prediction results in CSV format.
- Easy-to-use graphical interface for non-technical users.
- Efficient dimensionality reduction for classification tasks.

## 3. Installation and Usage (Linux)

### Installation

1. **Clone the Repository:**
git clone <repository_url>

2. **Navigate to the Directory:**
cd path/to/LinDy_v0.23_stable

3. **Create a Virtual Environment:**

#### For Linux systems
python3 -m virtualenv venv --copies  

#### For Windows systems
python -m venv venv --copies

4. **Activate the Virtual Environment:**

#### For Linux systems
source venv/bin/activate

#### For Windows systems
.\venv\Scripts\activate

5. **Install Dependencies:**
python3 -m pip install -r requirements.txt

### Usage

6. **Run LinDy GUI:**

#### For Linux systems
python3 -m LinDy

or

python3 LinDy.py

#### For Windows systems
python -m LinDy

or

python LinDy.py

## 4. Using Method

1. Launch the LinDy application.
2. Load training and test datasets using the "Load Training Data" and "Load Test Data" buttons respectively.
3. Enter the dependent column name and index column names.
4. Select the solver method and enter the tolerance level (defaults: [Solve: "svd", Tolerance: 0.0001])).
5. Click on the "Run LDA" button to perform the analysis.
6. View and save the ROC Curve and Confusion Matrix plots for evaluation.
7. Save the summary and prediction results using the "Save Summary" button.

## 5. Dependencies

LinDy relies on the following dependencies:

- Python 3.6 and above
- tkinter (for GUI)
- matplotlib (for plotting)
- numpy (for numerical operations)
- pandas (for data manipulation)
- pillow (for image processing)

## Dependency Sources

- [tkinter](https://docs.python.org/3/library/tkinter.html)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [pillow](https://python-pillow.org/)

## License

LinDy is released under the MIT License, promoting open and collaborative software development.

## Author Details and Copyright Information

LinDy is developed by SUVANKAR BANERJEE.  
For any inquiries or support, please contact [suvankarbanerjee73@gmail.com].

## Copyright

Copyright © 2024 SUVANKAR BANERJEE <https://github.com/n0b0dy-95>

