# -*- coding: utf-8 -*-
"""
@Created on Mon Oct 30 15:27:07 2023

@author: SUVANKAR BANERJEE
@email: suvankarbanerjee73@gmail.comm
@year: 2024
@location: Natural Science Laboratory, Dept of Pharm. tech, JU, INDIA.
"""

## Dependencies ##
import os
import numpy as np
import pandas as pd
from statsmodels.multivariate.manova import MANOVA
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import matplotlib
matplotlib.use('TkAgg')  # Set the backend
import matplotlib.pyplot as plt

class LDApy:
    # class variables
    __version__ = "0.75"

    def __init__(self):
        self.ROC_AUC_Train = ""
        self.ROC_AUC_Test = ""
        self.confusion_matrix_Train = None
        self.confusion_matrix_Test = None
        self.fpr = None
        self.tpr = None
        self.CV = LeaveOneOut()

    def Wilks_lambda(self, X, y):
        x = np.ones(X.shape[0])
        x = list(x)
        x = pd.DataFrame(x)
        x.columns = ['constant']
        X = pd.concat([X, x], axis=1)
        dp = pd.concat([X, y], axis=1)

        table = MANOVA.from_formula('X.values ~ y.values', data=dp).mv_test().results['y.values']['stat']
        Wilks_lambda = table.iloc[0, 0]
        F_value = table.iloc[0, 3]
        p_value = table.iloc[0, 4]

        return Wilks_lambda, F_value, p_value, pd.DataFrame(table)

    def load_data(self, train_file_name, test_file_name, y_col_name, index_col_name):
        folderpath = os.getcwd()

        readtrainfilename = train_file_name
        readtrainpath = os.path.join(folderpath, readtrainfilename)
        train = pd.read_csv(readtrainpath)
        print(train.columns)  # Print column names to check
        self.train_index = train[index_col_name]
        self.X_train = train.drop([index_col_name, y_col_name], axis=1)
        self.Y_obs_train = train[y_col_name]

        self.X_train += np.random.normal(0, 1e-10, self.X_train.shape)

        readtestfilename = test_file_name
        readtestpath = os.path.join(folderpath, readtestfilename)
        test = pd.read_csv(readtestpath)
        self.test_index = test[index_col_name]
        self.X_test = test.drop([index_col_name, y_col_name], axis=1)
        self.Y_obs_test = test[y_col_name]

        self.X_test += np.random.normal(0, 1e-10, self.X_test.shape)

        self.N_train = len(train)
        self.N_test = len(test)
        self.P = len(self.X_train.columns)
        self.Y_mean_train = np.mean(self.Y_obs_train)
        self.Y_mean_test = np.mean(self.Y_obs_test)


    def fit_lda(self, solver, tolerance):
        self.LDA = LinearDiscriminantAnalysis(n_components=None, priors=None, shrinkage=None,
                                              solver=solver, store_covariance=False, tol=tolerance)
        self.LDA.fit(self.X_train, self.Y_obs_train)

    def predict_lda(self):
        self.Y_pred_train_LDA = self.LDA.predict(self.X_train)
        self.Y_pred_test_LDA = self.LDA.predict(self.X_test)

    def calculate_metrics(self):
        self.feature_coefficients = pd.DataFrame(self.X_train.columns, columns=["Feature"])
        self.feature_coefficients["Coefs"] = self.LDA.coef_.flatten()
        
        try:
        	self.feature_coefficients = self.feature_coefficients.append({'Feature': 'Intercept', 'Coefs': self.LDA.intercept_[0]}, ignore_index=True)
        except:
        	new_row = pd.DataFrame([{'Feature': 'Intercept', 'Coefs': self.LDA.intercept_[0]}])
        	self.feature_coefficients = pd.concat([self.feature_coefficients, new_row], axis=0, ignore_index=True)
        
        self.Y_LOOCV_Prob = cross_val_predict(self.LDA, self.X_train, self.Y_obs_train,
                                              cv=self.CV, n_jobs=-1, method='predict_proba')[:, 1]

        self.confusion_matrix = confusion_matrix(self.Y_obs_train, self.Y_pred_train_LDA)
        self.confusion_matrix_Test = confusion_matrix(self.Y_obs_test, self.Y_pred_test_LDA)

        self.Y_Train_Prob = self.LDA.decision_function(self.X_train)
        self.Y_Test_Prob = self.LDA.decision_function(self.X_test)

        self.ROC_AUC_Train = roc_auc_score(self.Y_obs_train, self.Y_Train_Prob)
        self.ROC_AUC_LOOCV = roc_auc_score(self.Y_obs_train, self.Y_LOOCV_Prob)
        self.ROC_AUC_Test = roc_auc_score(self.Y_obs_test, self.Y_Test_Prob)

        self.Wilk_lambda_train, self.Fval_train, self.pval_train, self.table_train = self.Wilks_lambda(self.X_train, self.Y_obs_train)
        self.Wilk_lambda_test, self.Fval_test, self.pval_test, self.table_test = self.Wilks_lambda(self.X_test, self.Y_obs_test)

        self.fpr, self.tpr, self.thresholds = roc_curve(self.Y_obs_train, self.Y_Train_Prob, pos_label=1)
        self.fpr1, self.tpr1, self.thresholds1 = roc_curve(self.Y_obs_test, self.Y_Test_Prob, pos_label=1)

        self.TN, self.FP, self.FN, self.TP = self.confusion_matrix.ravel()
        self.TN1, self.FP1, self.FN1, self.TP1 = self.confusion_matrix_Test.ravel()

        self.Sensitivity = self.TP / (self.TP + self.FN)
        self.Specificity = self.TN / (self.TN + self.FP)
        self.Precision = self.TP / (self.TP + self.FP)

        self.Sensitivity_Test = self.TP1 / (self.TP1 + self.FN1)
        self.Specificity_Test = self.TN1 / (self.TN1 + self.FP1)
        self.Precision_Test = self.TP1 / (self.TP1 + self.FP1)

        self.Accuracy = (self.TP + self.TN) / (self.TP + self.FP + self.FN + self.TN)
        self.Accuracy_Test = (self.TP1 + self.TN1) / (self.TP1 + self.FP1 + self.FN1 + self.TN1)

        self.F1 = (2 * self.TP) / (2 * self.TP + self.FP + self.FN)
        self.F1_Test = (2 * self.TP1) / (2 * self.TP1 + self.FP1 + self.FN1)

        self.MCC = ((self.TP * self.TN - self.FP * self.FN) / np.sqrt((self.TP + self.FP) * (self.TP + self.FN) * (self.TN + self.FP) * (self.TN + self.FN)))
        self.MCC_Test = ((self.TP1 * self.TN1 - self.FP1 * self.FN1) / np.sqrt((self.TP1 + self.FP1) * (self.TP1 + self.FN1) * (self.TN1 + self.FP1) * (self.TN1 + self.FN1)))

        self.table_Metrics = pd.DataFrame(
            [["Train", self.ROC_AUC_Train, self.Sensitivity, self.Specificity, self.Precision, self.Accuracy, self.F1, self.MCC],
             ["LOOCV", self.ROC_AUC_LOOCV, self.Sensitivity, self.Specificity, self.Precision, self.Accuracy, self.F1, self.MCC],
             ["Test", self.ROC_AUC_Test, self.Sensitivity_Test, self.Specificity_Test, self.Precision_Test, self.Accuracy_Test, self.F1_Test, self.MCC_Test]],
            columns=["Data", "ROC AUC", "Sensitivity", "Specificity", "Precision", "Accuracy", "F1", "MCC"])

        return self.feature_coefficients, self.table_Metrics

    def plot_roc_curve(self, figsize=(8, 8), fontsize=12):
        plt.figure(figsize=figsize, dpi=100)  # Adjust the figure size and DPI as needed
        sns.set_style("ticks")
        plt.plot(self.fpr, self.tpr, color='lightsalmon', lw=2,
                 linestyle='dashed', label=f'ROC Training (AUC = {self.ROC_AUC_Train:.3f})')
        plt.plot(self.fpr1, self.tpr1, color='dodgerblue', lw=2,
                 linestyle='dashdot', label=f'ROC Test (AUC = {self.ROC_AUC_Test:.3f})')
        plt.fill_between(self.fpr, self.tpr, color='red', alpha=0.2)
        plt.fill_between(self.fpr1, self.tpr1, color='deepskyblue', alpha=0.2)
        plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.01])
        plt.xlabel('False Positive Rate', fontsize=fontsize)
        plt.ylabel('True Positive Rate', fontsize=fontsize)
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=fontsize)
        plt.legend(loc="lower right", fontsize=fontsize)
        plt.tick_params(axis='both', which='major', labelsize=fontsize)
        plt.show()

    def plot_confusion_matrix(self, figsize=(8, 8), fontsize=12):
        plt.figure()
        sns.heatmap(self.confusion_matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False,
                    xticklabels=["Predicted 0", "Predicted 1"],
                    yticklabels=["Actual 0", "Actual 1"])
        plt.title('Confusion Matrix - Train Data')
        plt.show(block=False)
        plt.pause(0.01)

        plt.figure()
        sns.heatmap(self.confusion_matrix_Test, annot=True, fmt=".0f", cmap="Blues", cbar=False,
                    xticklabels=["Predicted 0", "Predicted 1"],
                    yticklabels=["Actual 0", "Actual 1"])
        plt.title('Confusion Matrix - Test Data')
        plt.show(block=False)
        plt.pause(0.01)

    def save_results(self, filename):
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            self.feature_coefficients.to_excel(writer, sheet_name='Feature Coefficients', index=False)
            self.table_Metrics.to_excel(writer, sheet_name='Metrics', index=False)
        
    def save_predictions_to_csv(self, output_name):
        self.train_df = self.X_train.copy()
        self.train_df['Y_Obs_Train'] = self.Y_obs_train
        self.train_df['Y_Pred_Train'] = self.Y_pred_train_LDA
        self.train_df['Y_Train_Prob'] = self.Y_Train_Prob

        self.test_df = self.X_test.copy()
        self.test_df['Y_Obs_test'] = self.Y_obs_test
        self.test_df['Y_Pred_test'] = self.Y_pred_test_LDA
        self.test_df['Y_Test_Prob'] = self.Y_Test_Prob

        # Set the index of the data frames to match the input data
        self.train_df.index = self.train_index
        self.test_df.index = self.test_index

        # Save the training and test data frames as separate CSV files
        self.train_df.to_csv((output_name + "_" + "Output_Train.csv"), index=True)  # Index will be preserved
        self.test_df.to_csv((output_name + "_" +"Output_Test.csv"), index=True)    # Index will be preserved
        
    def output_metrics_string(self):
        output_str = (
            f"===========================================\n"
            f"                 LDA Summary               \n"
            f"===========================================\n\n"
            
            f"{self.feature_coefficients.to_string()}\n\n"
            
            f"===========================================\n"
            f"                  Input data               \n"
            f"===========================================\n"
            f"N-Train: {self.N_train}\n"
            f"N-Test: {self.N_test}\n"
            f"N-Features: {self.P}\n\n"            
            
            f"===========================================\n"
            f"                 Model Summary             \n"
            f"===========================================\n"
            f"Wilk's Lambda (Train): {round(self.Wilk_lambda_train, 5)}\n"
            f"Wilk's Lambda (Test): {round(self.Wilk_lambda_test, 5)}\n\n"
            
            f"ROC AUC (Train): {round(self.ROC_AUC_Train, 5)}\n"
            f"ROC AUC (LOO-CV): {round(self.ROC_AUC_LOOCV, 5)}\n"
            f"ROC AUC (Test): {round(self.ROC_AUC_Test, 5)}\n\n"
            
            f"===========================================\n"
            f"                Model Coefficients              \n"
            f"===========================================\n"
            f"{self.table_train.round(decimals=5).to_string(index=False)}\n\n"
            
            f"===========================================\n"
            f"                Train Metrics              \n"
            f"===========================================\n"
            f"TP (Train): {self.TP}\nFP (Train): {self.FP}\nFN (Train): {self.FN}\nTN (Train): {self.TN}\n\n"
            f"Sensitivity (Train): {round(self.Sensitivity, 5)}\n"
            f"Specificity (Train): {round(self.Specificity, 5)}\n"
            f"Precision (Train): {round(self.Precision, 5)}\n"
            f"Accuracy (Train): {round(self.Accuracy, 5)}\n"
            f"F1 Score (Train): {round(self.F1, 5)}\n"
            f"MCC (Train): {round(self.MCC, 5)}\n\n"
            
            f"===========================================\n"
            f"                Test Metrics               \n"
            f"===========================================\n"
            f"TP (Test): {self.TP1}\nFP (Test): {self.FP1}\nFN (Test): {self.FN1}\nTN (Test): {self.TN1}\n\n"
            f"Sensitivity (Test): {round(self.Sensitivity_Test, 5)}\n"
            f"Specificity (Test): {round(self.Specificity_Test, 5)}\n"
            f"Precision (Test): {round(self.Precision_Test, 5)}\n"
            f"Accuracy (Test): {round(self.Accuracy_Test, 5)}\n"
            f"F1 Score (Test): {round(self.F1_Test, 5)}\n"
            f"MCC (Test): {round(self.MCC_Test, 5)}\n"
            f"===========================================\n"
            )
        return output_str


if __name__ == "__main__":
    LDApy()
