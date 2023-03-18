import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from collections import Counter


def balance_classes(df, target_col):
    # Compute the class distribution
    class_distribution = dict(Counter(df[target_col]))

    # Determine the smallest class size
    min_class_size = min(class_distribution.values())

    # Balance each class
    balanced_dfs = []
    for class_label, class_size in class_distribution.items():
        if class_size == min_class_size:
            # No balancing required for this class
            balanced_dfs.append(df[df[target_col] == class_label])
        else:
            # Oversample the minority class
            df_class = df[df[target_col] == class_label]
            df_class_over = df_class.sample(min_class_size, replace=True)
            balanced_dfs.append(df_class_over)

    # Combine the balanced dataframes
    df_balanced = pd.concat(balanced_dfs, axis=0)

    # Shuffle the dataframe
    df_balanced = df_balanced.sample(frac=1, random_state=42)

    # Return the balanced dataframe
    return df_balanced

def remove_outliers(df, k=1.5):
    df_out = df.copy()
    numeric_cols = df.select_dtypes(include='number').columns
    for col in numeric_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        df_out = df[z_scores < 3]  # Keep only data points within 3 standard deviations from the mean
    return df_out

def transform_qualitative_to_quantitative(df):
    qualitative_cols = []
    for col in df.columns:
        try:
            df[col].astype(float)
        except ValueError:
            qualitative_cols.append(col)
    df_quantitative = df.copy()
    for col in qualitative_cols:
        df_quantitative[col], _ = pd.factorize(df[col])    
    return df_quantitative

def get_qualitative_columns(df):
    qualitative_cols = []
    for col in df.columns:
        try:
            df[col].astype(float)
        except ValueError:
            qualitative_cols.append(col)
    return qualitative_cols

def is_categorical(column):
    # Get the unique values in the column
    unique_values = column.unique()
    # If the number of unique values is less than or equal to 10% of the total number of values in the column, the column is categorical
    if len(unique_values) <= len(column) * 0.1:
        is_cat = True
    else:
        is_cat = False
    # Return the result
    return is_cat
import pandas as pd

def dummify_df(df, output_col):
    original_data = df.copy()
    df = df.copy().drop(output_col, axis=1)
    # Create a list to store the dummified DataFrames
    df_list = []
    # Loop through each column in the DataFrame
    for col in df.columns:
        # Check if the column is categorical
        if is_categorical(df[col]):
            # Use Pandas' get_dummies function to dummify the column
            dummies = pd.get_dummies(df[col], prefix=col)
            # Append the dummified DataFrame to the list
            df_list.append(dummies)
        else:
            # If the column is not categorical, append it to the list as-is
            df_list.append(df[[col]])
    # Concatenate all of the DataFrames in the list along the columns axis
    dummified_df = pd.concat(df_list, axis=1)
    # Adding the output column
    dummified_df[output_col] = original_data[output_col]
    # Return the dummified DataFrame
    return dummified_df

def missing_val(df):
    df_col = df.columns
    # Convert all values to numeric and replace non-numeric values with NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    #this imputer models each feature with missing values as a function of other features, and uses a regressor to estimate for imputation
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(df)
    # Convert back to DataFrame
    df = pd.DataFrame(data=imp.transform(df), columns=df_col)
    return df

# Define a function to normalize the data
def normalize_data(data, output_col):
    # Select all columns except for the output column
    input_cols = data.columns[data.columns != output_col]
    # Create a StandardScaler instance and fit it to the input columns
    scaler = StandardScaler()
    scaler.fit(data[input_cols])
    # Transform the input columns using the scaler
    data[input_cols] = scaler.transform(data[input_cols])
    return data

def preprocess_data(df, output_column = None): #output column is by default the last one
    
    if output_column is None:
        output_column = df.columns[-1]

    df_col = df.columns
    
    # Convert all values to numeric and replace non-numeric values with NaN
    df = df.apply(pd.to_numeric, errors='coerce')
    
    #this imputer models each feature with missing values as a function of other features, and uses a regressor to estimate for imputation
    imp = IterativeImputer(max_iter=10, random_state=0)
    imp.fit(df)
    df = imp.transform(df)
    
    # Convert back to DataFrame
    df = pd.DataFrame(data=imp.transform(df), columns=df_col)

    # Split the data into X and y
    X = df.copy().drop(output_column, axis=1)
    y = df[[output_column]]

    return X, y

def roc_auc(df_report, y_test, y_score):
    #Getting the classes from the df
    classes = df_report.index

    # binarize the labels
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_bin = lb.transform(y_test)

    # compute ROC curve and AUC for each class
    n_classes = len(lb.classes_)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
    #     print(roc_curve)
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i], drop_intermediate=False)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # generate a list of colors for each class
    colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))

    # plot ROC curves for each class
    plt.figure(figsize=(8, 6))
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (AUC = {1:0.2f})'
                 ''.format(lb.classes_[i], roc_auc[i]))

    # add diagonal line representing random classifier
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    # place the legend outside the figure
    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.show()


class logistic_alg:
    def __init__ (self, X_train, X_test, y_train, y_test):
        #saving all the data splits
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        
        # Instantiate a Logistic Regression model
        clf = LogisticRegression(max_iter=2000, random_state=42)

        # Fit the model on the training data
        clf.fit(X_train, y_train)

        #saving the model
        self.alg = clf
        
        # predict on the test set
        self.y_pred = clf.predict(X_test)
                  
        # probability of prediction on the test set
        self.y_score = clf.predict_proba(X_test)
        
        #saving the accuracy
        self.accuracy = np.round(accuracy_score(self.y_test, self.y_pred), 4)
    
    def get_report(self): #returns a df composed of precision, recall and f-score
        # generate classification report
        report = classification_report(self.y_test, self.y_pred, output_dict=True)

        # create a DataFrame from classification report
        df_report = pd.DataFrame(report).transpose()

        # exclude the rows for accuracy, macro avg, and weighted avg. And the support(last column)
        df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg']).iloc[:, :-1]

        # returns the DataFrame
        return df_report            
    
    def get_accuracy(self): #returns accuracy of the model
        return self.accuracy

    def predict(self, data):
        return self.alg.predict(data)   

    def plot_roc_auc(self):
        report = self.get_report()
        roc_auc(report,self.y_test,self.y_score)
                  
    def get_importance_of_features(self):
        # Get the coefficients of the model
        coef = self.alg.coef_[0]

        # Get the feature names from the dataframe
        features = self.X_train.columns

        # Create a dataframe to store the feature importances
        df_importance = pd.DataFrame({'Feature': features, 'Importance': coef})
        df_importance['Importance'] = abs(df_importance['Importance'])

        # Sort the features by importance
        df_importance = df_importance.sort_values(by='Importance', ascending=False)
        
        # return the df of importance of features
        return df_importance
    
                  
class rand_forest_alg:
    def __init__ (self, X_train, X_test, y_train, y_test):
        #saving all the data splits
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        
        # Create a Random Forest Classifier object with 100 trees
        rf_clf = RandomForestClassifier(n_estimators=100)

        # Fit the model on the training data
        rf_clf.fit(X_train, y_train)
        
        #saving the model
        self.alg = rf_clf

        # predict on the test set
        self.y_pred = rf_clf.predict(X_test)
        
        # probability of prediction on the test set
        self.y_score = rf_clf.predict_proba(X_test)                  
                  
        #saving the accuracy
        self.accuracy = np.round(accuracy_score(self.y_test, self.y_pred), 4)
        
    def get_report(self):
        # generate classification report
        report = classification_report(self.y_test, self.y_pred, output_dict=True)

        # create a DataFrame from classification report
        df_report = pd.DataFrame(report).transpose()

        # exclude the rows for accuracy, macro avg, and weighted avg. And the support(last column)
        df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg']).iloc[:, :-1]

        # returns the DataFrame
        return df_report
                  
    def get_accuracy(self): #returns accuracy of the model
        return self.accuracy
                  
    def predict(self, data):
        return self.alg.predict(data)                  
    
    def plot_roc_auc(self):
        report = self.get_report()
        roc_auc(report,self.y_test,self.y_score)
                  
    def get_importance_of_features(self):
        # Get the feature importances from the model
        importances = self.alg.feature_importances_

        # Get the feature names from the dataframe
        features = self.X_train.columns

        # Create a dataframe to store the feature importances
        df_importance_rf = pd.DataFrame({'Feature': features, 'Importance': importances})

        # Sort the features by importance
        df_importance_rf = df_importance_rf.sort_values(by='Importance', ascending=False)       

        # Returns dataframe
        return df_importance_rf        
                  
                  
class knn_alg: #We're taking best n using grid search
    def __init__ (self, X_train, X_test, y_train, y_test):
        #saving all the data splits
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
                  
        # Define the range of values to test for n_neighbors
        param_grid = {'n_neighbors': [2*i+1 for i in range(10)]}

        # Create a KNN model
        knn = KNeighborsClassifier()

        # Use grid search to find the best value of n_neighbors
        grid_search = GridSearchCV(knn, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        
        # Instantiate a KNN model
        knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])

        # Fit the model on the training data
        knn.fit(X_train, y_train)
        
        #saving the model and best n
        self.alg = knn
        self.best_n = grid_search.best_params_['n_neighbors']
                  
        # predict on the test set
        self.y_pred = knn.predict(X_test)
        
        # probability of prediction on the test set
        self.y_score = knn.predict_proba(X_test)                                  
        
        #saving the accuracy
        self.accuracy = np.round(accuracy_score(self.y_test, self.y_pred), 4)
    
    def get_report(self):
        # generate classification report
        report = classification_report(self.y_test, self.y_pred, output_dict=True)

        # create a DataFrame from classification report
        df_report = pd.DataFrame(report).transpose()

        # exclude the rows for accuracy, macro avg, and weighted avg. And the support(last column)
        df_report = df_report.drop(['accuracy', 'macro avg', 'weighted avg']).iloc[:, :-1]

        # returns the DataFrame
        return df_report
                  
    def get_best_n(self):
        return self.best_n
    
    def get_accuracy(self):
        return self.accuracy
    
    def predict(self, data):
        return self.alg.predict(data)
                  
    def plot_roc_auc(self):
        report = self.get_report()
        roc_auc(report,self.y_test,self.y_score)                  
                  
    def get_importance_of_features(self):
        # Select the top 10 features based on ANOVA F-value between class labels
        selector = SelectKBest(f_classif, k=10)
        selector.fit(self.X_train, self.y_train)

        # Get the feature importances
        importances = selector.scores_

        # Get the feature names from the dataframe
        features = self.X_train.columns

        # Create a dataframe to store the feature importances
        df_importance_knn = pd.DataFrame({'Feature': features, 'Importance': importances})

        # Sort the features by importance
        df_importance_knn = df_importance_knn.sort_values(by='Importance', ascending=False)      

        # Returns dataframe
        return df_importance_knn             
                  
                  
class class_supervised_classification:
    def __init__(self, data, output_column = None):

        if output_column is None:
            output_column = data.columns[-1]

        self.data = data
        #preparing the data
        self.X, self.y = preprocess_data(data, output_column)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        
        #classification algorithms
        self.logistic_alg = logistic_alg(self.X_train, self.X_test, self.y_train, self.y_test)
        self.rand_forest_alg = rand_forest_alg(self.X_train, self.X_test, self.y_train, self.y_test)
        self.knn_alg = knn_alg(self.X_train, self.X_test, self.y_train, self.y_test)