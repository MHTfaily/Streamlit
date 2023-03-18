import pandas as pd
import streamlit as st
import sqlite3
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from alg_classes import class_supervised_classification, missing_val, normalize_data, dummify_df, transform_qualitative_to_quantitative, get_qualitative_columns, remove_outliers, balance_classes
from sklearn.impute import IterativeImputer


# Function to read various file formats
def read_file(file):

    # Ask if the file has a header
    header = st.checkbox("Does the file have a header?", value=False)

    # Get the file type from its extension
    file_type = file.name.split('.')[-1]

    # If the file is a csv or a data file
    if file_type == 'csv' or file_type == 'data':
        # Ask for the separator to use
        separator = st.text_input("Separator", ",")
        try:
            # Try to read the csv file
            df = pd.read_csv(file, sep=separator, header=0 if header else None, encoding='utf-8')
        except Exception as e:
            st.write("Error:", e)

    # If the file is an xlsx file
    elif file_type == 'xlsx':
        if header:
            df = pd.read_excel(file, header=0)
        else:
            df = pd.read_excel(file, header=None)

    # If the file is a json file
    elif file_type == 'json':
        df = pd.read_json(file)

    # If the file is an xls file
    elif file_type == 'xls':
        if header:
            df = pd.read_excel(file, header=0)
        else:
            df = pd.read_excel(file, header=None)

    # If the file is a db, sqlite or sqlite3 file
    elif file_type == 'db' or file_type == 'sqlite' or file_type == 'sqlite3':
        # Connect to the database
        conn = sqlite3.connect(file.name)
        # Ask for the name of the table to query
        query = input("Please enter the name of the table: ")
        # Read the table into a pandas dataframe
        df = pd.read_sql(query, conn)
        # Close the connection to the database
        conn.close()

    # If the file type is not supported
    else:
        st.write("File type not supported.")
        return None

    return df


##########################################################################################################################################

    """Clean the input DataFrame based on user-selected options.

    Parameters:
    -----------
    df: pandas DataFrame
        Input DataFrame to be cleaned.
    remove_duplicates: bool, optional (default=False)
        Whether to remove duplicate rows from the DataFrame.
    strip_column_names: bool, optional (default=False)
        Whether to strip leading/trailing white space from column names.
    strip_string_cols: bool, optional (default=False)
        Whether to strip leading/trailing white space from string columns.
    convert_datetime_cols: bool, optional (default=False)
        Whether to convert date/time columns to datetime format.
    convert_categorical_cols: bool, optional (default=False)
        Whether to convert categorical columns to category type.
    remove_missing_values: bool, optional (default=False)
        Whether to remove rows with missing values.
    handle_missing_numeric_cols: str or None, optional (default=None)
        How to handle missing values in numeric columns. 
        Options: 'mean', 'median', 'mode', 'zero', or None.
    handle_missing_categorical_cols: str or None, optional (default=None)
        How to handle missing values in categorical columns. 
        Options: 'mode', 'constant', or None.
    handle_outliers_numeric_cols: str or None, optional (default=None)
        How to handle outliers in numeric columns. 
        Options: 'remove', 'clip', 'custom', or None.
    remove_missing_categorical_cols: bool, optional (default=None)
        Whether to remove rows with missing categorical values.
    remove_all_spaces_cols: bool, optional (default=False)
        Whether to remove all spaces from string columns.

    Returns:
    --------
    cleaned_df: pandas DataFrame
        Cleaned DataFrame.
    """
def clean_data(df, remove_duplicates=False, strip_column_names=False, strip_string_cols=False, 
               convert_datetime_cols=False, convert_categorical_cols=False, 
               remove_missing_values=False, handle_missing_numeric_cols=None, 
               handle_missing_categorical_cols=None, handle_outliers_numeric_cols=None, 
               handle_outliers_categorical_cols=None, 
               remove_missing_categorical_cols=None):

    old_df = df.copy()


    #Title
    st.sidebar.header("Clean your Data")
    st.sidebar.write("*Use these checkboxes to clean the data and scroll down to visualize it*")


    # create streamlit widgets to get user input
    
    remove_duplicates = st.sidebar.checkbox("Remove Duplicates", value=False)
    strip_column_names = st.sidebar.checkbox("Strip Column Names", value=False)
    strip_string_cols = st.sidebar.checkbox("Strip String Columns", value=False)
    remove_missing_values = st.sidebar.checkbox("Remove Missing Values", value=False)
    convert_datetime_cols = st.sidebar.checkbox("Convert Datetime Columns", value=False)
    convert_categorical_cols = st.sidebar.checkbox("Convert Categorical Columns", value=False)
    remove_missing_categorical_rows = st.sidebar.checkbox("Remove Missing Categorical Columns", value=False)
    remove_all_spaces_cols = st.sidebar.checkbox("Remove all spaces from string columns", value=False)

    handle_missing_numeric_cols = st.sidebar.selectbox("Handle Missing Numeric Columns", options=[None, "mean", "median", "mode"])
    handle_missing_categorical_cols = st.sidebar.selectbox("Handle Missing Categorical Columns", options=[None, "mode","constant"])
    handle_outliers_numeric_cols = st.sidebar.selectbox("Handle Outliers Numeric Columns", options=[None, "remove", "clip","custom"])


    # Remove duplicate rows
    if remove_duplicates:
        df = df.drop_duplicates()
    
    # Remove leading/trailing white space from column names
    if strip_column_names:
        df.columns = df.columns.astype(str).str.strip()

    
    # Remove leading/trailing white space from string columns
    if strip_string_cols:
        string_cols = df.select_dtypes(include='object').columns
        for col in string_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.strip()

    # Remove all spaces from string columns
    if remove_all_spaces_cols:
        string_cols = df.select_dtypes(include='object').columns
        for col in string_cols:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(' ', '')

    
    # Convert date/time columns to datetime format
    if convert_datetime_cols:
        date_cols = df.select_dtypes(include='datetime').columns
        df[date_cols] = df[date_cols].apply(lambda x: pd.to_datetime(x))
    
    # Convert categorical columns to category type
    if convert_categorical_cols:
        cat_cols = df.select_dtypes(include='category').columns
        df[cat_cols] = df[cat_cols].apply(lambda x: x.astype('category'))
    
    # Handle missing values in numeric columns if requested
    if handle_missing_numeric_cols:
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            if handle_missing_numeric_cols == 'drop':
                df = df.dropna(subset=[col])
            elif handle_missing_numeric_cols == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif handle_missing_numeric_cols == 'median':
                df[col] = df[col].fillna(df[col].median())
            elif handle_missing_numeric_cols == 'mode':
                df[col] = df[col].fillna(df[col].mode()[0])
            elif handle_missing_numeric_cols == 'zero':
                df[col] = df[col].fillna(0)
            else:
                st.write(f'Invalid method specified for {col}, leaving missing values as is.')



    # Handle missing values in categorical columns if requested
    if handle_missing_categorical_cols is not None:
        cat_cols = df.select_dtypes(include='object').columns

        if handle_missing_categorical_cols == 'mode':
            # Replace missing values with the mode (most frequent value) of each column
            df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
        elif handle_missing_categorical_cols == 'constant':
            # Replace missing values with a constant value
            df[cat_cols] = df[cat_cols].fillna('missing')
        else:
            # If handle_missing_categorical_cols is not None, but is not 'mode' or 'constant', raise an error
            st.write("handle_missing_categorical_cols must be 'mode' or 'constant'")
        
    # Handle outliers in numeric columns if requested
    if handle_outliers_numeric_cols is not None:
        numeric_cols = df.select_dtypes(include='number').columns
        for col in numeric_cols:
            if handle_outliers_numeric_cols == 'remove':
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < 3]  # Keep only data points within 3 standard deviations from the mean
            elif handle_outliers_numeric_cols == 'clip':
                df[col] = df[col].clip(lower=df[col].quantile(0.01), upper=df[col].quantile(0.99))
            else:
                st.write(f'Invalid method specified for {col}, skipping.')

    if remove_missing_categorical_rows:
        cat_cols = df.select_dtypes(include='category').columns
        cat_df = df[cat_cols].astype(str) # Convert categorical columns to string
        df = df.dropna(subset=cat_cols) # Drop rows with missing categorical values

    # Remove missing rows if requested
    if remove_missing_values:
        df = df.dropna()

    st.sidebar.markdown("""---""")
        
    return df


#######################################################################################################################################


#######################################################################################################################################



#The function explore_data() takes a Pandas DataFrame df as input and then creates a sidebar with checkboxes and dropdowns to allow the user to select different options for exploring the data.
#The function displays the descriptive statistics, data types, missing values, correlation matrix, and the distribution and countplot of numeric and categorical columns.
#It also allows for univariate and bivariate analysis by selecting columns of interest.

def explore_data(df):

    # Get list of columns
    columns = df.columns.tolist()

    # Display checkbox to select columns to display
    st.sidebar.title("Explore and Clean you Data")
    st.sidebar.markdown("""---""")
    st.sidebar.header("Explore the original data")
    st.sidebar.markdown('##### Select columns to display:')

    select_all_columns = st.sidebar.checkbox('Select all columns')

    if select_all_columns:
        columns_to_display = columns
    else:
        columns_to_display = st.sidebar.multiselect('Or Choose column(s)', columns)

    # Use selected columns in DataFrame
    if len(columns_to_display) > 0:
        df = df[columns_to_display]
        st.write(df)
    else:
        st.warning('Please select at least one column to display.')

    # Display checkbox to show descriptive statistics
    show_statistics = st.sidebar.checkbox('Show Descriptive Statistics')
    
    if show_statistics:
        if len(columns_to_display) > 0:
            st.write('### Descriptive Statistics')
            st.write(df[columns_to_display].describe())
        else:
            st.write('### Descriptive Statistics')
            st.warning('Please select at least one column to display.')
            

    # Display checkbox to show data types
    show_data_types = st.sidebar.checkbox('Show Data Types')

    if show_data_types:
        st.write('### Data Types')
        st.write(df[columns_to_display].dtypes)

        # Create a table showing count of categorical and numerical columns
        data_type_counts = df[columns_to_display].dtypes.value_counts()
        data_type_table = pd.DataFrame({'Data Type': data_type_counts.index, 'Count': data_type_counts.values})
        data_type_table['Data Type'] = data_type_table['Data Type'].astype(str).str.capitalize()
        st.write('### Column Type Counts')
        st.write(data_type_table)



    # Display checkbox to show missing values
    show_missing_values = st.sidebar.checkbox('Show Missing Values')
    
    if show_missing_values:
        st.write('### Missing Values')
        st.write(df[columns_to_display].isnull().sum())
    
    # Display checkbox to show correlation matrix
    show_correlation = st.sidebar.checkbox('Show Correlation Matrix')

    if show_correlation:
        st.write('### Correlation Matrix')
        df_float = df.astype(float)
        corr = df_float.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        plt.figure(figsize=(10, 8))  # set the figure size
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm')
        st.pyplot()

    
    # Display checkbox to show distribution of numeric columns
    show_numeric_distribution = st.sidebar.checkbox('Show Distribution of Numeric Columns')
    
    if show_numeric_distribution:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        st.write('### Distribution of Numeric Columns')
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=col, ax=ax, color = user_colour)
            st.pyplot(fig)
    
    # Display checkbox to show boxplot of numeric columns
    show_numeric_boxplot = st.sidebar.checkbox('Show Boxplot of Numeric Columns')
    
    if show_numeric_boxplot:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        st.write('### Boxplot of Numeric Columns')
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=col, ax=ax)
            st.pyplot(fig)
        # Display checkbox to do univariate analysis
    show_univariate_analysis = st.sidebar.checkbox('Show Univariate Analysis')

    if show_univariate_analysis:
        st.write('### Univariate Analysis')
        df = df[columns]

        # Get list of numeric and categorical columns
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='category').columns.tolist()

        # Display dropdown to select column for univariate analysis
        univariate_column = st.selectbox('Select column for Univariate Analysis:', columns)

        if univariate_column in numeric_cols:
            st.write('#### Distribution of Numeric Column:', univariate_column)
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=univariate_column, ax=ax, color = user_colour)
            st.pyplot(fig)

            st.write('#### Boxplot of Numeric Column:', univariate_column)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=univariate_column, ax=ax)
            st.pyplot(fig)

        elif univariate_column in categorical_cols:
            st.write('#### Countplot of Categorical Column:', univariate_column)
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=univariate_column, ax=ax, color = user_colour)
            st.pyplot(fig)

        else:
            st.write('Selected column is not numeric or categorical')

    # Display checkbox to do bivariate analysis
    show_bivariate_analysis = st.sidebar.checkbox('Show Bivariate Analysis')

    if show_bivariate_analysis:
        st.write('### Bivariate Analysis')
        df = df[columns]

        # Display dropdowns to select columns for bivariate analysis
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='category').columns.tolist()
        column1 = st.selectbox('Select first column:', columns)
        column2 = st.selectbox('Select second column:', columns)

        # Check if both selected columns are numeric
        if (column1 in numeric_cols) and (column2 in numeric_cols):
            st.write('#### Scatter Plot of Numeric Columns:', column1, 'vs', column2)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=column1, y=column2, ax=ax, color = user_colour)
            st.pyplot(fig)

            st.write('#### Line Plot of Numeric Columns:', column1, 'vs', column2)
            fig, ax = plt.subplots()
            sns.lineplot(data=df, x=column1, y=column2, ax=ax, color = user_colour)
            st.pyplot(fig)

            st.write('#### Regression Plot of Numeric Columns:', column1, 'vs', column2)
            fig, ax = plt.subplots()
            sns.regplot(data=df, x=column1, y=column2, ax=ax, color = user_colour)
            st.pyplot(fig)

        # Check if both selected columns are categorical
        elif (column1 in categorical_cols) and (column2 in categorical_cols):
            st.write('#### Grouped Bar Plot of Categorical Columns:', column1, 'vs', column2)
            fig, ax = plt.subplots()
            df.groupby([column1, column2]).size().unstack().plot(kind='bar', stacked=True, ax=ax)
            st.pyplot(fig)

        # Check if one selected column is numeric and the other is categorical
        elif (column1 in numeric_cols) and (column2 in categorical_cols):
            st.write('#### Boxplot of Numeric Column', column1, 'Grouped by Categorical Column', column2)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=column2, y=column1, ax=ax)
            st.pyplot(fig)

        elif (column2 in numeric_cols) and (column1 in categorical_cols):
            st.write('#### Boxplot of Numeric Column', column2, 'Grouped by Categorical Column', column1)
            fig, ax = plt

    st.sidebar.markdown("""---""")



#######################################################################################################################################

#This is the same function as explore_data


##########################################################################################################################################

def explore_data_after_cleaning(df):
    
    # Get list of columns
    columns = df.columns.tolist()

    # Display checkbox to select columns to display
    st.sidebar.header("Explore the cleaned data")
    st.sidebar.markdown('##### Select columns to display:')
    all_columns = st.sidebar.checkbox('Select All Columns', key='all_columns')
    if all_columns:
        columns_to_display_cleaned = columns
    else:
        columns_to_display_cleaned = st.sidebar.multiselect("Or choose columns:", columns)


    # Display checkbox to show cleaned dataframe
    show_cleaned_df = st.sidebar.checkbox('Show Cleaned Dataframe')
    if show_cleaned_df:
        if len(columns_to_display_cleaned) > 0:
            st.write(df[columns_to_display_cleaned])
        else:
            st.warning('Please select at least one column to display.')
    
    # Display checkbox to show descriptive statistics
    show_statistics_cleaned = st.sidebar.checkbox('Show descriptive statistics')
    
    if show_statistics_cleaned:
        st.write('### Descriptive Statistics')
        st.write(df[columns_to_display_cleaned].describe())
    
    # Display checkbox to show data types
    show_data_types_cleaned = st.sidebar.checkbox('Show data types')
    
    if show_data_types_cleaned:
        st.write('### Data Types')
        st.write(df[columns_to_display_cleaned].dtypes)

    # Display checkbox to show correlation matrix
    show_correlation_cleaned = st.sidebar.checkbox('Show correlation matrix')
    
    if show_correlation_cleaned:
        st.write('### Correlation Matrix')
        corr = df.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm')
        st.pyplot()
    
    # Display checkbox to show distribution of numeric columns
    show_numeric_distribution_cleaned = st.sidebar.checkbox('Show distribution of numeric columns')
    
    if show_numeric_distribution_cleaned:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        st.write('### Distribution of Numeric Columns')
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=col, ax=ax, color = user_colour)
            st.pyplot(fig)
    
    # Display checkbox to show countplot of categorical columns
    show_categorical_countplot_cleaned = st.sidebar.checkbox('Show countplot of categorical columns')
    
    if show_categorical_countplot_cleaned:
        categorical_cols = df.select_dtypes(include='category').columns.tolist()
        st.write('### Countplot of Categorical Columns')
        for col in categorical_cols:
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=col, ax=ax, color = user_colour)
            st.pyplot(fig)
    
    # Display checkbox to show boxplot of numeric columns
    show_numeric_boxplot_cleaned = st.sidebar.checkbox('Show boxplot of numeric columns')
    
    if show_numeric_boxplot_cleaned:
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        st.write('### Boxplot of Numeric Columns')
        for col in numeric_cols:
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=col, ax=ax)
            st.pyplot(fig)
        # Display checkbox to do univariate analysis
    show_univariate_analysis_cleaned = st.sidebar.checkbox('Show univariate analysis')

    if show_univariate_analysis_cleaned:
        st.write('### Univariate Analysis')
        df = df[columns_to_display_cleaned]

        # Get list of numeric and categorical columns
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='category').columns.tolist()

        # Display dropdown to select column for univariate analysis
        univariate_column_cleaned = st.selectbox('Select column for univariate analysis:', columns)

        if univariate_column_cleaned in numeric_cols:
            st.write('#### Distribution of Numeric Column:', univariate_column_cleaned)
            fig, ax = plt.subplots()
            sns.histplot(data=df, x=univariate_column_cleaned, ax=ax, color = user_colour)
            st.pyplot(fig)

            st.write('#### Boxplot of Numeric Column:', univariate_column_cleaned)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=univariate_column_cleaned, ax=ax)
            st.pyplot(fig)

        elif univariate_column_cleaned in categorical_cols:
            st.write('#### Countplot of Categorical Column:', univariate_column_cleaned)
            fig, ax = plt.subplots()
            sns.countplot(data=df, x=univariate_column_cleaned, ax=ax, color = user_colour)
            st.pyplot(fig)

        else:
            st.write('Selected column is not numeric or categorical')

    # Display checkbox to do bivariate analysis
    show_bivariate_analysis_cleaned = st.sidebar.checkbox('Show bivariate analysis')

    if show_bivariate_analysis_cleaned:
        st.write('### Bivariate Analysis')
        df = df[columns_to_display_cleaned]

        # Display dropdowns to select columns for bivariate analysis
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='category').columns.tolist()
        column1 = st.selectbox('Select 1st column:', columns)
        column2 = st.selectbox('Select 2nd column:', columns)

        # Check if both selected columns are numeric
        if (column1 in numeric_cols) and (column2 in numeric_cols):
            st.write('#### Scatter Plot of Numeric Columns:', column1, 'vs', column2)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=column1, y=column2, ax=ax, color = user_colour)
            st.pyplot(fig)

            st.write('#### Line Plot of Numeric Columns:', column1, 'vs', column2)
            fig, ax = plt.subplots()    
            sns.lineplot(data=df, x=column1, y=column2, ax=ax, color = user_colour)
            st.pyplot(fig)

            st.write('#### Regression Plot of Numeric Columns:', column1, 'vs', column2)
            fig, ax = plt.subplots()
            sns.regplot(data=df, x=column1, y=column2, ax=ax, color = user_colour)
            st.pyplot(fig)

        # Check if both selected columns are categorical
        elif (column1 in categorical_cols) and (column2 in categorical_cols):
            st.write('#### Grouped Bar Plot of Categorical Columns:', column1, 'vs', column2)
            fig, ax = plt.subplots()
            df.groupby([column1, column2]).size().unstack().plot(kind='bar', stacked=True, ax=ax)
            st.pyplot(fig)

        # Check if one selected column is numeric and the other is categorical
        elif (column1 in numeric_cols) and (column2 in categorical_cols):
            st.write('#### Boxplot of Numeric Column', column1, 'Grouped by Categorical Column', column2)
            fig, ax = plt.subplots()
            sns.boxplot(data=df, x=column2, y=column1, ax=ax)
            st.pyplot(fig)

        elif (column2 in numeric_cols) and (column1 in categorical_cols):
            st.write('#### Boxplot of Numeric Column', column2, 'Grouped by Categorical Column', column1)
            fig, ax = plt




######################################################################################################################################


##########################################################################################################################################

user_colour = st.color_picker(label='Choose a colour for all your plots')

# User interface
def main():
    #Cache boolean flag for which section to display
    if st.session_state == {}:
        st.session_state.is_logistic_regression = False
        st.session_state.is_Random_Forest = False 
        st.session_state.is_KNN = False

    # Create two columns of equal size
    st.title("Data Explorer")
    st.write("*Upload your Data, use the sidebar to explore it and clean it, and scroll down to train a machine learning model on your cleaned data*")
    st.subheader('Upload your file')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    file = st.file_uploader(" ", type=['csv', 'xlsx', 'json', 'db', 'sqlite', 'sqlite3','data'])
    # Add a checkbox for header
    if file:
        if file is not None:
            st.success("File uploaded successfully!")
            df = read_file(file)
            st.markdown("""---""")
            if df is not None:
                col1, col2 = st.columns(2)
                with col1:  
                    st.subheader("Original Data")
                    explore_data(df)
                with col2: 
                    st.subheader("Cleaned Data")

                    cleaned_df = clean_data(df)

                    explore_data_after_cleaning(cleaned_df)

                # Save cleaned dataframe to Excel file
                df.to_excel("cleaned_data.xlsx", index=False)

                #Download the data 
                st.markdown("""---""")
                st.subheader("Download your cleaned data")
                st.write("*When you finish cleaning download the cleaned data in an Excel format using this button:*")

                #Download button for cleaned Excel file
                with open("cleaned_data.xlsx", "rb") as file:
                    btn = st.download_button(
                        label="Download cleaned data",
                        data=file,
                        file_name="cleaned_data.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                st.markdown("""---""")

                #When the user finishes cleaning and wants to train a model 
                st.subheader("Train a model")
                st.write("##### Here you can train a machine learning model on your cleaned data and predict some values")
                st.write("*Note that we will automatically clean your data to be able to train the model you choose (dealing with missing values, transform qualitative to quantitative...)*")
                #if st.button("I want to train a model"):

                #Condition
                data = cleaned_df.copy()

                if data is not None:        #here we'll resolve the problem of missing values
                    if get_qualitative_columns(data) != []:
                        st.write("The qualitative columns are:", get_qualitative_columns(data))
                    data  = transform_qualitative_to_quantitative(data) #Transform qualitative to quantitative
                    data = missing_val(data) #Dealing with missing values
                    #data = remove_outliers(df)
                    original_data = data.copy()

                    # Add a selectbox for choosing the output variable column
                    output_col = st.selectbox("Choose the output variable column:", options=data.columns)
                    st.write("*Selected output variable column:*", output_col)
                    
                    st.write("##### Try normalizing, dummifying and balancing your dataset for better accuracy scores")

                    # Create a checkbox in Streamlit
                    normalize_checkbox = st.checkbox('Normalize data')
                    st.session_state.normalize = normalize_checkbox
                    # Check if the checkbox is checked
                    if st.session_state.normalize:
                        # Normalize the data and store the result in a new variable
                        data = normalize_data(data, output_col)
                        df = data.copy()       # update df variable
                    else:
                        data = original_data
                    

                    # Create a checkbox in Streamlit
                    dummify_checkbox = st.checkbox('Dummify data')
                    st.session_state.dummify = dummify_checkbox
                    # Check if the checkbox is checked
                    if st.session_state.dummify:
                        # Normalize the data and store the result in a new variable
                        data = dummify_df(data, output_col)
                        df = data.copy() # update df variable

                    elif not st.session_state.dummify and not st.session_state.normalize:
                        data = original_data

                    
                    before_balance = data
                    # Create a checkbox in Streamlit
                    balance_checkbox = st.checkbox('Balance data')
                    # Check if the checkbox is checked
                    if balance_checkbox:
                        # Balance the data
                        data = balance_classes(data, output_col)
                        df = data.copy() # update df variable

                    else:
                        data = before_balance

                    st.write("Here is your data:")
                    st.write(data)
                    supervised = class_supervised_classification(data, output_col)

                    #Selecting an algorithm
                    if not st.session_state.is_logistic_regression and not st.session_state.is_Random_Forest and not st.session_state.is_KNN:
                        st.write("#### Please select an algorithm to continue:")

                    # Create the buttons
                    col1, col2, col3 = st.columns(3)

                    if col1.button("Logistic Regression"):
                        st.session_state.is_logistic_regression = True
                        st.session_state.is_Random_Forest = False
                        st.session_state.is_KNN = False

                    if col2.button("Random Forest"):
                        st.session_state.is_logistic_regression = False
                        st.session_state.is_Random_Forest = True
                        st.session_state.is_KNN = False

                    if col3.button("KNN"):
                        st.session_state.is_logistic_regression = False
                        st.session_state.is_Random_Forest = False            
                        st.session_state.is_KNN = True
                    
                    # Create a button to show the Logistic Regression section
                    if st.session_state.is_logistic_regression:
                        # Add section for logistic regression
                        st.write("### Report")
                        report = supervised.logistic_alg.get_report()
                        st.write(report)
                        
                        st.write("### Accuracy")
                        accuracy = supervised.logistic_alg.get_accuracy()*100
                        st.write(accuracy,"%")
                        

                        st.write(f"### Please enter {len(data.columns)-1} values for prediction (separated by \",\"):")
                        prediction_input = st.text_input("Enter data here")
                        if st.button("Predict"):
                            try:
                                new_d = prediction_input.split(',')
                                new_d = [float(i) for i in new_d]
                                cl = int(supervised.logistic_alg.predict(np.array(new_d).reshape(1,-1))[0])
                                st.write(f"The class predicted for your data is {cl}")
                            except:                    
                                st.write(f"Your data is wrong!")    
                            

                        st.write("### ROC-AUC Curve")
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        roc_auc = supervised.logistic_alg.plot_roc_auc()
                        st.pyplot(roc_auc)

                        st.write("### Feature Importance")
                        feature_importance = supervised.logistic_alg.get_importance_of_features()
                        st.write(feature_importance)

                    # Create a button to show the Random Forest section
                    if st.session_state.is_Random_Forest:
                        # Add section for Random Forest
                        st.write("### Report")
                        report = supervised.rand_forest_alg.get_report()
                        st.write(report)
                        
                        st.write("### Accuracy")
                        accuracy = supervised.rand_forest_alg.get_accuracy()*100
                        st.write(accuracy,"%")
                        

                        st.write(f"### Please enter {len(data.columns)-1} values for prediction (separated by \",\"):")
                        prediction_input = st.text_input("Enter data here")
                        if st.button("Predict"):
                            try:
                                new_d = prediction_input.split(',')
                                new_d = [float(i) for i in new_d]
                                cl = int(supervised.rand_forest_alg.predict(np.array(new_d).reshape(1,-1))[0])
                                st.write(f"The class predicted for your data is {cl}")                    
                            except:                    
                                st.write(f"Your data is wrong!")    
                            
                        st.write("### ROC-AUC Curve")
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        roc_auc = supervised.rand_forest_alg.plot_roc_auc()
                        st.pyplot(roc_auc)

                        st.write("### Feature Importance")
                        feature_importance = supervised.rand_forest_alg.get_importance_of_features()
                        st.write(feature_importance)

                    # Create a button to show the KNN section
                    if st.session_state.is_KNN:
                        # Add section for KNN
                        st.write(f"### By Grid Search best k is {supervised.knn_alg.get_best_n()}")
                        st.write("### Report")
                        report = supervised.knn_alg.get_report()
                        st.write(report)
                        
                        st.write("### Accuracy")
                        accuracy = supervised.knn_alg.get_accuracy()*100
                        st.write(accuracy,"%")
                        

                        st.write(f"### Please enter {len(data.columns)-1} values for prediction (separated by \",\"):")
                        prediction_input = st.text_input("Enter data here")
                        if st.button("Predict"):
                            try:
                                new_d = prediction_input.split(',')
                                new_d = [float(i) for i in new_d]
                                cl = int(supervised.knn_alg.predict(np.array(new_d).reshape(1,-1))[0])                    
                                st.write(f"The class predicted for your data is {cl}")
                            except:                    
                                st.write(f"Your data is wrong!")    
                            
                        st.write("### ROC-AUC Curve")
                        st.set_option('deprecation.showPyplotGlobalUse', False)
                        roc_auc = supervised.knn_alg.plot_roc_auc()
                        st.pyplot(roc_auc)

                        st.write("### Feature Importance")
                        feature_importance = supervised.knn_alg.get_importance_of_features()
                        st.write(feature_importance)
                else:
                    st.write('lol')
if __name__ == '__main__':
    main()



