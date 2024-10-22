import pandas as pd
import numpy as np
class Handling_Outliers:
    """
    A class used to handle outliers in a dataset using the Interquartile Range (IQR) method.
    
    Methods
    -------
    __init__(self, data_name, column_name)
        Identifies, removes, and caps outliers for a specified column in the dataset.
    """
    
    def __init__(self, data_name, column_name):
        """
        Initializes the Handling_Outliers class and processes outliers for the specified column.
        
        Parameters
        ----------
        data_name : pd.DataFrame
            The dataset (either training or testing) to be processed.
        column_name : str
            The name of the column for which outliers need to be handled.
        
        Attributes
        ----------
        outliers : pd.DataFrame
            A DataFrame containing the identified outliers.
        df_no_outliers : pd.DataFrame
            A DataFrame excluding rows containing outliers.
        cleaned_data : pd.DataFrame
            The original dataset with outliers capped.
        """
        
        Q1 = data_name[column_name].quantile(0.25)
        Q3 = data_name[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outliers = data_name[(data_name[column_name] < lower_bound) | (data_name[column_name] > upper_bound)]
        
        # Remove outliers
        df_no_outliers = data_name[~((data_name[column_name] < lower_bound) | (data_name[column_name] > upper_bound))]
        
        # Cap outliers
        data_name[column_name] = np.where(data_name[column_name] > upper_bound, upper_bound, data_name[column_name])
        data_name[column_name] = np.where(data_name[column_name] < lower_bound, lower_bound, data_name[column_name])
        
        self.outliers = outliers
        self.df_no_outliers = df_no_outliers
        self.cleaned_data = data_name
