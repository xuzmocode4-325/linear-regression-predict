"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
   # engineer existing features for test dataset
    # 1. Isolating the target variable
    target_variable = 'load_shortfall_3h'
    # 2. Order the columns in alphabetical order
    feature_vector_df = feature_vector_df.reindex(sorted(feature_vector_df.columns), axis=1)
    # 3. Keep the "time" column in the first index position
    feature_vector_df = feature_vector_df[['time'] + [col for col in feature_vector_df.columns if col != 'time']]
    # 4. Convert all column titles to lowercase
    feature_vector_df.columns = feature_vector_df.columns.str.lower()
    # 5. Replacing null values in 'valencia_pressure' with the feature median value
    valencia_pressure_median = feature_vector_df['valencia_pressure'].median()
    feature_vector_df['valencia_pressure'] = feature_vector_df['valencia_pressure'].fillna(valencia_pressure_median)
    # 6. converting categorical features to a numeric format
    for col in feature_vector_df.columns:
        if feature_vector_df[col].dtype == object and col != "time":
            feature_vector_df[col] = feature_vector_df[col].str.extract(r'([0-9]+)')
            feature_vector_df[col] = pd.to_numeric(feature_vector_df[col])
    # 7. removing columns
    cols_to_remove = []
    for col in feature_vector_df.columns:
        if "rain" in col or "snow" in col:
            cols_to_remove.append(col)
        if "temp" in col and "min" not in col:
            cols_to_remove.append(col)
        if "clouds" in col:
            cols_to_remove.append(col)
        if "weather" in col:
            cols_to_remove.append(col)
        if "deg" in col:
            cols_to_remove.append(col)
        if "humidity" in col:
            cols_to_remove.append(col)
    # 8. selecting features to keep
    keep_cols = []
    for col in feature_vector_df.columns:
        if col not in cols_to_remove:
            keep_cols.append(col)
    feature_vector_df = feature_vector_df[keep_cols]
    # 9. changing time colum from string type to datetime object and then to a delta time feature
    feature_vector_df['time'] = pd.to_datetime(feature_vector_df['time'])
    feature_vector_df['time_delta_hours'] = (feature_vector_df['time'] - feature_vector_df['time'].min()).dt.components['hours']
    feature_vector_dict = feature_vector_df.drop("time", axis=1)
    # 10. reordering the columns to place delta time first
    feature_vector_df = feature_vector_df[['time_delta_hours'] + [col for col in feature_vector_df.columns if col != 'time_delta_hours']]
    # 11. isolating feature columns 
    features_cols = [col for col in feature_vector_df.columns != target_variable]
    # 12. scaling features to prepare for the model
    from sklearn.preprocessing import MinMaxScaler
        # Create a MinMaxScaler object
    scaler = MinMaxScaler()
        # Fit the scaler to the data
    y = feature_vector_df[target_variable]
    X = feature_vector_df[features_cols]
    scaler.fit(feature_vector_df[features_cols])
    # Transform the data
    scaled_data = scaler.transform(feature_vector_df[features_cols])
    # Update the dataframe with the scaled data
    feature_vector_df[features_cols] = scaled_data
    # Display the scaled data
    predict_vector = feature_vector_df.drop(target_variable, axis=1)
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
