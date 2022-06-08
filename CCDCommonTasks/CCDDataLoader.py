import os

import numpy as np
import pandas as pd
import logging
from imblearn.combine import SMOTETomek


class CCDDataLoader:
    """
    :Class Name: CCDDataLoader
    :Description: This class contains the method for loading the data into
                  a pandas dataframe for future usage

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):
        if is_training:
            self.operation = 'TRAINING'
            self.data_file = 'validated_file.csv'
            if not os.path.isdir("CCDLogFiles/training/"):
                os.mkdir("CCDLogFiles/training/")
            self.log_path = "CCDLogFiles/training/CCDDataLoader.txt"

        else:
            self.operation = 'PREDICTION'
            self.data_file = 'prediction_file.csv'
            if not os.path.isdir("CCDLogFiles/prediction/"):
                os.mkdir("CCDLogFiles/prediction/")
            self.log_path = "CCDLogFiles/prediction/CCDDataLoader.txt"

        self.ccd_dataloader_logging = logging.getLogger("ccd_dataloader_log")
        self.ccd_dataloader_logging.setLevel(logging.INFO)
        ccd_dataloader_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccd_dataloader_handler.setFormatter(formatter)
        self.ccd_dataloader_logging.addHandler(ccd_dataloader_handler)

    def ccd_get_data(self):
        """
        Method Name: ccd_get_data
        Description: This method reads the data from source.
        Output: A pandas DataFrame.
        On Failure: Raise Exception
        """
        try:

            self.data = pd.read_csv(self.data_file)

            self.data['SEX'] = np.where(self.data['SEX'] == 1, 'MALE', 'FEMALE')
            self.data['EDUCATION'] = np.where(self.data['EDUCATION'] == 1, 'Graduate School',
                                              np.where(self.data['EDUCATION'] == 2, 'University',
                                                       np.where(self.data['EDUCATION'] == 3, 'High School',
                                                                'Others_EDUCATION')))
            self.data['MARRIAGE'] = np.where(self.data['MARRIAGE'] == 1, 'Married',
                                             np.where(self.data['MARRIAGE'] == 2, 'Single',
                                                      np.where(self.data['MARRIAGE'] == 3, 'Divorced',
                                                               'Others_MARRIAGE')))
            categorical_col = ['SEX', 'EDUCATION', 'MARRIAGE']

            message = f"{self.operation}: {categorical_col} converted to categorical columns"
            self.ccd_dataloader_logging.info(message)

            # To round all the values to two decimal digits as it is usually in the data files.
            self.data = self.data.round(2)
            message = f"{self.operation}: The data is loaded successfully as a pandas dataframe"
            self.ccd_dataloader_logging.info(message)
            return self.data

        except Exception as e:
            message = f"{self.operation}: Error while trying to load the data for prediction to pandas dataframe: {str(e)}"
            self.ccd_dataloader_logging.error(message)
            raise e
