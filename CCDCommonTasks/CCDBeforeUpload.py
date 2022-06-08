import os
import logging
import pandas as pd


class CCDBeforeUpload:
    """
    :Class Name: CCDBeforeUpload
    :Description: This class is used to transform the Good Raw Files before uploading to
                  to cassandra database

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):

        if is_training:
            self.good_raw_path = "CCDDIV/ValidatedData/GoodRaw/"
            if not os.path.isdir("CCDLogFiles/training/"):
                os.mkdir("CCDLogFiles/training/")
            self.log_path = "CCDLogFiles/training/CCDBeforeUpload.txt"
            self.operation = "TRAINING"
        else:
            self.good_raw_path = "CCDDIV/PredictionData/GoodRaw/"
            if not os.path.isdir("CCDLogFiles/prediction/"):
                os.mkdir("CCDLogFiles/prediction/")
            self.log_path = "CCDLogFiles/prediction/CCDBeforeUpload.txt"
            self.operation = "PREDICTION"

        self.ccd_before_upload_logging = logging.getLogger("ccd_before_upload_log")
        self.ccd_before_upload_logging.setLevel(logging.INFO)
        ccd_before_upload_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccd_before_upload_handler.setFormatter(formatter)
        self.ccd_before_upload_logging.addHandler(ccd_before_upload_handler)

    def ccd_replace_missing_with_null(self):
        """
        :Method Name: ccd_replace_missing_with_null
        :Description: This method replaces all the missing values with 'null'.
        :return: None
        :On Failure: Exception
        """

        try:

            # Find all the files in the acceptable files folder and fill 'null' wherever there are missing values.
            # 'null' is being used so that cassandra database can accept missing values even in numerical columns.

            for filename in os.listdir(self.good_raw_path):
                print(filename)
                temp_df = pd.read_csv(os.path.join(self.good_raw_path, filename))
                temp_df.fillna('null', inplace=True)
                # temp_df.drop(columns='ID', inplace=True)
                temp_df.to_csv(os.path.join(self.good_raw_path, filename), header=True, index=None)
                message = f"{self.operation}: {filename} transformed successfully"
                self.ccd_before_upload_logging.info(message)

        except Exception as e:
            message = f"Data Transformation Failed: {str(e)}"
            self.ccd_before_upload_logging.error(message)
            raise e
