import os
import logging
import threading

from CCDCommonTasks.CCDDataFormatValidator import CCDDataFormatValidator
from CCDCommonTasks.CCDDBOperation import CCDDBOperation
from CCDCommonTasks.CCDBeforeUpload import CCDBeforeUpload


class CCDDataInjestionComplete:
    """
        :Class Name: CCDDataInjestionComplete
        :Description: This class utilized 3 Different classes
                        1. EEDataFormatTrain
                        2. EEBeforeUploadTrain
                        3. EEDBOperationTrain
                      to complete validation on the dataset names, columns, etc based on
                      the DSA with the client. It then uploads the valid files to a cassandra
                      Database. Finally it obtains a csv from the database to be used further
                      for preprocessing and training

        Written By: Jobin Mathew
        Interning at iNeuron Intelligence
        Version: 1.0
        """

    def __init__(self, is_training, data_dir="CCDUploadedFiles", do_database_operation=False):
        """
        :Method Name: __init__
        :Description: This method initializes the variables that will be used in methods of this class.

        :param is_training: Whether this class is instantiated for training.
        :param data_dir: Data directory where files are present.
        """
        self.data_format_validator = CCDDataFormatValidator(is_training=is_training, path=data_dir)
        self.db_operator = CCDDBOperation(is_training=is_training)
        self.data_transformer = CCDBeforeUpload(is_training=is_training)

        if is_training:
            self.operation = 'TRAINING'
            if not os.path.isdir("CCDLogFiles/training/"):
                os.mkdir("CCDLogFiles/training")
            self.log_path = os.path.join("CCDLogFiles/training/", "CCDDataInjestionComplete.txt")
        else:
            self.operation = 'TRAINING'
            if not os.path.isdir("CCDLogFiles/prediction/"):
                os.mkdir("CCDLogFiles/prediction")
            self.log_path = os.path.join("CCDLogFiles/prediction/", "CCDDataInjestionComplete.txt")

        self.do_database_operation = do_database_operation
        self.ccd_data_injestion_logging = logging.getLogger("ccd_data_injestion_log")
        self.ccd_data_injestion_logging.setLevel(logging.INFO)
        ccd_data_injestion_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccd_data_injestion_handler.setFormatter(formatter)
        self.ccd_data_injestion_logging.addHandler(ccd_data_injestion_handler)

    def ccd_data_injestion_complete(self):
        """
        :Method Name: ccd_data_injestion_complete
        :Description: This method is used to complete the entire data validation,
                      data injestion process to store the data in a database and
                      convert it for further usage in our project work

        :return: None
        :On Failure: Exception
        """
        try:
            message = f"{self.operation}: Start of Injestion and Validation"
            self.ccd_data_injestion_logging.info(message)

            length_date, length_time, dataset_col_names, dataset_col_num = self.data_format_validator.\
                ccd_value_from_schema()
            regex = self.data_format_validator.ccd_regex_file_name()
            self.data_format_validator.ccd_validating_file_name(regex)
            self.data_format_validator.ccd_validate_column_length(dataset_col_num)
            self.data_format_validator.ccd_validate_whole_columns_as_empty()
            self.data_format_validator.ccd_move_bad_files_to_archive()

            message = f"{self.operation}: Raw Data Validation complete"
            self.ccd_data_injestion_logging.info(message)

            message = f"{self.operation}: Start of Data Transformation"
            self.ccd_data_injestion_logging.info(message)

            self.data_transformer.ccd_replace_missing_with_null()

            message = f"{self.operation}: Data Transformation Complete"
            self.ccd_data_injestion_logging.info(message)

            message = f"{self.operation}: Start of upload of the Good Data to Cassandra Database"
            self.ccd_data_injestion_logging.info(message)

            print(self.do_database_operation)
            if self.do_database_operation:

                # Threading used to bypass time consuming database tasks to improve web application latency.
                t1 = threading.Thread(target=self.db_operator.ccd_complete_db_pipeline,
                                      args=[dataset_col_names, self.data_format_validator])
                t1.start()
                # t1 not joined so that it runs only after training has occurred.

            self.data_format_validator.ccd_convert_direct_csv_to_csv()

            message = f"{self.operation}: End of Injestion and Validation"
            self.ccd_data_injestion_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: Error During Injestion and Validation Phase{str(e)}"
            self.ccd_data_injestion_logging.error(message)
            raise e
