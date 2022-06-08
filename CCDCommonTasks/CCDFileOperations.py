import pickle
import os
import logging


class CCDFileOperations:
    """
    This class shall be used to save the model after training
    and load the saved model for prediction.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0

    """

    def __init__(self):
        """
        :Method Name: __init__
        :Description: Initializes the logging feature
        """

        if not os.path.isdir("CCDLogFiles/"):
            os.mkdir("CCDLogFiles/")
        self.log_path = os.path.join("CCDLogFiles/", "CCDFileOperation.txt")

        self.ccd_file_operations_logging = logging.getLogger("ccd_file_operations_log")
        self.ccd_file_operations_logging.setLevel(logging.INFO)
        ccd_file_operation_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccd_file_operation_handler.setFormatter(formatter)
        self.ccd_file_operations_logging.addHandler(ccd_file_operation_handler)

    def ccd_save_model(self, model, model_dir, model_name):
        """
        :Method Name: ccd_save_model
        :Description: This method saves the passed model to the given directory

        :param model: The sklearn model to save.
        :param model_dir: The folder/directory where model need to be stored
        :param model_name: the name of the model
        :return: None
        :On Failure: Exception
        """
        try:
            path = os.path.join(model_dir, model_name)

            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)

            with open(path, 'wb') as f:
                pickle.dump(model, f)

            message = f"{model_name} has been save in {model_dir}"
            self.ccd_file_operations_logging.info(message)

        except Exception as e:
            message = f"Error while save {model_name} in {model_dir}: {str(e)}"
            self.ccd_file_operations_logging.error(message)
            raise e

    def ccd_load_model(self, model_path):
        """
        :Method Name: ccd_load_model
        :Description: This method is used to obtain the model stored at the given file path.

        :param model_path: The path where model is stored.
        :return: The model stored at the passed path.
        :On Failure: Exception
        """

        try:
            f = open(model_path, 'rb')
            model = pickle.load(f)
            message = f"model at {model_path} loaded successfully"
            self.ccd_file_operations_logging.info(message)

            return model

        except Exception as e:
            message = f"Error while loading model at {model_path}: {str(e)}"
            self.ccd_file_operations_logging.error(message)
            raise e

    def ccd_load_ml_model(self, cluster_no):
        try:
            model_dir = "CCDModels/CCDMLModels/"
            for filename in os.listdir(model_dir):
                if filename.endswith(f"_cluster_{cluster_no}.pickle"):
                    message = f"file: {filename} selected for prediction"
                    self.ccd_file_operations_logging.info(message)
                    return self.ccd_load_model(os.path.join(model_dir, filename))

            message = "No Model Found"
            self.ccd_file_operations_logging.info(message)

        except Exception as e:
            message = f"Error while trying to retrieve data: {str(e)}"
            self.ccd_file_operations_logging.error(message)
            raise e
