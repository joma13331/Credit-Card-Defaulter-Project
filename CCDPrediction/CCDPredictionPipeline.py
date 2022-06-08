import os
import json
import logging
import pandas as pd

from CCDCommonTasks.CCDFileOperations import CCDFileOperations
from CCDCommonTasks.CCDDataLoader import CCDDataLoader
from CCDCommonTasks.CCDEDA import CCDEda
from CCDCommonTasks.CCDFeatureEngineering import CCDFeatureEngineering
from CCDCommonTasks.CCDFeatureSelection import CCDFeatureSelection


class CCDPredictionPipeline:
    """
    :Class Name: CCDPredictionPipeline
    :Description: This class contains the method that will perform the prediction of the
                  data submitted by the client

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self):
        """
        :Method Name: __init__
        :Description: This constructor sets up the logging feature and paths where the models and
                      relevant information are stored
        :return: None
        """

        self.operation = 'PREDICTION'
        if not os.path.isdir("CCDLogFiles/prediction/"):
            os.mkdir("CCDLogFiles/prediction")
        self.log_path = os.path.join("CCDLogFiles/prediction/", "CCDPredictionPipeline.txt")

        self.ccd_prediction_pipeline_logging = logging.getLogger("ccd_prediction_pipeline_log")
        self.ccd_prediction_pipeline_logging.setLevel(logging.INFO)
        ccd_prediction_pipeline_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccd_prediction_pipeline_handler.setFormatter(formatter)
        self.ccd_prediction_pipeline_logging.addHandler(ccd_prediction_pipeline_handler)

        if not os.path.isdir("CCDModels/CCDMLModels/"):
            os.mkdir("CCDModels/CCDMLModels/")
        self.ml_model_dir = "CCDModels/CCDMLModels/"

        if not os.path.isdir("CCDModels/"):
            os.mkdir("CCDModels/")
        self.models_dir = "CCDModels/"

        if not os.path.isdir("CCDRelInfo/"):
            os.mkdir("CCDRelInfo/")
        self.rel_info_dir = "CCDRelInfo/"
        self.cont_feat_file_name = "Continuous_Features.txt"

    def ccd_predict(self):
        """
        :Method Name: ccd_predict
        :Description: This method implements the prediction pipeline which will predict on
                      the client data during deployment.
        :return: the features and their corresponding predicted labels as a json object in string format
        """
        try:

            # Initial object setup
            message = f"{self.operation}: Start of Prediction Pipeline"
            self.ccd_prediction_pipeline_logging.info(message)

            data_loader = CCDDataLoader(is_training=False)
            eda = CCDEda(is_training=False)
            feature_engineer = CCDFeatureEngineering(is_training=False)
            feature_selector = CCDFeatureSelection(is_training=False)
            file_operator = CCDFileOperations()

            # Loading the data
            prediction_data = data_loader.ccd_get_data()

            message = f"{self.operation}: Data to predict on obtained"
            self.ccd_prediction_pipeline_logging.info(message)

            # DATA PRE-PROCESSING

            # Removing The 'id' column
            features = feature_selector.ccd_remove_columns(prediction_data, ['id', 'ID'])
            message = f"{self.operation}: Removed the 'id' and 'ID' column"
            self.ccd_prediction_pipeline_logging.info(message)

            # Removing all columns not trained on
            with open(os.path.join(self.rel_info_dir, "columns_to_drop.txt")) as f:
                val = f.read()
            col_to_drop = val.split(",")
            if col_to_drop[0] == '':
                col_to_drop = []
            features = feature_selector.ccd_remove_columns(features, col_to_drop)
            message = f"{self.operation}: Dropped all the irrelevant columns after feature selection"
            self.ccd_prediction_pipeline_logging.info(message)

            # Obtaining columns with 'null' values if present
            is_null_present, columns_with_null = eda.ccd_features_with_missing_values(features)
            # If null present handling it using KNNImputer.
            if is_null_present:
                features = feature_engineer.ccd_handling_missing_data_mcar(features, columns_with_null)
            message = f"{self.operation}: Checked for null values and if any were present imputed them"
            self.ccd_prediction_pipeline_logging.info(message)

            # performing One Hot Encoding of Categorical Features
            features = feature_engineer.ccd_one_hot_encoding(features)

            print(features)

            # Scaling the features
            features = feature_engineer.ccd_standard_scaling_features(features)
            message = f"{self.operation}: All the features have been scaled"
            self.ccd_prediction_pipeline_logging.info(message)

            # Using PowerTransformer on features which help improving normality
            with open(os.path.join(self.rel_info_dir, self.cont_feat_file_name)) as f:
                continuous_features_str = f.read()
            continuous_features = continuous_features_str.split(',')
            features = eda.ccd_obtain_normal_features(features, continuous_features)
            message = f"{self.operation}: Converted all possible continuous columns to normal"
            self.ccd_prediction_pipeline_logging.info(message)

            # Performing Principal Component Analysis
            features = feature_engineer.ccd_pca_decomposition(features, variance_to_be_retained=0.99)
            message = f"{self.operation}: Performed PCA and retained 99% of variance"
            self.ccd_prediction_pipeline_logging.info(message)

            message = f"{self.operation}: Data Preprocessing completed"
            self.ccd_prediction_pipeline_logging.info(message)

            # Performing clustering
            cluster = file_operator.ccd_load_model(os.path.join(self.models_dir, "cluster.pickle"))

            features['clusters'] = cluster.predict(features)
            features['id'] = prediction_data['id']
            features['ID'] = prediction_data['ID']

            result = []

            for i in features["clusters"].unique():
                cluster_data = features[features["clusters"] == i]
                id1 = cluster_data['id']
                id2 = cluster_data['ID']
                cluster_data = cluster_data.drop(columns=["clusters", 'id', 'ID'])
                model = file_operator.ccd_load_ml_model(i)
                pred_result = list(model.predict(cluster_data))
                result.extend(list(zip(id1, pred_result)))

            res_dataframe = pd.DataFrame(data=result, columns=["id", 'default.payment.next.month'])
            prediction_data = prediction_data.merge(right=res_dataframe, on='id', how='outer')

            prediction_data = prediction_data.round(2)
            prediction_data.to_csv("prediction_result.csv", header=True, index=False)

            message = f"{self.operation}: End of EEPrediction Pipeline"
            self.ccd_prediction_pipeline_logging.info(message)

            return json.loads(prediction_data.to_json(orient="records"))

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while performing prediction on given data: {str(e)}"
            self.ccd_prediction_pipeline_logging.error(message)
            raise e

