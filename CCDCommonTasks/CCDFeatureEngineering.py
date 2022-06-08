import logging
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from CCDCommonTasks.CCDFileOperations import CCDFileOperations
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from imblearn.combine import SMOTETomek


class CCDFeatureEngineering:
    """
    :Class Name: EEFeatureEngineeringPred
    :Description: This class is used to modify the dataframe while performing data
                  preprocessing

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):
        """
        :Method Name: __init__
        :Description: it initializes the logging and various variables used in the class.

        :param is_training: Whether this class has been instantiated
        """

        if is_training:
            if not os.path.isdir("CCDLogFiles/training/"):
                os.mkdir("CCDLogFiles/training")
            self.log_path = os.path.join("CCDLogFiles/training/", "CCDFeatureEngineering.txt")
            self.operation = "TRAINING"
        else:
            if not os.path.isdir("CCDLogFiles/prediction/"):
                os.mkdir("CCDLogFiles/prediction")
            self.log_path = os.path.join("CCDLogFiles/prediction/", "CCDFeatureEngineering.txt")
            self.operation = "PREDICTION"

        self.categorical_feat_names_path = 'CCDRelInfo/Categorical_Features.txt'
        self.models_path = "CCDModels/"
        self.scaler_model_name = "scaler.pickle"
        self.imputer_model_name = "imputer.pickle"
        self.ohe_model_name = "ohe.pickle"
        self.pca_model_name = "pca.pickle"
        self.file_operator = CCDFileOperations()

        self.ccd_feature_engineering_logging = logging.getLogger("ccd_feature_engineering_log")
        self.ccd_feature_engineering_logging.setLevel(logging.INFO)
        ccd_feature_engineering_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccd_feature_engineering_handler.setFormatter(formatter)
        self.ccd_feature_engineering_logging.addHandler(ccd_feature_engineering_handler)

    def ccd_standard_scaling_features(self, dataframe):
        """
        :Method Name: ccd_standard_scaling_features
        :Description: This method takes in a dataframe and scales it using standard scalar
        :param dataframe: this is the dataframe that needs to be scaled
        :return: The Scaled dataset.

        :On Failure: Exception
        """
        try:

            if self.operation == 'TRAINING':
                scalar = StandardScaler()
                scaled_df = pd.DataFrame(scalar.fit_transform(dataframe), columns=dataframe.columns)
                message = f"{self.operation}: The dataset has been scaled using Standard Scalar"
                self.ccd_feature_engineering_logging.info(message)
                self.file_operator.ccd_save_model(scalar, self.models_path, self.scaler_model_name)
                return scaled_df
            else:
                scalar = self.file_operator.ccd_load_model(os.path.join(self.models_path, self.scaler_model_name))
                scaled_df = pd.DataFrame(scalar.transform(dataframe), columns=dataframe.columns)
                message = f"{self.operation}: The dataset has been scaled using Standard Scalar"
                self.ccd_feature_engineering_logging.info(message)
                return scaled_df

        except Exception as e:
            message = f"{self.operation}: Error while trying to scale data: {str(e)}"
            self.ccd_feature_engineering_logging.error(message)
            raise e

    def ccd_handling_missing_data_mcar(self, dataframe, feature_with_missing):
        """
        :Method Name: ccd_handling_missing_data_mcar
        :Description: This method replaces the missing values if there are not greater than 75% missing using KNNImputer
        :param dataframe: The dataframe where null values have to be replaced
        :param feature_with_missing: The features where
        :return: dataframe - features with imputed values
                 dropped_features - features with more than 75% null
        """
        try:
            if self.operation == 'TRAINING':
                dropped_features = []
                imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)

                for feature in feature_with_missing:
                    if dataframe[feature].isna().mean > 0.75:
                        dataframe.drop(columns=feature)
                        dropped_features.append(feature)
                        message = f"{self.operation}: Dropped {feature} as more than 75% values are missing"
                        self.ccd_feature_engineering_logging.info(message)

                    else:
                        dataframe[feature + 'nan'] = np.where(dataframe[feature].isnull(), 1, 0)

                data = imputer.fit_transform(dataframe)
                self.file_operator.ccd_save_model(imputer, self.models_path, self.imputer_model_name)
                dataframe = pd.DataFrame(data=data, columns=dataframe.columns)

                message = f"{self.operation}: missing values imputed using KNNImputer " \
                          f"for {list(set(feature_with_missing).symmetric_difference(set(dropped_features)))} "
                self.ccd_feature_engineering_logging.info(message)
                return dataframe, dropped_features

            else:
                if os.path.isfile(os.path.join(self.models_path, self.imputer_model_name)):
                    imputer = self.file_operator.ccd_load_model(os.path.join(self.models_path,
                                                                             self.imputer_model_name))
                    data = imputer.transform(dataframe)
                    dataframe = pd.DataFrame(data=data, columns=dataframe.columns)
                    message = f"{self.operation}: missing values imputed using KNNImputer " \
                              f"for {list(set(feature_with_missing))} "
                    self.ccd_feature_engineering_logging.info(message)
                return dataframe

        except Exception as e:
            message = f"Error while trying to handle missing data due to mcar: {str(e)}"
            self.ccd_feature_engineering_logging.error(message)
            raise e

    def ccd_pca_decomposition(self, dataframe, variance_to_be_retained):
        """
        :Method Name: ccd_pca_decomposition
        :Description: This method performs Principal Component Analysis of the dataframe passed. To be used when the
                      multi-collinear features contain information vital enough that they will be lost if removed for
                      future analysis.
        :param dataframe: The dataframe on which PCA has to be carried out to retain the information which may get lost
                          if feature removal was carried out.
        :param variance_to_be_retained: The amount of variance to be retained after PCA has been carried out.
        :return: pca_dataframe - The resultant dataframe after PCA has been carried out
        """
        try:
            if self.operation == 'TRAINING':

                pca = PCA(n_components=variance_to_be_retained, svd_solver="full")
                pca_data = pca.fit_transform(dataframe)

                feature_names = [f"Feature_{i+1}" for i in range(pca.n_components_)]
                pca_dataframe = pd.DataFrame(data=pca_data, columns=feature_names)

                message = f"{self.operation}: The Principal Component Analysis model is Trained"
                self.ccd_feature_engineering_logging.info(message)

                self.file_operator.ccd_save_model(pca, self.models_path, self.pca_model_name)
                message = f"{self.operation}: The Principal Component Analysis model is saved at {self.models_path}"
                self.ccd_feature_engineering_logging.info(message)

                message = f"{self.operation}: The Principal Component Analysis was carried out on the data and the no "\
                          f"of components in the resultant features for future pipeline are {pca.n_features_}"
                self.ccd_feature_engineering_logging.info(message)
                return pca_dataframe

            else:

                pca = self.file_operator.ccd_load_model(os.path.join(self.models_path, self.pca_model_name))

                message = f"{self.operation}: The Principal Component Analysis model is loaded from {self.models_path}"
                self.ccd_feature_engineering_logging.info(message)

                pca_data = pca.transform(dataframe)

                feature_names = [f"Feature_{i + 1}" for i in range(pca.n_components_)]
                pca_dataframe = pd.DataFrame(data=pca_data, columns=feature_names)
                pca_dataframe = pca_dataframe.round(1)

                message = f"{self.operation}: Data transformed into {pca.n_features_} using pca model"
                self.ccd_feature_engineering_logging.info(message)

                return pca_dataframe

        except Exception as e:
            message = f"{self.operation}: Error while doing pca decomposition: {str(e)}"
            self.ccd_feature_engineering_logging.error(message)
            raise e

    def ccd_imbalance_handler_train(self, features, labels):
        """
        :Method Name: ccd_imbalance_handler
        :param features: A pandas dataframe of input features
        :param labels: A pandas Series or Dataframe

        :return: feature_smotek - A pandas dataframe with imbalance of data balanced for the inputs
                 labels_smotek - A pandas dataframe or series with imbalance of data balanced for the labels
        """

        try:
            if self.operation == 'TRAINING':
                smotek = SMOTETomek(sampling_strategy=0.75)
                print(features.info())
                features_smotek, labels_smotek = smotek.fit_resample(features, labels)
                features_smotek = features_smotek.round(1)

                features = pd.DataFrame(data=features_smotek, columns=features.columns)
                labels = pd.DataFrame(data=labels_smotek, columns=labels.columns)

                message = f"{self.operation}: Dataset has become more balanced"
                self.ccd_feature_engineering_logging.info(message)

                return features, labels

        except Exception as e:
            message = f"{self.operation}: Error while balancing dataset: {str(e)}"
            self.ccd_feature_engineering_logging.error(message)
            raise e

    def ccd_one_hot_encoding(self, dataframe):
        """
        :Method Name: ccd_one_hot_encoding
        :Description: This method takes a dataframe and a list of categorical features on which one hot encoding has
                      to be carried out

        :param dataframe: The dataframe on which one hot encoding has to be carried out
        :return: dataframe - one hot encoded dataframe
        """
        try:
            with open(self.categorical_feat_names_path, 'r') as f:
                categorical_feat_str = f.read()

            categorical_feat = categorical_feat_str.split(',')

            message = f"{self.operation}: List of categorical features obtained"
            self.ccd_feature_engineering_logging.info(message)

            if self.operation == 'TRAINING':

                ohe = OneHotEncoder(sparse=False, drop='first')
                values = ohe.fit_transform(dataframe[categorical_feat])

                self.file_operator.ccd_save_model(model=ohe, model_dir=self.models_path,
                                                  model_name=self.ohe_model_name)

                message = f"{self.operation}: One hot encoder model saved"
                self.ccd_feature_engineering_logging.info(message)

                new_categories = []
                for l in list(ohe.categories_):
                    new_categories.extend(list(l)[1:])

                temp_dataframe = pd.DataFrame(data=values, columns=new_categories, dtype='int32')
                dataframe = pd.concat([dataframe, temp_dataframe], axis=1)
                dataframe.drop(columns=categorical_feat, inplace=True)

                message = f"{self.operation}: Deleted Columns = {categorical_feat}\n" \
                          f"Created columns = {new_categories}"
                self.ccd_feature_engineering_logging.info(message)

            else:
                ohe = self.file_operator.ccd_load_model(os.path.join(self.models_path, self.ohe_model_name))
                message = f"{self.operation}: One hot encoder model loaded"
                self.ccd_feature_engineering_logging.info(message)

                values = ohe.transform(dataframe[categorical_feat])

                new_categories = []
                for l in list(ohe.categories_):
                    new_categories.extend(list(l)[1:])

                temp_dataframe = pd.DataFrame(data=values, columns=new_categories, dtype='int32')
                dataframe = pd.concat([dataframe, temp_dataframe], axis=1)
                dataframe.drop(columns=categorical_feat, inplace=True)

                message = f"{self.operation}: Deleted Columns = {categorical_feat}\n" \
                          f"Created columns = {new_categories}"
                self.ccd_feature_engineering_logging.info(message)

            return dataframe

        except Exception as e:
            message = f"{self.operation}: Error while one hot encoding categorical features: {str(e)}"
            self.ccd_feature_engineering_logging.error(message)
            raise e
