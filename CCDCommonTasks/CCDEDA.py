import os
import logging
import numpy as np
from scipy.stats import normaltest
from sklearn.preprocessing import PowerTransformer
from CCDCommonTasks.CCDFileOperations import CCDFileOperations


class CCDEda:
    """
    :Class Name: CCDEda
    :Description: This class is used to explore the data given by the client and come
                  to some conclusion about the data.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self, is_training):
        """
        :Method Name: __init__
        :Description: This method is Constructor for class CCDEda.
                      Initializes variables for logging.
        :param is_training: whether this class is instantiated for training purpose.
        """

        if is_training:
            if not os.path.isdir("CCDLogFiles/training"):
                os.mkdir("CCDLogFiles/training")
            self.log_path = os.path.join("CCDLogFiles/training", "CCDEda.txt")
            self.operation = "TRAINING"

        else:
            if not os.path.isdir("CCDLogFiles/prediction"):
                os.mkdir("CCDLogFiles/prediction")
            self.log_path = os.path.join("CCDLogFiles/prediction", "CCDEda.txt")
            self.operation = "PREDICTION"

        self.numerical_feat_names_path = 'CCDRelInfo/Numerical_Features.txt'
        self.categorical_feat_names_path = 'CCDRelInfo/Categorical_Features.txt'
        self.cont_feat_names_path = 'CCDRelInfo/Continuous_Features.txt'
        self.discrete_feat_names_path = 'CCDRelInfo/Discrete_Features.txt'
        self.normal_feature_path = 'CCDRelInfo/Normal_Features.txt'
        self.power_transformed_feature = 'CCDRelInfo/Power_Transformed_Features.txt'
        self.log_transformed_feature = 'CCDRelInfo/Log_Features.txt'
        self.model_dir = "CCDModels/"
        self.power_transformer_model_name = "power_transformer.pickle"
        self.file_operator = CCDFileOperations()

        self.ccd_eda_logging = logging.getLogger("ccd_eda_log")
        self.ccd_eda_logging.setLevel(logging.INFO)
        ccd_eda_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccd_eda_handler.setFormatter(formatter)
        self.ccd_eda_logging.addHandler(ccd_eda_handler)

    def ccd_feature_label_split(self, dataframe, label_col_names):
        """
        :Method Name: ccd_feature_label_split
        :Description: This method splits the features and labels from the validated
                      dataset and it returns them

        :param dataframe: The pandas dataframe to obtain features and labels from
        :param label_col_names: the name of label columns
        :return: features - a pandas dataframe composed of all the features
                 labels - a dataseries representing the output
        :On Failure: Exception
        """

        try:
            features = dataframe.drop(columns=label_col_names)
            labels = dataframe[label_col_names]

            message = f"{self.operation}: The features and labels have been obtained from the dataset"
            self.ccd_eda_logging.info(message)

            return features, labels

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while obtaining feature and labels: {str(e)}"
            self.ccd_eda_logging.error(message)
            raise e

    def ccd_features_with_missing_values(self, dataframe):
        """
        :Method Name: ccd_features_with_missing_values
        :Description: This method finds out whether there are missing values in the
                      validated data and returns a list of feature names with missing
                      values

        :param dataframe: the Dataframe in which features with missing values are
                          required to be found
        :return: missing_val_flag - whether the dataframe has missing values or not
                 features_with_missing - If missing values are present then list of
                 columns with missing values otherwise an empty list
        :On Failure: Exception
        """
        try:
            features_with_missing = [feature for feature in dataframe.columns if dataframe[feature].isna().sum() > 0]
            missing_val_flag = False
            if len(features_with_missing) > 0:
                missing_val_flag = True

            message = f"{self.operation}: There are {len(features_with_missing)} features with missing values"
            self.ccd_eda_logging.info(message)

            return missing_val_flag, features_with_missing

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while obtaining feature and labels: {str(e)}"
            self.ccd_eda_logging.error(message)
            raise e

    def ccd_numerical_and_categorical_columns(self, dataframe):
        """
        :Method Name: ccd_numerical_and_categorical_columns
        :Description: This method return lists of numerical and categorical features in a dataframe

        :param dataframe:The dataframe from which the column name of numerical and categorical features have to be
                          obtained
        :return: numerical_features - List of all the numerical columns in the dataframe
                 categorical_features - List of all the categorical columns in the dataframe
        """
        try:
            if self.operation == 'TRAINING':

                numerical_features = [feature for feature in dataframe.columns if dataframe[feature].dtypes != 'O']
                numerical_features_str = ",".join(numerical_features)
                with open(self.numerical_feat_names_path, 'w') as f:
                    f.write(numerical_features_str)

                message = f'{self.operation}: {numerical_features_str} are the Numerical features'
                self.ccd_eda_logging.info(message)

                categorical_features = [feature for feature in dataframe.columns if dataframe[feature].dtypes == 'O']
                categorical_features_str = ",".join(categorical_features)
                with open(self.categorical_feat_names_path, 'w') as f:
                    f.write(categorical_features_str)

                message = f'{self.operation}: {categorical_features_str} are the Categorical features'
                self.ccd_eda_logging.info(message)
            else:
                with open(self.numerical_feat_names_path, 'r') as f:
                    numerical_features_str = f.read()
                numerical_features = numerical_features_str.split(',')

                message = f'{self.operation}: {numerical_features_str} are the Numerical features'
                self.ccd_eda_logging.info(message)

                with open(self.categorical_feat_names_path, 'r') as f:
                    categorical_features_str = f.read()
                categorical_features = categorical_features_str.split(',')

                message = f'{self.operation}: {categorical_features_str} are the Categorical features'
                self.ccd_eda_logging.info(message)

            return numerical_features, categorical_features

        except Exception as e:
            message = f'{self.operation}: Error in obtaining the Numerical and Categorical ' \
                      f'features from the data: {str(e)}'
            self.ccd_eda_logging.error(message)

    def ccd_continuous_discrete_variables(self, dataframe, num_col):
        """
        :Method Name: ccd_continuous_discrete_variables
        :Description: This method return lists of continuous and discrete features in a dataframe

        :param dataframe: The dataframe from which the column name of continuous and discrete features have to be
                          obtained
        :param num_col: List of all the numerical columns in the dataframe
        :return: cont_feat - list of continuous features in the given dataframe
                 discrete_feat - list of discrete features in the given dataframe

        """
        try:
            if self.operation == 'TRAINING':

                cont_feat = [feature for feature in num_col if len(dataframe[feature].unique()) >= 25]
                cont_feat_str = ",".join(cont_feat)
                with open(self.cont_feat_names_path, 'w') as f:
                    f.write(cont_feat_str)

                message = f'{self.operation}: {cont_feat_str} are the continuous features'
                self.ccd_eda_logging.info(message)

                discrete_feat = [feature for feature in num_col if len(dataframe[feature].unique()) < 25]
                discrete_feat_str = ",".join(discrete_feat)
                with open(self.discrete_feat_names_path, 'w') as f:
                    f.write(discrete_feat_str)

                message = f'{self.operation}: {discrete_feat_str} are the Discrete features'
                self.ccd_eda_logging.info(message)

            else:

                with open(self.cont_feat_names_path, 'r') as f:
                    cont_feat_str = f.read()
                cont_feat = cont_feat_str.split(',')
                message = f'{self.operation}: {cont_feat_str} are the continuous features'
                self.ccd_eda_logging.info(message)

                with open(self.discrete_feat_names_path, 'r') as f:
                    discrete_feat_str = f.read()
                discrete_feat = discrete_feat_str.split(',')
                message = f'{self.operation}: {discrete_feat_str} are the Discrete features'
                self.ccd_eda_logging.info(message)

            return cont_feat, discrete_feat

        except Exception as e:
            message = f'{self.operation}: Error in obtaining the Continuous and Discrete ' \
                      f'features from the data: {str(e)}'
            self.ccd_eda_logging.error(message)

    def ccd_normal_not_normal_distributed_features(self, dataframe, cont_columns):
        """
        :Method Name: ccd_normal_not_normal_distributed_features
        :param dataframe: the dataframe which needs to be checked for normal and not normal features.
        :param cont_columns: the list of continuous columns
        :return: normal_features - list of normal features
                 not_normal_features - list of features which are not normal
        """
        try:

            normal_features = []
            not_normal_features = []
            for feature in cont_columns:
                if normaltest(dataframe[feature].values)[1] >= 0.05:
                    normal_features.append(feature)
                else:
                    not_normal_features.append(feature)
            message = f'{self.operation}: {normal_features} are originally normal'
            self.ccd_eda_logging.info(message)

            normal_features_str = ','.join(normal_features)
            with open(self.normal_feature_path, 'w') as f:
                f.write(normal_features_str)

            return normal_features, not_normal_features

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while obtaining normal " \
                      f"features and features which are not normal: {str(e)}"
            self.ccd_eda_logging.error(message)
            raise e

    def ccd_obtain_normal_features(self, dataframe, cont_columns):
        """
        :Method Name: ccd_obtain_normal_features
        :param dataframe: The dataframe which needs to convert its columns to normal if possible.
        :param cont_columns: the features which are continuous in nature
        :return:
        """
        try:
            if self.operation == 'TRAINING':

                normal_features, not_normal_features = self.ccd_normal_not_normal_distributed_features(dataframe,
                                                                                                       cont_columns)
                feature_power_transformed = []
                for feature in not_normal_features:
                    power_transformer_temp = PowerTransformer()
                    transformed_data = power_transformer_temp.fit_transform(np.array(dataframe[feature]).reshape(-1, 1))

                    if normaltest(transformed_data)[0] < normaltest(dataframe[feature])[0]:
                        feature_power_transformed.append(feature)

                feature_power_transformed_str = ",".join(feature_power_transformed)

                with open(self.power_transformed_feature, 'w') as f:
                    f.write(feature_power_transformed_str)

                power_transformer = PowerTransformer()
                dataframe[feature_power_transformed] = power_transformer.fit_transform(
                    dataframe[feature_power_transformed])
                self.file_operator.ccd_save_model(power_transformer, self.model_dir, self.power_transformer_model_name)

            else:

                with open(self.power_transformed_feature, 'r') as f:
                    feature_power_transformed_str = f.read()
                    feature_power_transformed = feature_power_transformed_str.split(",")

                power_transformer = self.file_operator.ccd_load_model(os.path.join(self.model_dir,
                                                                                   self.power_transformer_model_name))

                dataframe[feature_power_transformed] = power_transformer.transform(dataframe[feature_power_transformed])

            return dataframe

        except Exception as e:
            message = f"{self.operation}: There was an ERROR while converting all possible columns to normal " \
                      f"features : {str(e)}"
            self.ccd_eda_logging.error(message)
            raise e
