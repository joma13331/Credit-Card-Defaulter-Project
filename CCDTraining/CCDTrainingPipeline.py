import os
import logging

from sklearn.model_selection import train_test_split

from CCDCommonTasks.CCDFileOperations import CCDFileOperations
from CCDCommonTasks.CCDDataLoader import CCDDataLoader
from CCDCommonTasks.CCDEDA import CCDEda
from CCDCommonTasks.CCDFeatureEngineering import CCDFeatureEngineering
from CCDCommonTasks.CCDFeatureSelection import CCDFeatureSelection
from CCDTraining.CCDClusteringTrain import CCDClusteringTrain
from CCDTraining.CCDModelFinderTrain import CCDModelFinderTrain


class CCDTrainingPipeline:
    """
    :Class Name: CCDTrainingPipeline
    :Description: This class contains the methods which integrates all the relevant classes and their methods
                  to perform data preprocessing, training and saving of the best model for later predictions.

    Written By: Jobin Mathew
    Interning at iNeuron Intelligence
    Version: 1.0
    """

    def __init__(self):
        """
        :Method Name: __init__
        :Description: This constructor sets up the path variables where logs and models will be stored.
                      Sets up logging.
        """
        self.operation = 'TRAINING'

        if not os.path.isdir("CCDLogFiles/training/"):
            os.mkdir("CCDLogFiles/training")
        self.log_path = os.path.join("CCDLogFiles/training/", "CCDTrainingPipeline.txt")

        self.ccd_training_pipeline_logging = logging.getLogger("ccd_training_pipeline_log")
        self.ccd_training_pipeline_logging.setLevel(logging.INFO)
        ccd_training_pipeline_handler = logging.FileHandler(self.log_path)
        formatter = logging.Formatter('%(levelname)s %(asctime)s %(message)s',
                                      datefmt='%m/%d/%Y %I:%M:%S %p')
        ccd_training_pipeline_handler.setFormatter(formatter)
        self.ccd_training_pipeline_logging.addHandler(ccd_training_pipeline_handler)

        if not os.path.isdir("CCDModels/CCDMLModels/"):
            os.mkdir("CCDModels/CCDMLModels/")
        self.ml_model_dir = "CCDModels/CCDMLModels/"

        if not os.path.isdir("CCDModels/"):
            os.mkdir("CCDModels/")
        self.cluster_dir = "CCDModels/"

        if not os.path.isdir("CCDRelInfo/"):
            os.mkdir("CCDRelInfo/")
        self.rel_info_dir = "CCDRelInfo/"

    def ccd_model_train(self):
        """
        :Method Name: ccd_model_train
        :Description: This method integrates all the relevant classes and their methods to perform
                      Data Preprocessing, Clustering and saving the best model for each of the cluster.
        :return: None
        :On Failure: Exception
        """

        try:
            message = f"{self.operation}: Start of Training Pipeline"
            self.ccd_training_pipeline_logging.info(message)

            message = f"{self.operation}: Getting Validated Data"
            self.ccd_training_pipeline_logging.info(message)

            # GETTING THE DATA
            data_loader = CCDDataLoader(is_training=True)
            validated_data = data_loader.ccd_get_data()

            message = f"{self.operation}: Validated Data Obtained"
            self.ccd_training_pipeline_logging.info(message)

            # DATA PRE-PROCESSING

            message = f"{self.operation}: Data Preprocessing started"
            self.ccd_training_pipeline_logging.info(message)

            # Initializing The objects needed for Data Preprocessing
            eda = CCDEda(is_training=True)
            feature_engineer = CCDFeatureEngineering(is_training=True)
            feature_selector = CCDFeatureSelection(is_training=True)
            file_operator = CCDFileOperations()

            # Removing The 'id' column
            temp_df = feature_selector.ccd_remove_columns(validated_data, 'id')
            message = f"{self.operation}: Removed the 'id' column"
            self.ccd_training_pipeline_logging.info(message)

            # Splitting the data into features and label
            features, label = eda.ccd_feature_label_split(temp_df, ['default.payment.next.month'])
            message = f"{self.operation}: Separated the features and labels"
            self.ccd_training_pipeline_logging.info(message)

            is_null_present, columns_with_null = eda.ccd_features_with_missing_values(features)

            col_to_drop = []
            if is_null_present:
                features, dropped_features = feature_engineer.ccd_handling_missing_data_mcar(features,
                                                                                             columns_with_null)
                col_to_drop.extend(dropped_features)

            message = f"{self.operation}: Checked for null values and if any were present imputed them"
            self.ccd_training_pipeline_logging.info(message)

            numerical_feat, categorical_feat = eda.ccd_numerical_and_categorical_columns(features)
            message = f"{self.operation}: Obtained the Numerical and Categorical Features"
            self.ccd_training_pipeline_logging.info(message)

            cont_feat, discrete_feat = eda.ccd_continuous_discrete_variables(features, numerical_feat)
            message = f"{self.operation}: Obtained the Continuous and Discrete Features"
            self.ccd_training_pipeline_logging.info(message)

            features = feature_engineer.ccd_one_hot_encoding(features)
            message = f"{self.operation}: Converted Categorical features to One Hot Encoding"
            self.ccd_training_pipeline_logging.info(message)

            features, label = feature_engineer.ccd_imbalance_handler_train(features, label)
            message = f"{self.operation}: Compensated the imbalance in Data using SMOTETomek"
            self.ccd_training_pipeline_logging.info(message)

            # Feature Selection
            col_to_drop.extend(feature_selector.ee_features_with_zero_std(features))
            
            col_to_drop = list(set(col_to_drop))
            col_to_drop_str = ",".join(col_to_drop)

            with open(os.path.join(self.rel_info_dir, "columns_to_drop.txt"), 'w') as f:
                f.write(col_to_drop_str)

            features = feature_selector.ccd_remove_columns(features, col_to_drop)
            message = f"{self.operation}: Dropped all the irrelevant columns after feature selection"
            self.ccd_training_pipeline_logging.info(message)

            features = feature_engineer.ccd_standard_scaling_features(features)
            message = f"{self.operation}: All the features have been scaled"
            self.ccd_training_pipeline_logging.info(message)

            features = eda.ccd_obtain_normal_features(features, cont_feat)
            message = f"{self.operation}: Converted all possible continuous columns to normal"
            self.ccd_training_pipeline_logging.info(message)

            features = feature_engineer.ccd_pca_decomposition(features, variance_to_be_retained=0.99)
            message = f"{self.operation}: Performed PCA and retained 99% of variance"
            self.ccd_training_pipeline_logging.info(message)

            message = f"{self.operation}: Data Preprocessing completed"
            self.ccd_training_pipeline_logging.info(message)

            # CLUSTERING

            message = f"{self.operation}: Data Clustering Started"
            self.ccd_training_pipeline_logging.info(message)

            cluster = CCDClusteringTrain()
            num_clusters = cluster.ccd_obtain_optimum_cluster(features)
            features = cluster.ccd_create_cluster(features, num_clusters)

            features['default.payment.next.month'] = label

            list_of_cluster = features['cluster'].unique()

            message = f"{self.operation}: Data Clustering Completed"
            self.ccd_training_pipeline_logging.info(message)

            # Training of Each Cluster

            for i in list_of_cluster:
                message = f"{self.operation}: Start of Training for cluster {i}"
                self.ccd_training_pipeline_logging.info(message)

                cluster_data = features[features['cluster'] == i]
                cluster_feature = cluster_data.drop(columns=['default.payment.next.month', 'cluster'])
                cluster_label = cluster_data['default.payment.next.month']

                train_x, test_x, train_y, test_y = train_test_split(cluster_feature, cluster_label, random_state=42)
                train_x = train_x
                test_x = test_x
                model_finder = CCDModelFinderTrain()
                model_name, model = model_finder.ccd_best_model(train_x=train_x, train_y=train_y,
                                                                test_x=test_x, test_y=test_y)

                file_operator.ccd_save_model(model=model, model_dir=self.ml_model_dir,
                                             model_name=f"{model_name}_cluster_{i}.pickle")

                message = f"{self.operation}:Model for cluster {i} trained"
                self.ccd_training_pipeline_logging.info(message)

            message = f"{self.operation}: Successful End of Training "
            self.ccd_training_pipeline_logging.info(message)

            message = f"{self.operation}: Training Pipeline Successfully Completed"
            self.ccd_training_pipeline_logging.info(message)

        except Exception as e:
            message = f"{self.operation}: There was an ERROR in obtaining best model: {str(e)}"
            self.ccd_training_pipeline_logging.info(message)
            raise e
