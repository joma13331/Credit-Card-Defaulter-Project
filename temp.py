"""
I. No. normalization, no PCA
    1. load the data
    2. remove the id column
    3. split features and label
    4. Standard Scaling
    5. perform clustering
    6. train on various models
    7. compare the roc auc scores
"""
# from sklearn.model_selection import train_test_split
#
# from CCDCommonTasks.CCDDataLoader import CCDDataLoader
# from CCDCommonTasks.CCDEDA import CCDEda
# from CCDCommonTasks.CCDFeatureEngineering import CCDFeatureEngineering
# from CCDCommonTasks.CCDFeatureSelection import CCDFeatureSelection
# from CCDCommonTasks.CCDFileOperations import CCDFileOperations
# from CCDTraining.CCDClusteringTrain import CCDClusteringTrain
# from CCDTraining.CCDModelFinderTrain import CCDModelFinderTrain
# eda = CCDEda(is_training=True)
# feature_engineer = CCDFeatureEngineering(is_training=True)
# feature_selector = CCDFeatureSelection(is_training=True)
# file_operator = CCDFileOperations()
# dataloader = CCDDataLoader(is_training=True)
#
# cluster = CCDClusteringTrain()
#
# validated_data = dataloader.ccd_get_data()
#
# temp_df = feature_selector.ccd_remove_columns(validated_data, 'id')
# features, label = eda.ccd_feature_label_split(temp_df, ['default.payment.next.month'])
#
# features = feature_engineer.ccd_standard_scaling_features(features)
# num_clusters = cluster.ccd_obtain_optimum_cluster(features)
# features = cluster.ccd_create_cluster(features, num_clusters)
# features['default.payment.next.month'] = label
# list_of_cluster = features['cluster'].unique()
# print(list_of_cluster)
#
# for i in list_of_cluster:
#     cluster_data = features[features['cluster'] == i]
#     cluster_feature = cluster_data.drop(columns=['default.payment.next.month', 'cluster'])
#     cluster_label = cluster_data['default.payment.next.month']
#
#     train_x, test_x, train_y, test_y = train_test_split(cluster_feature, cluster_label, random_state=42)
#     model_finder = CCDModelFinderTrain()
#
#     model_name, model = model_finder.ccs_best_model(train_x=train_x, train_y=train_y,
#                                                     test_x=test_x, test_y=test_y)
import numpy as np

from CCDCommonTasks.CCDDataInjestion import CCDDataInjestionComplete

"""
II. No. normalization, PCA
    1. load the data
    2. remove the id column
    3. split features and label
    4. Standard Scaling
    5. perform PCA
    6. perform clustering
    7. train on various models
    8. compare the roc auc scores
"""

# from sklearn.model_selection import train_test_split
#
# from CCDCommonTasks.CCDDataLoader import CCDDataLoader
# from CCDCommonTasks.CCDEDA import CCDEda
# from CCDCommonTasks.CCDFeatureEngineering import CCDFeatureEngineering
# from CCDCommonTasks.CCDFeatureSelection import CCDFeatureSelection
# from CCDCommonTasks.CCDFileOperations import CCDFileOperations
# from CCDTraining.CCDClusteringTrain import CCDClusteringTrain
# from CCDTraining.CCDModelFinderTrain import CCDModelFinderTrain
# eda = CCDEda(is_training=True)
# feature_engineer = CCDFeatureEngineering(is_training=True)
# feature_selector = CCDFeatureSelection(is_training=True)
# file_operator = CCDFileOperations()
# dataloader = CCDDataLoader(is_training=True)
#
# cluster = CCDClusteringTrain()
#
# validated_data = dataloader.ccd_get_data()
#
# temp_df = feature_selector.ccd_remove_columns(validated_data, 'id')
# features, label = eda.ccd_feature_label_split(temp_df, ['default.payment.next.month'])
#
# features, label = feature_engineer.ccd_imbalance_handler_train(features, label)
#
# features = feature_engineer.ccd_standard_scaling_features(features)
# features = feature_engineer.ccd_pca_decomposition(features, variance_to_be_retained=0.975)
# num_clusters = cluster.ccd_obtain_optimum_cluster(features)
# features = cluster.ccd_create_cluster(features, num_clusters)
# features['default.payment.next.month'] = label
# list_of_cluster = features['cluster'].unique()
# print(list_of_cluster)
#
# for i in list_of_cluster:
#     cluster_data = features[features['cluster'] == i]
#     cluster_feature = cluster_data.drop(columns=['default.payment.next.month', 'cluster'])
#     cluster_label = cluster_data['default.payment.next.month']
#
#     train_x, test_x, train_y, test_y = train_test_split(cluster_feature, cluster_label, random_state=42)
#     train_x = train_x.round(2)
#     test_x = test_x.round(2)
#     print(train_x)
#     model_finder = CCDModelFinderTrain()
#
#     model_name, model = model_finder.ccs_best_model(train_x=train_x, train_y=train_y,
#                                                     test_x=test_x, test_y=test_y)
"""
I. normalization, PCA
    1. load the data
    2. remove the id column
    3. split features and label
    4. perform normalization
    5. Standard Scaling
    6. perform clustering
    7. train on various models
    8. compare the roc auc scores
"""
# from sklearn.model_selection import train_test_split
#
# from CCDCommonTasks.CCDDataLoader import CCDDataLoader
# from CCDCommonTasks.CCDEDA import CCDEda
# from CCDCommonTasks.CCDFeatureEngineering import CCDFeatureEngineering
# from CCDCommonTasks.CCDFeatureSelection import CCDFeatureSelection
# from CCDCommonTasks.CCDFileOperations import CCDFileOperations
# from CCDTraining.CCDClusteringTrain import CCDClusteringTrain
# from CCDTraining.CCDModelFinderTrain import CCDModelFinderTrain
# eda = CCDEda(is_training=True)
# feature_engineer = CCDFeatureEngineering(is_training=True)
# feature_selector = CCDFeatureSelection(is_training=True)
# file_operator = CCDFileOperations()
# dataloader = CCDDataLoader(is_training=True)
#
# cluster = CCDClusteringTrain()
#
# validated_data = dataloader.ccd_get_data()
#
# # print(validated_data.info())
# temp_df = feature_selector.ccd_remove_columns(validated_data, 'id')
# features, label = eda.ccd_feature_label_split(temp_df, ['default.payment.next.month'])
# numerical_feat, categorical_feat = eda.ccd_numerical_and_categorical_columns(features)
# cont_feat, discrete_feat = eda.ccd_continuous_discrete_variables(dataframe=features, num_col=numerical_feat)
# # print(cont_feat, discrete_feat)
#
# features = feature_engineer.ccd_one_hot_encoding(features, categorical_feat)
# print(features, label)
# features, label = feature_engineer.ccd_imbalance_handler_train(features, label)
# features = feature_engineer.ccd_standard_scaling_features(features)
# features = eda.ccd_obtain_normal_features(features, cont_columns=cont_feat)
# features = feature_engineer.ccd_pca_decomposition(features, variance_to_be_retained=0.99)
# num_clusters = cluster.ccd_obtain_optimum_cluster(features)
# features = cluster.ccd_create_cluster(features, num_clusters)
# features['default.payment.next.month'] = label
# list_of_cluster = features['cluster'].unique()
# print(list_of_cluster)
#
# for i in list_of_cluster:
#     cluster_data = features[features['cluster'] == i]
#     cluster_feature = cluster_data.drop(columns=['default.payment.next.month', 'cluster'])
#     cluster_label = cluster_data['default.payment.next.month']
#
#     train_x, test_x, train_y, test_y = train_test_split(cluster_feature, cluster_label, random_state=42)
#     train_x = train_x.astype(np.float32)
#     test_x = test_x.astype(np.float32)
#     print(train_x)
#     print(test_x)
#     model_finder = CCDModelFinderTrain()
#
#     model_name, model = model_finder.ccd_best_model(train_x=train_x, train_y=train_y,
#                                                     test_x=test_x, test_y=test_y)

# from CCDTraining.CCDTrainingPipeline import CCDTrainingPipeline
#
#
# trainer = CCDTrainingPipeline()
# trainer.ccd_model_train()

# pred_injestion_obj = CCDDataInjestionComplete(is_training=False, data_dir="CCDPredictionDatasets",
#                                               do_database_operation=True)
# pred_injestion_obj.ccd_data_injestion_complete()

from CCDPrediction.CCDPredictionPipeline import CCDPredictionPipeline

pred_pipeline = CCDPredictionPipeline()
print(pred_pipeline.ccd_predict())
