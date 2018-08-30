Machine Learning ReadMe
===
This file captures machine learning algorithm use.

1. [Ensemble Methods](#ensemble-methods)
2. [Convolutional Neural Networkds](#convolutional-neural-networkds)
3. [Stacked Auto Encoders](#stacked-auto-encoders)


### Ensemble Methods
Ensemble Methods

* Combine the decisions from multiple models to improve overall performance.  

Multiple models make predictions("a vote") for each data point:

* **Max Voting**: Classification problems. Predictions from majority of models used as final prediction-mode of all the predictions.
* **Average**: Regression problems / probabilities for classification problems. Take an average of predictions from all the models.
* **Weighted Average**: Models assigned different weights defining the importance of each model for prediction.

Learning Techniques:

* **Stacking**: Use predictions from multiple models to build a new model for making predictions on the test set
* **Blending**: Same approach as stacking but uses only a holdout (validation) set from the train set to make predictions. 
* **Bagging**: Combine results of multiple models to get a generalized result. Bagging(Bootstrap Aggregating) uses subsets (bags) to get a fair idea of the distribution (complete set). 
* **Boosting**: Sequential process, subsequent models attempt to correct the errors of the previous model. Succeeding models are dependent on the previous model.











##### Bagging meta-estimator
Classification and regression problems

* Params:
    - **base_estimator**: It defines the base estimator to fit on random subsets of the dataset. When nothing is specified, the base estimator is a decision tree.
    - **n_estimators**: It is the number of base estimators to be created. The number of estimators should be carefully tuned as a large number would take a very long time to run, while a very small number might not provide the best results.
    - **max_samples**: This parameter controls the size of the subsets. It is the maximum number of samples to train each base estimator.
    - **max_features**: Controls the number of features to draw from the whole dataset. It defines the maximum number of features required to train each base estimator.
    - **n_jobs**: The number of jobs to run in parallel. Set this value equal to the cores in your system. If -1, the number of jobs is set to the number of cores.
    - **random_state**: It specifies the method of random split. When random state value is same for two models, the random selection is same for both models. This parameter is useful when you want to compare different models.

##### Random Forest
Bagging technique

* Params:
    - **n_estimators**: It defines the number of decision trees to be created in a random forest. Generally, a higher number makes the predictions stronger and more stable, but a very large number can result in higher training time.
    - **criterion**: It defines the function that is to be used for splitting.
    The function measures the quality of a split for each feature and chooses the best split.
    - **max_features** : It defines the maximum number of features allowed for the split in each decision tree. Increasing max features usually improve performance but a very high number can decrease the diversity of each tree.
    - **max_depth**: Random forest has multiple decision trees. This parameter defines the maximum depth of the trees.
    - **min_samples_split**: Used to define the minimum number of samples required in a leaf node before a split is attempted. If the number of samples is less than the required number, the node is not split.
    - **min_samples_leaf**: This defines the minimum number of samples required to be at a leaf node. Smaller leaf size makes the model more prone to capturing noise in train data.
    - **max_leaf_nodes**: This parameter specifies the maximum number of leaf nodes for each tree. The tree stops splitting when the number of leaf nodes becomes equal to the max leaf node.
    - **n_jobs**: This indicates the number of jobs to run in parallel. Set value to -1 if you want it to run on all cores in the system.
    - **random_state**: This parameter is used to define the random selection. It is used for comparison between various models.

##### Adaptive Boosting (AdaBoost)

* Params:
    - **base_estimators**: It helps to specify the type of base estimator, that is, the machine learning algorithm to be used as base learner.
    - **n_estimators**: It defines the number of base estimators. The default value is 10, but you should keep a higher value to get better performance.
    - **learning_rate**: This parameter controls the contribution of the estimators in the final combination. There is a trade-off between learning_rate and n_estimators.
    - **max_depth**: Defines the maximum depth of the individual estimator. Tune this parameter for best performance.
    - **n_jobs**: Specifies the number of processors it is allowed to use. Set value to -1 for maximum processors allowed.
    - **random_state**: An integer value to specify the random data split. A definite value of random_state will always produce same results if given with same parameters and training data.

##### Adaptive Boosting (AdaBoost)
Regression/classification. Uses boosting technique, combining a number of weak learners to form a strong learner.

* Params:
    - **min_samples_split**: Defines the minimum number of samples (or observations) which are required in a node to be considered for splitting. Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
    - **min_samples_leaf**: Defines the minimum samples required in a terminal or leaf node. Generally, lower values should be chosen for imbalanced class problems because the regions in which the minority class will be in the majority will be very small.
    - **min_weight_fraction_leaf**: Similar to min_samples_leaf but defined as a fraction of the total number of observations instead of an integer.
    - **max_depth**: The maximum depth of a tree. Used to control over-fitting as higher depth will allow the model to learn relations very specific to a particular sample. Should be tuned using CV.
    - **max_leaf_nodes**: The maximum number of terminal nodes or leaves in a tree.
    Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves. If this is defined, GBM will ignore max_depth.
    - **max_features**: The number of features to consider while searching for the best split. These will be randomly selected.
    As a thumb-rule, the square root of the total number of features works great but we should check up to 30-40% of the total number of features.
    Higher values can lead to over-fitting but it generally depends on a case to case scenario.

##### XGBoost
Advanced implementation of gradient boosting algorithm.

* Params:
    - **Regularization**: Standard GBM implementation has no regularisation like XGBoost. Thus XGBoost also helps to reduce overfitting.
    - **Parallel Processing**: XGBoost implements parallel processing and is faster than GBM . XGBoost also supports implementation on Hadoop.
    - **High Flexibility**: XGBoost allows users to define custom optimization objectives and evaluation criteria adding a whole new dimension to the model.
    - **Handling Missing Values**: XGBoost has an in-built routine to handle missing values.
    - **Tree Pruning**: XGBoost makes splits up to the max_depth specified and then starts pruning the tree backwards and removes splits beyond which there is no positive gain.
    - **Built-in Cross-Validation**: XGBoost allows a user to run a cross-validation at each iteration of the boosting process and thus it is easy to get the exact optimum number of boosting iterations in a single run.
    - Since XGBoost takes care of the missing values itself, you do not have to impute the missing values. 
        + **nthread**: This is used for parallel processing and the number of cores in the system should be entered..If you wish to run on all cores, do not input this value. The algorithm will detect it automatically.
        + **eta**: Analogous to learning rate in GBM. Makes the model more robust by shrinking the weights on each step.
        + **min_child_weight**:  Defines the minimum sum of weights of all observations required in a child. Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
        + **max_depth**: It is used to define the maximum depth. Higher depth will allow the model to learn relations very specific to a particular sample.
        + **max_leaf_nodes**: The maximum number of terminal nodes or leaves in a tree.
        Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
        If this is defined, GBM will ignore max_depth.
        + **gamma**: A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split. Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
        + **subsample**: Same as the subsample of GBM. Denotes the fraction of observations to be randomly sampled for each tree. Lower values make the algorithm more conservative and prevent overfitting but values that are too small might lead to under-fitting.
        + **colsample_bytree**: It is similar to max_features in GBM. Denotes the fraction of columns to be randomly sampled for each tree.

##### Light GBM
Light GBM beats all other algorithms when dataset is extremely large.

* Params:
    - **num_iterations**: It defines the number of boosting iterations to be performed.
    - **num_leaves**: This parameter is used to set the number of leaves to be formed in a tree. In case of Light GBM, since splitting takes place leaf-wise rather than depth-wise, num_leaves must be smaller than 2^(max_depth), otherwise, it may lead to overfitting.
    - **min_data_in_leaf**: A very small value may cause overfitting. It is also one of the most important parameters in dealing with overfitting.
    - **max_depth**: It specifies the maximum depth or level up to which a tree can grow. A very high value for this parameter can cause overfitting.
    - **bagging_fraction**: It is used to specify the fraction of data to be used for each iteration. This parameter is generally used to speed up the training.
    - **max_bin**: Defines the max number of bins that feature values will be bucketed in. A smaller value of max_bin can save a lot of time as it buckets the feature values in discrete bins which is computationally inexpensive.

##### CatBoost

* Params:
    - **loss_function**: Defines the metric to be used for training.
    - **iterations**: The maximum number of trees that can be built. The final number of trees may be less than or equal to this number.
    - **learning_rate**: Defines the learning rate. Used for reducing the gradient step.
    - **border_count**: It specifies the number of splits for numerical features. It is similar to the max_bin parameter.
    - **depth**: Defines the depth of the trees.
    - **random_seed**: This parameter is similar to the ‘random_state’ parameter we have seen previously. It is an integer value to define the random seed for training.


### Convolutional Neural Networkds




### Stacked Auto Encoders










