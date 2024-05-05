# **
 **Our project is Upper Confidence Bound (UCB) Tuning**
 

#### This project is centered around optimizing hyperparameter settings for classification algorithms using the Upper Confidence Bound (UCB) algorithm. Our aim is to swiftly identify optimal hyperparameters by initially focusing on the most promising ones. The scope of this project encompasses implementing hyperparameter tuning strategies for Gradient Boosting Trees (XGBoost), Support Vector Classifier, and Random Forest algorithms.


### **Table of Contents**

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Usage](#usage)
- [Upper Confidence Bound](#upperconfidencebound)
- [Gradient Boosting Trees (XGBoost)](#XGboost)
- [Support Vector Classifier](#SVC)
- [Random Forest Classifier](#RFC)
- [Results](#results)
- [Contributions](#contributions)

### Project Structure

#### The project has the following structure:

- #### [EDA_Titanic.ipynb](EDA_Titanic.ipynb) : Jupyter Notebook containing Exploratory Data Analysis (EDA) of the Titanic dataset, including dataset analysis, interesting findings, and visualizations.
- #### [utils.py](utils.py): Python module containing utility functions used in this project.
- #### [XGBoost_Random.ipynb](XGBoost_Random.ipynb): Jupyter Notebook exploring the hyperparameter space of the XGBoost classifier, applying hyperparameter tuning using the UCB algorithm, comparing with randomly chosen hyperparameter configurations, and evaluating the validation error.
- #### [SupportVectorClassifier_Random.ipynb](SupportVectorClassifier_Random.ipynb): Jupyter Notebook exploring the hyperparameter space of the Support Vector Classifier, applying hyperparameter tuning using the UCB algorithm, comparing with randomly chosen hyperparameter configurations, and evaluating the validation error.
- #### [RandomForest_Random.ipynb](RandomForest_Random.ipynb): Jupyter Notebook exploring the hyperparameter space of the Random Forest classifier, applying hyperparameter tuning using the UCB algorithm, comparing with randomly chosen hyperparameter configurations, and evaluating the validation error.

### Dataset

#### The dataset used in this project is the Titanic dataset, which contains information about passengers aboard the Titanic, including whether they survived or not. The dataset is analyzed in the [EDA_Titanic.ipynb](EDA_Titanic.ipynb) notebook. 
#### You can access to the dataset by following this link 'https://www.kaggle.com/datasets/vinicius150987/titanic3?resource=download'.


### Usage

#### To perform hyperparameter tuning and evaluate the classification models, follow these steps:

1. #### Install the project dependencies using the command mentioned in the "Dependencies" section.
2. #### Open the [utils.py](utils.py) file and run it to ensure all the required utility functions are loaded.
3. #### Open the desired Jupyter Notebook (e.g., [XGBoost_Random.ipynb](XGBoost_Random.ipynb)) for the algorithm you want to explore.
4. #### Run the notebook cells to load the dataset, perform hyperparameter tuning using the UCB algorithm, compare with randomly chosen hyperparameter configurations, and evaluate the validation error.
5. #### Adjust the hyperparameter space and UCB parameters as needed.
6. #### Analyze the results and choose the best configuration of hyperparameters based on the evaluation metrics.
7. #### Repeat steps 2-5 for other algorithms (e.g., [SupportVectorClassifier_Random.ipynb](SupportVectorClassifier_Random.ipynb), [RandomForest_Random.ipynb](RandomForest_Random.ipynb)).





### Upper Confidence Bound


#### **The Upper Confidence Bound (UCB)** algorithm is a method for *balancing exploration* and *exploitation* in decision-making processes. Unlike approaches that randomly select actions, UCB adjusts its exploration-exploitation balance based on accumulated knowledge. 

#### Initially, it focuses on exploration by preferring actions that have been tried the least, but as more information is gathered, it shifts towards exploitation by selecting the action with the highest estimated reward.

#### The UCB algorithm selects the action 'Aₜ' at time step 't' using the formula:

#### *Aₜ = argmax[Qₜ(a) + c * sqrt((log(t))/Nₜ(a))]*

#### In this formula:
- #### *Qₜ(a)* represents the estimated value of action 'a' at time step 't'.
- #### *Nₜ(a)* is the number of times action 'a' has been selected prior to time 't'.
- #### *'c'* is a confidence value that controls the level of exploration.

#### The UCB algorithm consists of two parts: **exploitation** and **exploration**. 
- #### The exploitation part *(Qₜ(a))* selects the action with the highest estimated reward. 
- #### The exploration part *(c * sqrt((log(t))/Nₜ(a)))* introduces uncertainty and encourages the selection of actions that have been tried infrequently.

#### Initially, when an action has been tried few or no times, the uncertainty term is large, making it more likely to be selected for exploration. As an action is chosen more frequently, the uncertainty term decreases, making it less likely to be selected for exploration. The exploration term is larger for actions that have been selected infrequently, reflecting the uncertainty in their reward estimates.

#### Over time, the exploration term gradually decreases, and actions are selected based primarily on the exploitation term. This allows the UCB algorithm to adapt its exploration-exploitation balance based on the available information.

#### By using the UCB algorithm, you can effectively explore the hyperparameter space and find promising configurations for hyperparameter tuning in your classification models.







### Gradient Boosting Trees (XGBoost)

#### **Gradient Boosting Trees** is a powerful ensemble learning method that combines multiple weak predictive models, typically decision trees, to create a stronger predictive model. It works by iteratively training weak models on the residuals of the previous models, with each subsequent model focusing on minimizing the errors of the previous models.

#### **XGBoost (Extreme Gradient Boosting)** is a popular implementation of gradient boosting trees that provides high performance and efficient computation. It is known for its scalability, flexibility, and ability to handle diverse types of data.

#### Here is a brief explanation of the hyperparameters that are used in the context of XGBoost:

1. #### **learning_rate**: It controls the step size or shrinkage at each boosting iteration. A lower learning rate makes the model more robust to overfitting but requires more boosting iterations to converge.

2. #### **n_estimators**: It determines the number of boosting iterations or the total number of weak models to be built. Increasing the number of estimators can improve the model's performance but also increases the computational cost.

3. #### **max_depth**: It limits the depth of the individual decision trees or weak models. A deeper tree can capture more complex patterns but is more prone to overfitting. It is essential to find an optimal balance between model complexity and generalization.

4. #### **reg_alpha**: It is the L1 regularization term applied to the weights of the leaf nodes. It encourages sparsity in the feature space and helps reduce model complexity.

5. #### **reg_lambda**: It is the L2 regularization term applied to the weights of the leaf nodes. It helps control the overall complexity of the model and reduces the impact of individual features.

#### These hyperparameters control various aspects of the XGBoost model's behavior, such as the learning rate, complexity of weak models, regularization, and trade-off between bias and variance. Tuning these hyperparameters can significantly impact the model's performance, and it often involves finding the right balance between underfitting and overfitting.

#### By trying different values for these hyperparameters, you can explore different combinations of learning rates, number of estimators, tree depths, and regularization terms to find the optimal configuration that yields the best performance on your specific dataset and problem.





### Support Vector Classifier


#### **Support Vector Classifier (SVC)** is a popular machine learning algorithm used for classification tasks. It constructs a hyperplane or a set of hyperplanes in a high-dimensional feature space to separate different classes of data points.

#### Here is a brief explanation of the hyperparameters used in the context of SVC:

1. #### **C**: It is the regularization parameter that controls the trade-off between maximizing the margin and minimizing the classification error. A smaller value of C allows for a larger margin but may lead to more misclassifications, while a larger value of C focuses on correctly classifying training examples but may result in a narrower margin.

2. #### **kernel**: It specifies the type of kernel function used to transform the input space into a higher-dimensional feature space. Common choices for the kernel function include 'linear', which represents a linear hyperplane, and 'rbf' (Radial Basis Function), which allows for non-linear decision boundaries.

3. #### **gamma**: It defines the kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. It determines the influence of each training example on the decision boundary. A smaller value of gamma indicates a larger influence, resulting in a more complex and tightly fit decision boundary.

4. #### **degree**: It is the degree of the polynomial kernel function used when the kernel is set to 'poly'. It controls the flexibility of the decision boundary for polynomial kernels.

5. #### **probability**: It indicates whether to enable probability estimates for classification. If set to 'True', SVC computes probability estimates based on the Platt scaling method.

6. #### **shrinking**: It determines whether to use the shrinking heuristic. When set to 'True', it enables a faster training process by eliminating some of the support vectors.

7. #### **cache_size**: It sets the size of the kernel cache in megabytes. A larger cache size can speed up training but requires more memory. 

#### These hyperparameters control various aspects of the SVC algorithm, such as the regularization strength, the choice of kernel function, the complexity of the decision boundary, and additional options for probability estimates and memory usage. Tuning these hyperparameters can significantly impact the model's performance, and it involves finding the right combination to balance bias and variance and achieve good generalization on the given dataset.





### Random Forest Classifier

#### Random Forest Classifier is an ensemble learning algorithm that combines multiple decision trees to make predictions. It creates a collection of decision trees and aggregates their predictions to produce a final prediction.

#### Here is a brief explanation of the hyperparameters used in the context of Random Forest Classifier:

1. #### **n_estimators**: It specifies the number of decision trees to be created in the random forest. Increasing the number of estimators can improve the model's performance, but it also increases the computational cost.

2. #### **max_depth**: It determines the maximum depth of each decision tree in the random forest. A deeper tree can capture more complex patterns but is more prone to overfitting. It is important to find an optimal balance between model complexity and generalization.

3. #### **min_samples_split**: It sets the minimum number of samples required to split an internal node during the construction of each decision tree. A higher value can prevent overfitting by ensuring that a node is split only if it contains enough samples.

4. #### **min_samples_leaf**: It sets the minimum number of samples required to be at a leaf node. Similar to min_samples_split, a higher value can prevent overfitting by ensuring that each leaf node has a sufficient number of samples.

5. #### **max_features**: It determines the maximum number of features to consider when looking for the best split at each node. 'sqrt' corresponds to the square root of the total number of features, and 'log2' corresponds to the logarithm base 2 of the total number of features. Limiting the number of features can reduce the correlation between trees and improve diversity.

6. #### **bootstrap**: It indicates whether to use bootstrap samples when building decision trees. If set to 'True', each tree is trained on a random subset of the training data with replacement.

7. #### **criterion**: It defines the quality measure used to evaluate the quality of a split. 'gini' refers to the Gini impurity, and 'entropy' refers to the information gain. Both measures aim to find the best split that maximizes the separation of classes.

#### These hyperparameters control various aspects of the Random Forest Classifier, such as the number of trees, the depth and structure of individual trees, the sampling strategy, and the splitting criteria. Tuning these hyperparameters can significantly impact the model's performance, and it involves finding the right configuration to balance bias and variance, improve generalization, and handle the complexity of the dataset.









### Results

#### Each notebook provides a detailed exploration of the hyperparameter space, the best configuration of hyperparameters based on the UCB algorithm, comparison with randomly chosen hyperparameter configurations, and evaluation of the validation error. The notebooks allow you to easily identify the best-performing hyperparameter configurations for each algorithm.







### **Contributions**

#### Contributions to this project are welcome. If you find any issues or have suggestions for improvement, feel free to open an issue or submit a pull request. Please make sure to follow the contribution guidelines and maintain code quality.
