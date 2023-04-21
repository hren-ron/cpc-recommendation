## Recommendation Cross Project Correlated Issues based on Process Metrics

#### There are two different folders in this repository: dataset and code. The former contains the datasets in each project, and the latter contains the source code.

- The dataset contains 7 different projects, which has three different files. 
    1. The train_issues.csv contains CPC issue pairs and non-CPC issue pairs in the project. 
    2. The cross-project_train_datas.csv is a training dataset calculated on these issue pairs, including 26 process features and corresponding labels. 
    3. The cross-project_test_issues.csv is the test issues which contains the query issues and their real labels. Since we need to form issue pairs with all the issues in all project and each query issue, the test data is relatively large, so the corresponding test data is not provided in the repository. But developers can still build the corresponding test data by themselves.
- The code contains the source code, which has four inputs: the path of the training set, the path of the test set, the path of the test issues, and the path to save the result.