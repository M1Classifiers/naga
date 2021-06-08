# naga
Naga is a Python-based AI that would automate the data science pipeline for an arbitrary renewable energy-related dataset from clients. So far, we have prototyped three products. 

Firstly, there is "Naga, the forecaster" which automatically selects the top-performing time-series forecaster among 15 models against 5 accuracy measures given a univariate time-series dataset. In the future, we hope to improve the selection rate of this product by building a classifier trained on data collected from the results of the old algorithm. In particular, in this training phase, the labels would be the 15 models while the features would be the time-series variables of various datasets. 

Secondly, there is "Naga, the analyst" which utilizes natural language processing to automate the exploratory data analysis (EDA) of the client dataset. So far, we have managed to automate the visualization of variables according to data types. We have also collected descriptions of open-source renewable energy-related datasets from Kaggle and clustered their topics using Latent Dirichlet Allocation. In the future, we hope to perform text categorization given a dataset name and column names as features, and topic clusters as labels. This would enable us to classify datasets based solely on dataset name and column names which would help in creating topic-specific templates for the EDA.

Thirdly, there is "Naga, the researcher" which utilizes a recursive algorithm of dropping columns corresponding to categorical variables to search for clusters via k-Means, k-Modes, and k-Prototypes in an unlabeled dataset. This product would help clients discover patterns in their dataset without the need to label their data.

In the future, we also hope to include other aspects of the data science pipeline, such as web scraping and model deployment.
