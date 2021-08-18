# Capstone Project ML Engineer with Azure

## Project Overview
This project is part of the Udacity Azure ML Nanodegree. The aim of the project is to solve an interesting problem using Azure Machine Learning.\
Two models will be created: one using *Automated ML* (denoted as *AutoML*) and one customized model whose hyperparameters are tuned using *HyperDrive*. Then, the performance of both the models will be compared and the best performing model will be deployed and consumed through *Azure Container Instance* and *REST endpoint* respectively.

The following diagram shows all the steps of the entire process:

![Project Workflow](./img/project_workflow.png)

## Project Set Up and Installation
The project contains the following files:
- `hyperparameter_tuning.ipynb`: jupyter notebook used to perform *HyperDrive*
- `automl.ipynb`: jupyter notebook used to perform *AutoML*
- `train.py`: python script used for hyperparameters tuning, containing logistic regression model
- `endpoint.py`: python script used to interact with the deployed model
- `img`: folder containing all the screenshots produced during the experiments
- `data`: folder containing `heart_failure_clinical_records_dataset.csv`, the dataset used in the project

To run the project:
1. Run the `hyperparameter_tuning.ipynb` notebook to find the best model with *HyperDrive*
2. Run the `automl.ipynb` notebook to find the best model with *AutoML*
3. Compare the two model accuracy, and choose the best performing one
4. Deploy the best model identified at the above step, running the cells of the best model notebook

## Table of Contents
- [Project Overview](#project-overview)
- [Project Set Up and Installation](#project-set-up-and-installation)
- [Dataset](#dataset)
  - [Overview](#overview)
  - [Task](#task)
  - [Access](#access)
- [Automated ML](#automated-ml)
  - [Results](#results)
- [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Results](#results)
- [Model Deployment](#model-deployment)
- [Screen Recording](#screen-recording)
- [Future Work](#future-work)

## Dataset
In this project, a [heart failure dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) from kaggle is used.\
The dataset has been stored [here](./data/heart_failure_clinical_records_dataset.csv) and is used for predicting mortality caused by heart failure.

### Overview
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.\
Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.\
People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

### Task
The dataset contains 299 observations and 13 columns (12 features + 1 target):

| Features |  Description |
|---|---|
| age | age (years) |
| anaemia  | Decrease of red blood cells or hemoglobin (boolean) |
| creatinine_phosphokinase  | Level of the CPK enzyme in the blood (mcg/L) |
| diabetes  | If the patient has diabetes (boolean)  |
| ejection_fraction  | Percentage of blood leaving the heart at each contraction (percentage) |
| high_blood_pressure  | If the patient has hypertension (boolean) |
| platelets  | Platelets in the blood (kiloplatelets/mL) |
| serum_creatinine  | Level of serum creatinine in the blood (mg/dL) |
| serum_sodium  | Level of serum sodium in the blood (mEq/L) |
| sex  | Woman or man (binary) |
| smoking  | If the patient smokes or not (boolean) |
| time  | Follow-up period (days) |

| Target | Description |
|---|---|
| DEATH_EVENT | If the patient deceased during the follow-up period (boolean) |

The task is addressed as a classification task and the goal is to predict if a patient deceases during the follow-up period, using the features listed above.\
The target, DEATH_EVENT variable, has two possible values: 0 or 1.

### Access
The dataset has been loaded as `TabularDataset` and registered in workspace using SDK:
```python
found = False
key = "Heart Failure Dataset"
description_text = "Heart Failure Dataset for Udacity Project 3"

if key in ws.datasets.keys():
        found = True
        dataset = ws.datasets[key]

if not found:
        # Create AML Dataset and register it into Workspace
        example_data = 'https://github.com/peppegili/3_Capstone_Project_ML_Engineer/blob/master/data/heart_failure_clinical_records_dataset.csv'
        dataset = Dataset.Tabular.from_delimited_files(example_data)        
        #Register Dataset in Workspace
        dataset = dataset.register(workspace=ws,
                                   name=key,
                                   description=description_text)

df = dataset.to_pandas_dataframe()
```

![Dataset](./img/dataset.png)

## Automated ML
In this task, an `AutoML` pipeline is built.
Automated machine learning, also referred to as automated ML or AutoML, is the process of automating the time-consuming, iterative tasks of machine learning model development. It allows to build ML models with high scale, efficiency, and productivity all while sustaining model quality.

`AutoMLConfig` class is responsable of the automated machine learning process. It contains the parameters for configuring the experiment run:

```python
# AutoML config
automl_settings = {
    "experiment_timeout_minutes": 20,
    "max_concurrent_iterations": 5,
    "n_cross_validations": 4,
    "primary_metric" : 'accuracy'
}

automl_config = AutoMLConfig(compute_target=compute_cluster,
                             task="classification",
                             training_data=dataset,
                             label_column_name="DEATH_EVENT",   
                             path=project_folder,
                             enable_early_stopping=True,
                             featurization='auto',
                             debug_log="automl_errors.log",
                             **automl_settings
                            )
```

The main configuration parameters are explained below:
- `experiment_timeout_minutes`: maximum amount of time in minutes that all iterations combined can take before the experiment terminates. It has been set to *20*
- `max_concurrent_iterations`: maximum number of iterations that would be executed in parallel. It has been set to *5*
- `n_cross_validations`: how many cross validations to perform when user validation data is not specified. It has been set to *4*
- `primary_metric`: the metric that Automated Machine Learning will optimize for model selection. It has been set to *accuracy*

### Results
Once the AutoML run has been submitted to the experiment, the best run and the related metrics have been collected:
```python
best_run_automl, best_model_automl = automl.get_output()
best_run_metrics_automl = best_run_automl.get_metrics()

print('Best run ID: ', best_run_automl.id)
print('Best run Accuracy: ', best_run_metrics_automl['Accuracy'])
print('Metrics: ', best_run_metrics_automl)
```
```
Best run ID: AutoML_5ab60472-18a9-4ec1-97c9-b43685d2a6dd_29
Best run Accuracy: 0.9190440060698029
Metrics: {'f1_score_macro': 0.7939297719210586, 'recall_score_micro': 0.9190440060698029, 'matthews_correlation': 0.5881065523806792, 'norm_macro_recall': 0.581082337069211, 'AUC_micro': 0.9808406538623611, 'precision_score_weighted': 0.9182078162705778, 'precision_score_macro': 0.7976971592815697, 'recall_score_weighted': 0.9190440060698029, 'AUC_weighted': 0.9481838936880631, 'recall_score_macro': 0.7905411685346055, 'balanced_accuracy': 0.7905411685346055, 'f1_score_weighted': 0.9185753244363074, 'f1_score_micro': 0.9190440060698029, 'weighted_accuracy': 0.9509449763133989, 'log_loss': 0.22503428584436114, 'average_precision_score_weighted': 0.9560620267940345, 'AUC_macro': 0.948183893688063, 'accuracy': 0.9190440060698029, 'average_precision_score_micro': 0.9815596471253963, 'average_precision_score_macro': 0.8271358405542905, 'precision_score_micro': 0.9190440060698029, 'confusion_matrix': 'aml://artifactId/ExperimentRun/dcid.AutoML_5ab60472-18a9-4ec1-97c9-b43685d2a6dd_29/confusion_matrix', 'accuracy_table': 'aml://artifactId/ExperimentRun/dcid.AutoML_5ab60472-18a9-4ec1-97c9-b43685d2a6dd_29/accuracy_table'}
```

The pipeline has been completed in 33m 18s.

![AutoML Experiment Completed](./img/automl_experiment_completed.png)

#### Best Model
![AutoML Best Model](./img/automl_best_model.png)

The best model was **VotingEnsemble** with:
- ***Parameters***
  - `max_iter` = 1000
  - `n_jobs` = 1
  - `penalty` = none
  - `tol` = 0.0001
  - ...

  The complete information has been reported below:
  ```
  Pipeline(memory=None,
       steps=[('datatransformer',
               DataTransformer(enable_dnn=False, enable_feature_sweeping=True, feature_sweeping_config={}, feature_sweeping_timeout=86400, featurization_config=None, force_text_dnn=False, is_cross_validation=True, is_onnx_compatible=False, observer=None, task='classification', working_dir='/mnt/batch/tasks/shared/LS_root/mount...), random_state=None, reg_alpha=0.3157894736842105, reg_lambda=0.8421052631578947, subsample=1))], verbose=False)), ('13', Pipeline(memory=None, steps=[('maxabsscaler', MaxAbsScaler(copy=True)), ('sgdclassifierwrapper', SGDClassifierWrapper(alpha=7.5510448979591835, class_weight='balanced', eta0=0.001, fit_intercept=True, l1_ratio=0.42857142857142855, learning_rate='constant', loss='modified_huber', max_iter=1000, n_jobs=1, penalty='none', power_t=0.7777777777777777, random_state=None, tol=0.0001))], verbose=False)), ('20', Pipeline(memory=None, steps=[('truncatedsvdwrapper', TruncatedSVDWrapper(n_components=0.7026315789473684, random_state=None)), ('randomforestclassifier', RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight='balanced', criterion='gini', max_depth=None, max_features='log2', max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=0.01, min_samples_split=0.01, min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))], verbose=False)), ('19', Pipeline(memory=None, steps=[('truncatedsvdwrapper', TruncatedSVDWrapper(n_components=0.8015789473684211, random_state=None)), ('randomforestclassifier', RandomForestClassifier(bootstrap=False, ccp_alpha=0.0, class_weight='balanced', criterion='entropy', max_depth=None, max_features='sqrt', max_leaf_nodes=None, max_samples=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=0.035789473684210524, min_samples_split=0.15052631578947367, min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1, oob_score=False, random_state=None, verbose=0, warm_start=False))], verbose=False))], flatten_transform=None, weights=[0.26666666666666666, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667, 0.13333333333333333, 0.13333333333333333, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667, 0.06666666666666667]))],
       verbose=False)
  ```

- ***Accuracy***: 0.91

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

Some screenshots of the experiment have been store [here](./img).

*TODO* add how could you have improved it?

## Hyperparameter Tuning
In this task, an `HyperDrive` pipeline using a customized model is built.\
The model chosen was scikit-learn `logistic regression` model.\
Logistic regression is a classification algorithm used in binary classification problems, when the dependent variable (DEATH_EVENT) is categorical. It uses the logistic function to model the probability of a certain class or event (1 or 0).\
For this reason, this model is suitable for our task.

The `train.py` script has been used to train the model, using two hyperparameters tuned with HyperDrive:
- `C`: inverse of regularization strength. Smaller values cause stronger regularization
- `max_iter`: maximum number of iterations for model to converge

Hyperparameter tuning can be computationally expensive, so HyperDrive helps to automate and speeds up hyperparameter tuning process, choosing these parameters. `HyperDriveConfig` class is responsible of the hyperparameters tuning process. It includes information about hyperparameter space sampling, termination policy, primary metric and estimator.

Specify hyperparameter space sampling and termination policy is very important:
- `hyperparameter_sampling`: defines the hyperparameters sampling space. `RandomParameterSampling` randomly select hyperparameters values over the search space. It is not computationally expensive and it is not exhaustive but it works well in most cases.\
`C` and `max_iter` parameters have been passed to the sampler:
```python
# Specify parameter sampler
param_sampling = RandomParameterSampling(
    {
        "--C": uniform(0.1, 1.0),
        "--max_iter": choice(25, 50, 100, 150)
    }
)
```
- `policy`: defines the early termination policy to use. `BanditPolicy` is based on slack criteria, and a frequency and delay interval for evaluation. Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. It automatically terminates poorly performing runs, saving time and improving computational efficiency:
```python
# Specify a Policy
early_termination_policy = BanditPolicy(evaluation_interval=2, slack_factor=0.1)
```

### Results
Once the HyperDrive run has been submitted to the experiment, the best run and the related metrics have been collected:
```python
best_run_hdr = hdr.get_best_run_by_primary_metric()
best_run_metrics_hdr = best_run_hdr.get_metrics()
best_params_hdr = best_run_hdr.get_details()['runDefinition']['arguments']

print('Best run ID: ', best_run_hdr.id)
print('Best run Accuracy: ', best_run_metrics_hdr['Accuracy'])
print('Metrics: ', best_run_metrics_hdr)
```
```python
Best run ID: HD_d4628b90-0e0d-4602-b36f-903f7ea498ec_2
Best run Accuracy: 0.90
Metrics: {'Regularization Strength:': 0.50, 'Max iterations:': 25, 'Accuracy': 0.90}
```

The pipeline has been completed in 10m 20s.

![Hyperdrive Experiment](./img/hyperdrive_experiment.png)

#### Best Model
- ***Parameters***
  - `C` = 0.50
  - `max_iter` = 20
- ***Accuracy***: 0.90

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

Some screenshots of the experiment have been store [here](./img).

*TODO* add how could you have improved it?

## Model Deployment
Comparing the performances (accuracy) of the two approaches, we have obtained:
- `AutoML`: 0.94
- `HyperDrive`: 0.90

AutoML model performs slightly better than HyperDrive one.\
For this reason the AutoML best model has been deployed to *Azure Container Instance*

## Screen Recording
[Link]() to the video

## Future Work
- Try to run HyperDrive with a more exhaustive parameter sampler for hyperparameters tuning, i.e. grid sampling, or try to increase the search space
- Try to validate more metrics, i.e. AUC
- Try to use different customized models, i.e. neural network (non trivial choice with a small dataset)
- Try to work more in depth in feature engineering process, creating and investigating new predictive features
- Show model output using an user friendly dashboard
