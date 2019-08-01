# CRISP-DM on AWS

## Introduction

* CRISP-DM: CRoss Industry Standard Process - Data Mining
* Framework for data science projects
* Created in 1996

The phases

* Business understanding - User story/business objective, frame problem - decide if not ML problem or suitable for ML. If suitable continue with CRISP-DM
* Data understanding - collection, verifying, exploring the data. Identify the data to use, where it is housed, how to access it. Examine the data, looking for patterns and applying standard statistical techniques to understand the properties of the data set. Understand the quality of the dataset and decide if it can be used.
* Data prep - prepare the data set, do feature engineering. 
* Modeling - select and create ML model. Tune the model, if more data or diff features iterate back to the data prep stage.
* Evaluation - what are the outcomes, what are the false positives, false negatives, examine the performance in the context of the business objectives. Decide to deploy or to start the entire cycle again.
* Deployment - planning the deployment, maint and monitoring, final report, project review.

Cyclical endeavor