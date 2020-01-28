---
layout: post
title: Examination of ML diabetes prediction from diet and demographic data.
subtitle: Live, predictive model, and an interactive look at making decisions based on model probabilities.
image: /img/diabetic_graph.png
---

Dash app, hosted by Heroku, with API endpoint for inference with XGBoost Random Forest Classifier.

Diet and behavioral health survey data from King County Washington was used to train and test the model to predict a likelihood of a diabetes diagnosis.

An interactive Plotly graph combines ROC-AUC, confusion matrix, and prediction probability distribution plots to demonstrate how we can use model probabilities to fine tune our predictions.  

Check it out [here](https://recommend-diabetes-screening.herokuapp.com).