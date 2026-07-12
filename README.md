# 📰 Fake News Detector
ML model that classifies news articles as REAL or FAKE using NLP (TF-IDF) + classical ML.

## Project Structure
fake-news-detector/
├── data/
├── notebooks/
├── models/
├── src/

## Dataset
6,335 labeled news articles (title, text, label) — balanced FAKE/REAL split.

## Working:
How this pipeline works basically works:-
Data Ingestion -> Transformation -> Test -> Accuracy -> Report Genration -> visualization -> Saving the model

 1. The dataset which is used is first cleaned and after thar transforming the data into a single format for processing.
 2. Then the model is trained and we determine an accuracy score of the data and also create a single visualized report.
 3. And finally the model is saved in the directory 'models/'.


## Requirements
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
jupyter


