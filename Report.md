# Capstone Project Report
Stepan Rakhimov

September 17, 2020

## 1. Introduction
### 1.1 Background
Car accidents are still a problem for many nations. Moreover some particular roads or locations tend to have higher probability of accidents and their severity as the others due to various factors.
### 1.2 Problem
Imagine you are driving to another city for work or to visit some friends. It is rainy and windy, and on the way, you come across a terrible traffic jam on the other side of the highway. Long lines of cars barely moving. As you keep driving, police car start appearing from afar shutting down the highway. Oh, it is an accident and there's a helicopter transporting the ones involved in the crash to the nearest hospital. So you start asking yourself if you could take another, probably less dangerous route or even better had something in place that could warn you, given the weather and the road conditions about the possibility of you getting into a car accident and how severe it would be, so that you would drive more carefully or even change your route.

The end goal of this study is to see if we can build a prediction software using Machine Learning from existing historical data.
### 1.3 Interest
As production solution this kind of service might be integrated in to car navigation systems or stand alone application for smartphones. That way it may bring value to people driving in Seattle. Bigger datasets or service partitioning may cover wider areas like whole US or world.
## 2. Data acquisition and cleaning
### 2.1 Data sources
Our data source is open dataset of car accidents in the city of Seattle. It sources from Traffic Management of the city of Seattle and comes as two files: CSV file of actual data and PDF file with metadata.

The dataset covers the period from 2004 till 2020 and contains about 200k records. The table has various accident properties such as severity, weather, road, geo point, number of vehicles and persons involed etc. This data should be enough for us to pick meaninful features and build consistent and robust model that fulfills the goal of the study.

The dataset is not suitable for Machine Learning "as is", but it will be after some cleanup and reformat.

It also has 38 fields per every accident, which might be too much for our purpose since 37 dimentions will slow down model training process but may not add to the outcome. So only meaninful features will be selected from the dataset and used as training an test sets.
### 2.2 Data cleaning
### 2.3 Feature selection
## 3. Exploratory Data Analysis
### 3.1 Calculation of target variable
### 3.9 Relationship between improvement and teams
## 4. Predictive Modeling
### 4.1 Regression models
#### 4.1.1 Applying standard algorithms and their problems
#### 4.1.2 Solution to the problems
#### 4.1.3 Performances of different models
### 4.2 Classification models
## 5. Conclusions
## 6. Future directions