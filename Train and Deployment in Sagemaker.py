#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import boto3
import sagemaker

# Replace with your S3 bucket name
bucket = 'dataset-bucket-project'
prefix = 'recommendation'  # Set your desired prefix

sagemaker_session = sagemaker.Session()

# Specify the path to your training data in S3
s3_path = "s3://{}/{}/training_data/".format(bucket, prefix)

# Create an estimator
from sagemaker import get_execution_role
from sagemaker.sklearn.estimator import SKLearn

role = get_execution_role()
entry_point = 'combined_train.py'  # Replace with the actual script name

# Instantiate SKLearn estimator
sklearn_estimator = SKLearn(
    entry_point=entry_point,
    role=role,
    instance_count=1,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    base_job_name='your-movie-train-job',  # Replace with your desired job name
    metric_definitions=[{'Name': 'rmse', 'Regex': 'RMSE: (.*?);'}],  # Modify as needed
)

# Train the model
sklearn_estimator.fit({'train': s3_path})


# In[ ]:


##deployment
import sagemaker
from sagemaker.sklearn.model import SKLearnModel

role = sagemaker.get_execution_role()

# Deploy the Movie Model
movie_model = SKLearnModel(
    model_data='s3://sagemaker-us-east-2-364844187647/movie-training-job-2023-11-28-19-26-08-767/output/model.tar.gz',
    role=role, 
    entry_point='combined_train.py',
    framework_version='0.23-1'
)

# Specify instance type and initial instance count for deployment
movie_predictor = movie_model.deploy(
    instance_type='ml.m5.large',
    endpoint_name='movie-endpoint',
    initial_instance_count=1 
)

