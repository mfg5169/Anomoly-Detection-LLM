import json
import boto3
import uuid
from transformers import pipeline
import helpers
sagemaker_client = boto3.client('sagemaker')
s3_client = boto3.client('s3')

company_tickers = {
    "Google": "GOOGL",
    "Apple": "AAPL",
    "Amazon": "AMZN",
    "Tesla": "TSLA"
}

def lambda_handler(event, context):

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


    company_labels = [
        "Google",
        "Apple",
        "Amazon",
        "Tesla"
    ]

    question = event['body']['companyname']

    result = classifier(question, candidate_labels=company_labels)
    most_likely_label = result['labels'][0]

    if "MessageBody" in event['body']:
        X,y, data = helpers.preprocess_data(company_tickers[most_likely_label])
        
    uploaded_data = event['body']
    file_name = f"data/{uuid.uuid4()}.csv"
    bucket_name = "<your-s3-bucket-name>"
    
    
    s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=uploaded_data)
    
    training_job_name = f"training-job-{uuid.uuid4()}"
    input_data_s3_uri = f"s3://{bucket_name}/{file_name}"
    
    training_params = {
        'TrainingJobName': training_job_name,
        'AlgorithmSpecification': {
            'TrainingImage': '<your-custom-training-image>',  # Or built-in algorithm
            'TrainingInputMode': 'File'
        },
        'InputDataConfig': [{
            'ChannelName': 'train',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': input_data_s3_uri,
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        }],
        'OutputDataConfig': {
            'S3OutputPath': f"s3://{bucket_name}/output/"
        },
        'ResourceConfig': {
            'InstanceType': 'ml.m5.large',
            'InstanceCount': 1,
            'VolumeSizeInGB': 50
        },
        'RoleArn': '<your-sagemaker-role-arn>',
    }
    
    sagemaker_client.create_training_job(**training_params)
    
    return {
        'statusCode': 200,
        'body': json.dumps("success")
    }
