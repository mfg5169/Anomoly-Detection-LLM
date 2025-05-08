#
# Uploads a PDF to S3 and then inserts a new job record
# in the BenfordApp database with a status of 'uploaded'.
# Sends the job id back to the client.
#


#add to the database the job
#add to the queue the job to trigger event

import json
import boto3
import os
import uuid
import pathlib
import datatier
import requests


from configparser import ConfigParser

def lambda_handler(event, context):
  try:
    print("**STARTING**")
    print("**lambda: proj03_upload**")
    
    #
    # setup AWS based on config file:
    #
    config_file = 'queue_config.ini'
    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = config_file
    
    configur = ConfigParser()
    configur.read(config_file)
    
    #
    # configure for S3 access:
    #
    print("**Accessing SQS PROF**")
    sqs_profile = 'sqsreadwrite'
    boto3.setup_default_session(profile_name=sqs_profile)
    
    
    queue_url = configur.get('sqs-queue', 'url')
    


    print("Getting Stock DB config")
    stock_config_file = 'stockapp-config.ini'
    stock_configur = ConfigParser()
    stock_configur.read(stock_config_file)

    print(stock_configur)
    
    rds_endpoint = stock_configur.get('rds', 'endpoint')
    rds_portnum = int(stock_configur.get('rds', 'port_number'))


    #auth login
    rds_auth_username = stock_configur.get('auth-role', 'user_name')
    rds_auth_pwd = stock_configur.get('auth-role', 'user_pwd')
    rds_auth_dbname = stock_configur.get('auth-role', 'db_name')
    

    #stock login
    rds_stock_username = stock_configur.get('stock-role', 'user_name')
    rds_stock_pwd = stock_configur.get('stock-role', 'user_pwd')
    rds_stock_dbname = stock_configur.get('stock-role', 'db_name')

    stock_db = datatier.get_dbConn(rds_endpoint, rds_portnum, rds_stock_username, rds_stock_pwd, rds_stock_dbname)
    # userid from event: could be a parameter
    # or could be part of URL path ("pathParameters"):
    #
    print("**Accessing event/pathParameters**")
    

    print("**Accessing request headers to get authenticated user info**")
    if "headers" not in event:
      msg = "no headers in request"
      print("**ERROR:", msg)
      return { 'statusCode': 400,
          'body': json.dumps(msg)
        }
    headers = event["headers"]

    if "Authentication" not in headers:
      msg = "no security credentials in request "
      print("**ERROR:", msg)
      return {
          'statusCode': 400,
            'body': json.dumps(msg)
        }


 

    
  
    #
    # the user has sent us two parameters:
    #  1. filename of their file
    #  2. raw file data in base64 encoded string
    #
    # The parameters are coming through web server 
    # (or API Gateway) in the body of the request
    # in JSON format.
    #
    print("**Accessing request body**")
    
    print(event)
    auth_config_file = 'auth_file.ini'

    auth_configur = ConfigParser()
    auth_configur.read(auth_config_file)
    auth_url = auth_configur.get('client', 'webservice')

    print("after this")

    try:
        body = json.loads(event.get("body", "{}"))
    except json.JSONDecodeError:
        print("Error: Failed to parse JSON body.")
        return {"statusCode": 400, "body": "Invalid JSON format"}

    tolerance = body.get("tolerance", 0.5)
    term = body.get("term", "short") 

    print("Validation Token....")
    token = event['headers']['Authentication']
    res = requests.post(auth_url + "/auth", json={'body': {'token': token}})
    print(f"status code: {res.status_code}")
    res = res.json()
    if res['statusCode'] != 200:
      msg = "authentication failed"
      print("**ERROR:", msg)
      
      print(res['statusCode'])
      if res['statusCode'] == 401:
        return {
            'statusCode': res['statusCode'],
              'body': json.dumps(res)
          }
      elif res['statusCode'] in [400, 500]:
        return {
            'statusCode': 500,
              'body': json.dumps(res)
        }
      else:
        return {
            'statusCode': 400,
              'body': json.dumps(res)
        }
    

    userid = res['body']

    #
    # open connection to the database:
    #

    if userid == None:
      return {
        'statusCode': 400,
        'body': json.dumps("no such user...")
      }

    print("USER ID: ", userid)
    print("**Opening connection**")
    


    #
    # first we need to make sure the userid is valid:
    #
    print("**Checking if userid is valid**")
    

    #Initialize the SQS client
    sqs_client = boto3.client('sqs', region_name='us-east-2')

    print("SQS Client initialized")
    data = body.get("data", {})
    name = body.get("name", {})
    # Send a message to the queue

    print("Sending message to SQS queue")
    response = sqs_client.send_message(
        
        QueueUrl=queue_url,
        MessageBody= json.dumps({
            'userid': userid,
            'name': name,
            'tolerance': tolerance,
            'term': term,
            'train-data': data
        }),
        MessageGroupId='Group1',
        MessageDeduplicationId=str(uuid.uuid4())
    )

    # Print the message ID
    print(f"Message sent with ID: {response['MessageId']}")
    print(userid)
    sql = """
      INSERT INTO jobs(userid, status, model_title, private)
                  VALUES(%s, %s , %s , %s);
    """
    
    #
    # TODO #2 of 3: what values should we insert into the database?
    #

    userid = int(userid.replace('"', '').strip())
    datatier.perform_action(stock_db, sql, [userid, 'not started', name, True])

    #
    # grab the jobid that was auto-generated by mysql:
    #
    sql = "SELECT LAST_INSERT_ID();"
    
    row = datatier.retrieve_one_row(stock_db, sql)
    
    jobid = row[0]
    
    print("jobid:", jobid)

    print("**DONE, returning jobid**")
    
    return {
      'statusCode': 200,
      'body': json.dumps(str(jobid))
    }
    
  except Exception as err:
    print("**ERROR**")
    print(str(err))
    
    return {
      'statusCode': 500,
      'body': json.dumps(str(err))
    }
