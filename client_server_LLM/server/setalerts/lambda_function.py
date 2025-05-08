#
# Retrieves and returns all the jobs in the 
# BenfordApp database.
#

import json
import boto3
import os
import datatier
import requests

from configparser import ConfigParser

def lambda_handler(event, context):
  try:
    print("**STARTING**")
    print("**lambda: proj03_jobs**")
    print("event:",event['pathParameters'])

    
    #
    # setup AWS based on config file:
    #
    config_file = 'stockapp-config.ini'
    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = config_file
    
    configur = ConfigParser()
    configur.read(config_file)
    
    #
    # configure for RDS access
    #
    rds_endpoint = configur.get('rds', 'endpoint')
    rds_portnum = int(configur.get('rds', 'port_number'))
    rds_username = configur.get('rds', 'user_name')
    rds_pwd = configur.get('rds', 'user_pwd')
    rds_dbname = configur.get('rds', 'db_name')
    lambda_function_arn = configur.get('lambda', 'arn')

    #
    # open connection to the database:
    #
    print("**Opening connection**")
    
    dbConn = datatier.get_dbConn(rds_endpoint, rds_portnum, rds_username, rds_pwd, rds_dbname)
    authsvc_config_file = 'authsvc_config_file.ini'
    configur.read(authsvc_config_file)
    auth_url = configur.get('client', 'webservice')
    # get the authentication token from the request headers:

    print("event again: " , event)
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
    token = headers['Authentication']

    query_string = event.get("pathParameters", {})
    print("query-string: ", query_string)
    body = event.get("body", {})

    print("**Retrieving authentication user info**")
    print(auth_url )
    res = requests.post(auth_url + "/auth", json={"body" :{'token': token}})
    print(f"status code: {res.status_code}")
    print(res.json())
    res = res.json()
    if res.get('statusCode') != 200:
      msg = "authentication failed"
      print("**ERROR:", msg)


      if res.get('statusCode') == 401:
        return {
            'statusCode': res.get('statusCode'),
              'body': json.dumps(res)
          }
      elif res.get('statusCode') in [400, 500]:
        return {
            'statusCode': 500,
              'body': json.dumps(res)
        }
    userid = res['body']
    # open connection to the database:
    #
    print("**Opening connection**")
    # now retrieve all the jobs:
    #
    print("**Retrieving data**")
    print(body)
    if "frequency" not in event["body"]:
      msg = "no frequency parameters in request"
      print("**ERROR:", msg)
      return {
          'statusCode': 400,
            'body': json.dumps(msg)
        }
    print("Fre")
    # if :
    #   msg = "no freq in request"
    #   print("**ERROR:", msg)
    #   return {
    #       'statusCode': 400,
    #         'body': json.dumps(msg)
    #     }

    # q = event.get("queryStringParameters", {})
    # f = q.get('frequency', "")
    # print(f)




    sql = """SELECT modelid 
                FROM models 
                WHERE model_title = %s 
                AND model_version = %s 
                AND (userid = %s OR userid IS NULL)"""

    userid = int(userid.replace('"', '').strip())
    print(f"userid: {userid}")
    print(query_string['model'])
    rows = datatier.retrieve_all_rows(dbConn, sql, [ query_string['model'], 1 , userid])

    print('recieved rows')
    if rows == None:
      return {
          'statusCode': 400,
            'body': json.dumps("no such model version exists")
        }

    if len(rows) == 1:
      return {
          'statusCode': 400,
            'body': json.dumps("multiple models exist with, try creating a new unique model")
        }
    
    modelid = rows[0]

    sql = "select * from subscriptions where modelid = %s and frequency = %s"
    sql2 = "select * from subscriptions where modelid = %s and frequency = %s and userid = %s"

    rows = datatier.retrieve_all_rows(dbConn, sql, [modelid, frequency])

    if rows == None:
        print("no subscription so create subscription")

        # Initialize EventBridge and Lambda clients
        eventbridge = boto3.client('events')
        lambda_client = boto3.client('lambda')

        data = {
            "modelid": modelid,
        }

        # Create an EventBridge rule (schedule the event to run daily)
        rule_response = eventbridge.put_rule(
            Name='DailyTrigger',  # Rule name
            ScheduleExpression='rate(10 minutes)',  # Cron expression for daily at midnight UTC
            State='ENABLED',  # The rule is enabled immediately after creation
            Description='Triggers the  Lambda function daily at midnight UTC'
        )


        # Add Lambda as a target for the rule with input data
        target_response = eventbridge.put_targets(
          Rule='DailyTrigger',
          Targets=[
              {
                  'Id': '1',  # Unique ID for the target
                  'Arn': lambda_function_arn,  # Lambda ARN
                  'Input': json.dumps(input_data),  # Pass input data as a JSON string
              }
            ]
        )

        # Add permission for EventBridge to invoke the Lambda function
        permission_response = lambda_client.add_permission(
            FunctionName='finalproj_createML',  # Lambda function name
            Principal='events.amazonaws.com',  # Allow EventBridge to invoke the Lambda
            StatementId='MLTrainID',  # Unique statement ID
            Action='lambda:InvokeFunction',  # Action allowed
            SourceArn=rule_response['RuleArn']  # ARN of the EventBridge rule
        )

        # Output the responses (for debugging/confirmation)
        print("EventBridge Rule Created:", rule_response)
        print("Target Added to Rule:", target_response)
        print("Lambda Permission Added:", permission_response)

        sql3 = "insert into subscriptions (modelid, frequency, userid) values (%s, %s, %s)"
        datatier.perform_action(dbConn, sql3, [modelid, frequency, userid])

        sql = "SELECT LAST_INSERT_ID();"

        row = datatier.retrieve_one_row(dbConn, sql, [])
        subscriptionid = row[0]
        sql4 = "insert into notifications (modelid, subscriptionid, NOTI_TIMESTAMP) values (%s, %s, %s)"
        datatier.perform_action(dbConn, sql3, [modelid, subscriptionid, datetime.datetime.now() ])

    else:
        rows2 = datatier.retrieve_all_rows(dbConn, sql2, [modelid, frequency, userid])



        if rows2 == None:
            sql3 = "insert into subscriptions (model, frequency, userid) values (%s, %s, %s)"
            datatier.perform_action(dbConn, sql3, [modelid, frequency, userid])
            return {
                'statusCode': 200,
                'body': json.dumps("subscription created")
            }
        else:
            return {
                'statusCode': 400,
                'body': json.dumps("you have already subscribed to this model")
            }
    #
    # respond in an HTTP-like way, i.e. with a status
    # code and body in JSON format:
    #
    print("**DONE, returning rows**")
    
    return {
      'statusCode': 200,
      'body': json.dumps(rows)
    }
    
  except Exception as err:
    print("**ERROR**")
    print(str(err))
    
    return {
      'statusCode': 500,
      'body': json.dumps(str(err))
    }
