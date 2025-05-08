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
    print("**lambda: getalerts**")
    
    #
    # setup AWS based on config file:
    #
    config_file = 'auth-config.ini'
    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = config_file
    
    configur = ConfigParser()
    configur.read(config_file)
    
    #
    # configure for RDS access
    #
    rds_endpoint = configur.get('rds', 'endpoint')
    rds_portnum = int(configur.get('rds', 'port_number'))


    # auth_username = configur.get('auth-role', 'user_name')
    # auth_pwd = configur.get('auth-role', 'user_pwd')
    # auth_dbname = configur.get('auth-role', 'db_name')

    stock_username = configur.get('stock-role', 'user_name')
    stock_pwd = configur.get('stock-role', 'user_pwd')
    stock_dbname = configur.get('stock-role', 'db_name')
    #
    # open connection to the database:
    #
    print("**Opening connection**")
    
    stockdbConn = datatier.get_dbConn(rds_endpoint, rds_portnum, stock_username, stock_pwd, stock_dbname)
    authsvc_config_file = 'authsvc_config_file.ini'
    configur.read(authsvc_config_file)
    auth_url = configur.get('client', 'webservice')
    # get the authentication token from the request headers:

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

    print("**Retrieving authentication user info**")
    print(auth_url )
    res = requests.post(auth_url + "/auth", json={'body': {'token': token}})
    print(f"status code: {res.status_code}")
    print(res.json())
    if res.status_code != 200:
      msg = "authentication failed"
      print("**ERROR:", msg)
      print(res.json())

      if res.status_code == 401:
        return {
            'statusCode': res.status_code,
              'body': json.dumps(res.json())
          }
      elif res.status_code in [400, 500]:
        return {
            'statusCode': 500,
              'body': json.dumps(res.json())
        }
    # open connection to the database:
    #
    print("**Opening connection**")
    # now retrieve all the jobs:
    #
    print("**Retrieving data**")
    
    sql = """SELECT 
  
        s.frequency, 
        n.NOTI_TIMESTAMP,
        n.message
    FROM 
        SUBSCRIPTIONS s
    JOIN 
        NOTIFICATIONS n ON s.subscriptionid = n.subscriptionid
    WHERE 
        s.userid = %s;"""
    print(res.json())
    res = res.json()
    userid = res['body']
    print(f"userid: {userid}")
    rows = datatier.retrieve_all_rows(stockdbConn, sql, [userid])
    
    for row in rows:
      print(row)

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
