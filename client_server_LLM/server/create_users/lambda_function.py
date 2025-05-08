#
# POST /auth
#
# Lambda function to handle authentication. The caller can 
# post a token, or a username/password, and the function 
# authenticates the token or username. Valid message formats:
#
# { "body": { "token": "..." } }
# { "body": { "username": "...", "password": "..." } }
#
# If a token is passed, the function returns 200 --- and the
# userid --- if the token is still valid , and 401 if not.
# If a username/password is passed, the function returns 
# 200 --- and a token --- if the username exists and the 
# password matches. Otherwise 401 is returned.
#
# If the function is called incorrectly, a status code of 400
# is returned, and the body of the message contains an error
# message. Server-side programming errors are returned with a
# status code of 500, with the body containing the error msg.
#
# When passing username/password, an optional "duration" can
# be posted, which is the duration in minutes for the token
# before it expires --- passing small values like 1 or 2 is
# good for testing. The default is 30 minutes, and at most
# the caller can set the duration to 60 minutes. Values < 1
# or > 60 are ignored.
#
# Original author: Dilan Nair
# Modifed by: Prof. Joe Hummel
# Northwestern University
#

import json
import os
import datetime
import uuid
import datatier
import auth
import api_utils
import re

from configparser import ConfigParser

def lambda_handler(event, context):
  try:
    print("**STARTING**")
    print("**lambda: proj04_auth**")

    #
    # setup AWS based on config file
    #
    config_file = 'db_config.ini'
    os.environ['AWS_SHARED_CREDENTIALS_FILE'] = config_file
    
    configur = ConfigParser()
    configur.read(config_file)
    
    #
    # configure for RDS access
    #
    rds_endpoint = configur.get('rds', 'endpoint')
    rds_portnum = int(configur.get('rds', 'port_number'))

    #auth login
    rds_auth_username = configur.get('auth-role', 'user_name')
    rds_auth_pwd = configur.get('auth-role', 'user_pwd')
    rds_auth_dbname = configur.get('auth-role', 'db_name')
    

    #stock login
    rds_stock_username = configur.get('stock-role', 'user_name')
    rds_stock_pwd = configur.get('stock-role', 'user_pwd')
    rds_stock_dbname = configur.get('stock-role', 'db_name')
    #
    # open connection to the database
    #
    print("**Opening connection**")
    
    auth_dbConn = datatier.get_dbConn(rds_endpoint, rds_portnum, rds_auth_username, rds_auth_pwd, rds_auth_dbname)
    stock_dbConn = datatier.get_dbConn(rds_endpoint, rds_portnum, rds_stock_username, rds_stock_pwd, rds_stock_dbname)

    #
    # We are expecting either a token, or username/password:
    #
    print("**Accessing request body**")
    

    username = ""
    email = ""
    password = ""

    print("event:", event)

    if "body" not in event:
      return api_utils.error(400, "no body in request")
    
    print("Check body: ")
    print("event[body]:", event["body"])

    body = (event["body"])
    print("body:", body)

    x = "username" in body
    y = "password" in body
    z = "email" in body
    print("Check body: ", x)
    print("Check body: ", y)
    print("Check body: ", z)
    if "name" in body and "password" in body and "email" in body:
      print("All three are in")

      username = body["name"]
      email = body["email"]


      # Click on Edit and place your email ID to validate

      # if valid == None:
      #   return api_utils.error(400, "invalid email address")

      password = body["password"]
    else:
      print()
      return api_utils.error(400, "missing credentials in body")
      
    #
    # if we were passed a token, lookup in the database and
    # see if exists, and still valid:
    #

    #
    # we were passed username/password, authenticate
    # and return token if so:
    #
    print("**We were passed username/password**")
    print("username:", username)
    print("email:", email)
    print("password:", password)
    
    #
    # did they pass a duration for the token? It's optional:
    #

    

      
    print("**Looking up user**")
    
    row = ()
    userid = 123
    pwdhash = "apple"
    
 
    #
    # TODO #3 of 5:
    #
    # lookup the username in the database, retrieve userid and pwdhash:
    #

    sql_one = """   SELECT EXISTS ( SELECT 1 FROM users  
              WHERE username = %s) AS user_exists;"""


    
    row = datatier.retrieve_one_row(auth_dbConn,sql_one, [username])

    if row[0] != 0:

      return api_utils.error(401, "invalid username : A user already has your username (check users to see what has been used)")



    sql_two = """   SELECT EXISTS ( SELECT 1 FROM users  
              WHERE email = %s) AS user_exists;"""


    
    row = datatier.retrieve_one_row(stock_dbConn,sql_two, [email])

    if row[0] != 0:

      return api_utils.error(401, "invalid email : A user already has used this email (check users to see what has been used)")

    pwdhash = auth.hash_password(password)

    sql_job = "insert into users(username, email, pwdhash) values(%s, %s, %s)" 
    sql_job2 = "insert into users(username, pwdhash) values(%s, %s)" 

    row_stock = datatier.perform_action(stock_dbConn, sql_job, [username, email, pwdhash])
    row_auth = datatier.perform_action(auth_dbConn, sql_job2, [username, pwdhash])


    sql = "SELECT LAST_INSERT_ID();"

    row = datatier.retrieve_one_row(auth_dbConn, sql, [])

    print("row:", row)
    return api_utils.success(200, row)
    
  except Exception as err:
    print("**ERROR**")
    print(str(err))

    return api_utils.error(500, str(err))
