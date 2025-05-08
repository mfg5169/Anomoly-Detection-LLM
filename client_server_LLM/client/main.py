#
# Client-side python app for benford app, which is calling
# a set of lambda functions in AWS through API Gateway.
# The overall purpose of the app is to process a PDF and
# see if the numeric values in the PDF adhere to Benford's
# law. This version adds authentication with user names, 
# passwords, and login tokens.
#
# Authors:
#   << YOUR NAME >>
#
#   Prof. Joe Hummel (initial template)
#   Northwestern University
#   CS 310
#

import requests
import jsons
import pandas
import uuid
import pathlib
import logging
import sys
import os
import base64
import time

from configparser import ConfigParser
from getpass import getpass


############################################################
#
# classes
#
class User:

  def __init__(self, row):
    self.userid = row[0]
    self.username = row[1]
    self.pwdhash = row[2]


class Job:

  def __init__(self, row):
    self.jobid = row[0]
    self.userid = row[1]
    self.status = row[2]
    self.originaldatafile = row[3]
    self.datafilekey = row[4]
    self.resultsfilekey = row[5]


###################################################################
#
# web_service_get
#
# When calling servers on a network, calls can randomly fail. 
# The better approach is to repeat at least N times (typically 
# N=3), and then give up after N tries.
#
def web_service_get(url):
  """
  Submits a GET request to a web service at most 3 times, since 
  web services can fail to respond e.g. to heavy user or internet 
  traffic. If the web service responds with status code 200, 400 
  or 500, we consider this a valid response and return the response.
  Otherwise we try again, at most 3 times. After 3 attempts the 
  function returns with the last response.
  
  Parameters
  ----------
  url: url for calling the web service
  
  Returns
  -------
  response received from web service
  """

  try:
    retries = 0
    
    while True:
      response = requests.get(url)
        
      if response.status_code in [200, 400, 480, 481, 482, 500]:
        #
        # we consider this a successful call and response
        #
        break;

      #
      # failed, try again?
      #
      retries = retries + 1
      if retries < 3:
        # try at most 3 times
        time.sleep(retries)
        continue
          
      #
      # if get here, we tried 3 times, we give up:
      #
      break

    return response

  except Exception as e:
    print("**ERROR**")
    logging.error("web_service_get() failed:")
    logging.error("url: " + url)
    logging.error(e)
    return None
    

############################################################
#
# prompt
#
def prompt():
  """
  Prompts the user and returns the command number

  Parameters
  ----------
  None

  Returns
  -------
  Command number entered by user (0, 1, 2, ...)
  """
  try:
      print()
      print(">> Enter a command:")
      print("   0 => end")
      print("   1 => users")
      print("   2 => jobs")
      print("   3 => get stock prediction")
      print("   4 => upload stock data")
      print("   5 => create account")
      print("   6 => upload stock data and poll")
      print("   7 => login")
      print("   8 => authenticate token")
      print("   9 => logout")
      print("   10 => check stock")
      print("   11 => setup alerts")

      cmd = input()

      if cmd == "":
        cmd = -1
      elif not cmd.isnumeric():
        cmd = -1
      else:
        cmd = int(cmd)

      return cmd

  except Exception as e:
      print("**ERROR")
      print("**ERROR: invalid input")
      print("**ERROR")
      return -1


############################################################
#
# users
#
def users(baseurl, token):
  """
  Prints out all the users in the database

  Parameters
  ----------
  baseurl: baseurl for web service

  Returns
  -------
  nothing
  """

  try:
    #
    # call the web service:
    #
    api = '/users'
    url = baseurl + api

    # res = requests.get(url)
    res = web_service_get(url)

    #
    # let's look at what we got back:
    #
    if res.status_code == 200: #success
      pass
    else:
      # failed:
      print("**ERROR: failed with status code:", res.status_code)
      print("url: " + url)
      if res.status_code == 500:
        # we'll have an error message
        body = res.json()
        print("Error message:", body)
      #
      return

    #
    # deserialize and extract users:
    #
    body = res.json()

    #
    # let's map each row into a User object:
    #
    users = []
    for row in body:
      user = User(row)
      users.append(user)
    #
    # Now we can think OOP:
    #
    if len(users) == 0:
      print("no users...")
      return

    for user in users:
      print(user.userid)
      print(" ", user.username)
      print(" ", user.pwdhash)
    #
    return

  except Exception as e:
    logging.error("**ERROR: users() failed:")
    logging.error("url: " + url)
    logging.error(e)
    return


############################################################
#
# jobs
#
def jobs(baseurl, token):
  """
  Prints out all the jobs in the database

  Parameters
  ----------
  baseurl: baseurl for web service

  Returns
  -------
  nothing
  """

  try:
    #
    # call the web service:
    #
    api = '/jobs'
    url = baseurl + api

    #
    # make request:
    #
    # res = requests.get(url)
 
    print("Starting jobs()....")
    print("foken: ", token)

    headers = {'Authentication': token}
    res = requests.get(url, headers=headers)

    #
    # let's look at what we got back:
    #
    print("status code:", res.status_code)
    if res.status_code == 200: #success
      pass
    else:
      # failed:
      print("**ERROR: failed with status code:", res.status_code)
      print("url: " + url)
      if res.status_code == 401:
        # we'll have an error message
        body = res.json()
        print("Error message:", body)
      elif res.status_code == 500:
        # we'll have an error message
        body = res.json()
        print("Error message:", body)
      #
      return

    #
    # deserialize and extract jobs:
    #

    print("about to obtain body....")
    body = res.json()
    #
    # let's map each row into an Job object:
    #
    jobs = []
    for row in body:
      job = Job(row)
      jobs.append(job)
    #
    # Now we can think OOP:
    #
    if len(jobs) == 0:
      print("no jobs...")
      return

    for job in jobs:
      print(job.jobid)
      print(" ", job.userid)
      print(" ", job.status)
      print(" ", job.originaldatafile)
      print(" ", job.datafilekey)
      print(" ", job.resultsfilekey)
    #
    return

  except Exception as e:
    logging.error("**ERROR: jobs() failed:")
    logging.error("url: " + url)
    logging.error(e)
    return


############################################################
#
# predict

def predict(baseurl, token):
  try:

    print("What stock would you like to predict?>")
    stock_name = input()
    print("Are you focused on short-term or long-term prediction?>(Enter short or long)")
    prediction_type = input()
    print("What is your risk tolerance?(number between 0 and 1, with 1 as the highest)>")
    risk_tolerance = input()

    authorization = {'Authentication': token}

    res = requests.get(baseurl + '/predictions/' + stock_name , params={"term": prediction_type, "tolerance": risk_tolerance}, headers=authorization)
    print("status code:", res)
    if res.status_code == 200:
      print("Success!")
      print(res.json())
      pass
    elif res.status_code == 400:
      print("Bad request")
      print(res.json())
      return
    elif res.status_code == 401:
      print("Unauthorized")
      print(res.json())
      return
    else:
      print("500 Server Error")
      return
    body = res.json()
    print("Job id for this prediction is:" , body)
    return
  except Exception as e:
    logging.error("**ERROR: predict() failed:")
    logging.error("url: " + baseurl)
    logging.error(e)


############################################################
#
# create_account
#
def create_account(baseurl):
  try:
      print("Enter Name>")
      name = input()

      print("Enter email>")
      email = input()

      print("create password>")
      password = getpass()




      
      #
      # build the data packet:
      #
  

      #
      # call the web service:
      #
      api = '/new-account'
      url = baseurl + api 

      data = {'body': {"name": name, "email": email, "password": password}}
      res = requests.post(url, json=data)

      #
      # let's look at what we got back:
      #
    
      if res.status_code == 200: #success
        pass
      elif res.status_code == 400: # no such user
        body = res.json()
        print(body)
        return
      else:
        # failed:
        print("**ERROR: failed with status code:", res.status_code)
        print("url: " + url)
        if res.status_code == 500:
          # we'll have an error message
          body = res.json()
          print("Error message:", body)
        #
        return

      #
      # success, extract jobid:
      #
      print(res)
      body = res.json()

      print("BODY: " , body)

      print("Congrats you've created an account your userid is =", body)
      return

  except Exception as e:
      logging.error("**ERROR: create_account failed:")
      logging.error("url: " + url)
      logging.error(e)
      return
############################################################
#
# upload
#
def upload(baseurl,token):
  """
  Prompts the user for a local filename and user id, 
  and uploads that asset (PDF) to S3 for processing. 

  Parameters
  ----------
  baseurl: baseurl for web service

  Returns
  -------
  nothing
  """
  try:
   

    print("For best performance the first column should be the date/time and the second column should be the stock price with the top being the most recent date/time\n")
    print('-'*80)
    print("Enter excel or csv filename(File must have the first column as the date/time and the second column as the stock price)>")
    
    local_filename = input()


    tp = local_filename.split(".")
    if tp[-1] not in ["csv", "xlsx"]:
      print("File must be a csv or xlsx file")

    df = pandas.read_csv(local_filename) if tp[-1] == "csv" else pandas.read_excel(local_filename)
    if len(df.columns) != 2:
      print("**ERROR: File must have two columns, the first column as the date/time and the second column as the stock price, top being the most recent date/time")
      return


    
    print("Data preview:")
    print(df.head())

    df_cleaned = df.dropna(how="all")
    print("Data preview after cleaning:")
    print(df_cleaned.head())
    dct = df_cleaned.to_dict(orient='records')
    print("Give Model a name>")
    name = input()
    tolerance = input("Enter the tolerance level for the model (number between 0 and 1, with 1 as the highest)>")
    term = input("Enter the term for the model (short or long)>")

    try:
      dp = jsons.dumps(dct)
    except Exception as e:
      print("**ERROR: failed to serialize data")
      print(e)
      return
    data = {"name": name, "data": jsons.dumps(dct), "tolerance": tolerance, "term": term}
    headers = {'Authentication': token}


    print("Uploading data...")
    res = requests.post(baseurl + '/train', json=data, headers=headers)
    print("status code:", res)

    #
    # let's look at what we got back:
    #
    if res.status_code == 200: #success
      pass
    elif res.status_code == 400: # no such user
      body = res.json()
      print(body)
      return
    elif res.status_code == 401:
      print("Unauthorized")
      print(res.json())
      return
    else:
      # failed:
      print("**ERROR: failed with status code:", res.status_code)
      print("url: " + baseurl)
      if res.status_code == 500:
        # we'll have an error message
        body = res.json()
        print("Error message:", body)
      #
      return

    #
    # success, extract jobid:
    #
    body = res.json()

    jobid = body

    print("uploaded train data, job id =", jobid)
    return

  except Exception as e:
    logging.error("**ERROR: upload() failed:")
    logging.error("url: " + baseurl)
    logging.error(e)
    return




############################################################
#
# upload_and_poll
#
# def upload_and_poll(baseurl, token):
  

#   try:

#     while True:
#       res = requests.get(url)
#       if res.status_code == 200: #success

#         break
    
#       elif res.status_code == 400: # no such job
#         body = res.json()
#         print(body)
#         return
      
#       elif res.status_code in [480, 481, 482]:  # uploaded
#         msg = res.json()
#         print("No results available yet...")
#         print("Job status:", msg)
        
#       else:
#         # failed:
#         print("Failed with status code:", res.status_code)
#         print("url: " + url)
#         if res.status_code == 500:
#           # we'll have an error message
#           body = res.json()
#           print("Error message:", body)
#         #
#         return
#       time.sleep(random.randint(0, 5) )
      
#     #
#     # if we get here, status code was 200, so we
#     # have results to deserialize and display:
#     #
    

#     body = res.json()
#     # deserialize the message body:


#     datastr = body

#     #
#     # encode the data string to obtain the raw bytes in base64,
#     # then call b64decode to obtain the original raw bytes.
#     # Finally, decode() the bytes to obtain the results as a 
#     # printable string.
#     #
    
#     if not datastr:
#         print("No data received.")
#         return

#       # Decode base64 data
#     base64_bytes = datastr.encode()
#     bytes_data = base64.b64decode(base64_bytes)
      
#       # Save to file
#     #output_filename = f"results_{jobid}.txt"


#     #with open(output_filename, "wb") as outfile:
#      #   outfile.write(bytes_data)


#     print("\n\n" + bytes_data.decode(errors="ignore"))

#     return

############################################################
#
# login
#
def login(auth_url):
  """
  Prompts the user for a username and password, then tries
  to log them in. If successful, returns the token returned
  by the authentication service.

  Parameters
  ----------
  auth_url: url for auth web service

  Returns
  -------
  token if successful, None if not
  """
  try:
    username = input("username: ")
    password = getpass()
    duration = input("# of minutes before expiration? [ENTER for default] ")

    #
    # build message:
    #
    if duration == "":  # use default
      data = {"name": username, "password": password}
    else:
      data = {"name": username, "password": password, "duration": duration}

    #
    # call the web service to upload the PDF:
    #
    api = '/auth'
    url = auth_url + api

    res = requests.post(url, json={"body" : data})

    #
    # clear password variable:
    #
    password = None

    #
    # let's look at what we got back:
    #
    if res.status_code == 401:
      #
      # authentication failed:
      #
      body = res.json()
      print(body)
      return None

    if res.status_code == 200: #success
      pass
    elif res.status_code in [400, 500]:
      # we'll have an error message
      body = res.json()
      print("**Error:", body)
      return
    else:
      # failed:
      print("**ERROR: Failed with status code:", res.status_code)
      print("url: " + url)
      return

    #
    # success, extract token:
    #
    body = res.json()

    token = body

    print("logged in, token:", token)
    print(jsons.loads(token['body']))
    return  jsons.loads(token['body'])

  except Exception as e:
    logging.error("**ERROR: login() failed:")
    logging.error("url: " + url)
    logging.error(e)
    return None


############################################################
#
# authenticate
#
def authenticate(auth_url, token):
  """
  Since tokens expire, this function authenticates the 
  current token to see if still valid. Outputs the result.

  Parameters
  ----------
  auth_url: url for auth web service

  Returns
  -------
  nothing
  """
  try:
    if token is None:
      print("No current token, please login")
      return

    print("token:", token)

    #
    # build message:
    #
    data = {"token": token}

    #
    # call the web service to upload the PDF:
    #
    api = '/auth'
    url = auth_url + api

    res = requests.post(url, json=data)

    #
    # let's look at what we got back:
    #
    if res.status_code == 401:
      #
      # authentication failed:
      #
      body = res.json()
      print(body)
      return

    if res.status_code == 200: #success
      pass
    elif res.status_code in [400, 500]:
      # we'll have an error message
      body = res.json()
      print("**Error:", body)
      return
    else:
      # failed:
      print("**ERROR: Failed with status code:", res.status_code)
      print("url: " + url)
      return

    #
    # success, token is valid:
    #
    print("token is valid!")
    return

  except Exception as e:
    logging.error("**ERROR: authenticate() failed:")
    logging.error("url: " + url)
    logging.error(e)
    return


############################################################
#
# setup_alerts
#
def setup_alerts(auth_url, token):
  """
  Sets up alerts for a given user. Alerts are sent to the
  user's email address.

  Parameters
  ----------
  auth_url: url for auth web service

  Returns
  -------
  nothing
  """
  try:
    print("Enter model name>")
    model = input()



    print("Enter frequency of alerts>")
    frequency = input()

    #
    # build message:
    #
    data = { "frequency": frequency}

    #
    # call the web service to upload the PDF:
    #
    api = '/alerts/'

    url = auth_url + api + model
    print("url: ", url)

    headers = {'Authentication': token}

    res = requests.put(url, json=data, headers=headers)

    #
    # let's look at what we got back:
    #
    if res.status_code == 200: #success
      pass
    elif res.status_code in [400, 500]:
      # we'll have an error message
      body = res.json()
      print("**Error:", body)
      return
    else:
      # failed:
      print("**ERROR: Failed with status code:", res.status_code)
      print("url: " + url)
      return

    #
    # success, extract token:
    #
    body = res.json()

    msg = body

    print("alerts set up, msg:", msg)
    return

  except Exception as e:
    logging.error("**ERROR: setup_alerts() failed:")
    logging.error("url: " + url)
    logging.error(e)
    return

############################################################
#
#  check_stock
#
def check_stocks(baseUrl, token):
  """
  Checks the stock price for a given symbol.

  Parameters
  ----------
  auth_url: url for auth web service

  Returns
  -------
  nothing
  """
  try:
    # print("Enter stock symbol>")
    # symbol = input()

    #
    # build message:
    #
    # data = {"symbol": symbol}

    #
    # call the web service to upload the PDF:
    #
    api = '/alerts'
    url = baseUrl + api
    print("token: ", token)
    headers = {'Authentication': token}
    res = requests.get(url, headers=headers)

    #
    # let's look at what we got back:
    #
    if res.status_code == 200: #success
      pass
    elif res.status_code in [400, 500]:
      # we'll have an error message
      body = res.json()
      print("**Error:", body)
      return
    else:
      # failed:
      print("**ERROR: Failed with status code:", res.status_code)
      print("url: " + url)
      return

    #
    # success, extract token:
    #
    body = res.json()

    msg = body
    if msg == []:
      print("no stock messages")
    else:
      print("stock messages:", msg)
    return

  except Exception as e:
    logging.error("**ERROR: check_stock() failed:")
    logging.error("url: " + url)
    logging.error(e)
    return

############################################################
#
# check_url
#
def check_url(baseurl):
  """
  Performs some checks on the given url, which is read from a config file.
  Returns updated url if it needs to be modified.

  Parameters
  ----------
  baseurl: url for a web service

  Returns
  -------
  same url or an updated version if it contains an error
  """

  #
  # make sure baseurl does not end with /, if so remove:
  #
  if len(baseurl) < 16:
    print("**ERROR: baseurl '", baseurl, "' is not nearly long enough...")
    sys.exit(0)

  if baseurl == "https://YOUR_GATEWAY_API.amazonaws.com":
    print("**ERROR: update config file with your gateway endpoint")
    sys.exit(0)

  if baseurl.startswith("http:"):
    print("**ERROR: your URL starts with 'http', it should start with 'https'")
    sys.exit(0)

  lastchar = baseurl[len(baseurl) - 1]
  if lastchar == "/":
    baseurl = baseurl[:-1]
    
  return baseurl
  

############################################################
# main
#
try:
  print('** Welcome to BenfordApp with Authentication **')
  print()

  # eliminate traceback so we just get error message:
  sys.tracebacklimit = 0

  #
  # we have two config files:
  # 
  #    1. benfordapp API endpoint
  #    2. authentication service API endpoint
  #
  #
  benfordapp_config_file = 'benfordapp-client-config.ini'
  authsvc_config_file = 'authsvc-client-config.ini'

  print("First, enter name of BenfordApp config file to use...")
  print("Press ENTER to use default, or")
  print("enter config file name>")
  s = input()

  if s == "":  # use default
    pass  # already set
  else:
    benfordapp_config_file = s

  #
  # does config file exist?
  #
  if not pathlib.Path(benfordapp_config_file).is_file():
    print("**ERROR: benfordapp config file '", benfordapp_config_file, "' does not exist, exiting")
    sys.exit(0)

  #
  # setup base URL to web service:
  #
  configur = ConfigParser()
  configur.read(benfordapp_config_file)
  baseurl = configur.get('client', 'webservice')
  
  baseurl = check_url(baseurl)
  
  #
  # now we need to process the 2nd config file:
  #
  print("Second, enter name of Auth Service config file to use...")
  print("Press ENTER to use default, or")
  print("enter config file name>")
  s = input()

  if s == "":  # use default
    pass  # already set
  else:
    authsvc_config_file = s

  #
  # does config file exist?
  #
  if not pathlib.Path(authsvc_config_file).is_file():
    print("**ERROR: authsvc config file '", authsvc_config_file, "' does not exist, exiting")
    sys.exit(0)

  #
  # setup base URL to web service:
  #
  configur.read(authsvc_config_file)
  auth_url = configur.get('client', 'webservice')
  
  auth_url = check_url(auth_url)

  #
  # initialize login token:
  #
  token = None

  #
  # main processing loop:
  #
  cmd = prompt()

  while cmd != 0:
    #
    if cmd == 1:
      users(baseurl, token)
    elif cmd == 2:
      jobs(baseurl, token)
    elif cmd == 3:
      predict(baseurl, token)
    elif cmd == 4:
      upload(baseurl,token)
    elif cmd == 5:
      create_account(auth_url)
    elif cmd == 6:
      upload_and_poll(baseurl, token)
    elif cmd == 7:
      token = login(auth_url)
    elif cmd == 8:
      authenticate(auth_url, token)
    elif cmd == 9:
      #
      # logout
      #
      token = None
    elif cmd == 10:
      check_stocks(baseurl, token)
    elif cmd == 11:
      setup_alerts(baseurl, token)
    else:
      print("** Unknown command, try again...")
    #
    cmd = prompt()

  #
  # done
  #
  print()
  print('** done **')
  sys.exit(0)

except Exception as e:
  logging.error("**ERROR: main() failed:")
  logging.error(e)
  sys.exit(0)
