from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import os

# Create Chrome options
chrome_options = Options()

# Generate a unique temporary directory for user data
user_data_dir = f"/tmp/chrome_user_data_{os.getpid()}" 
chrome_options.add_argument(f"--user-data-dir={user_data_dir}")  
#we
driver = webdriver.Chrome(options=chrome_options)  # Or use webdriver.Firefox()
driver.get("https://www.msci.com/end-of-day-data-country?_cookiemanager_WAR_cookiemanager_6AZTm7sLbwpQZrd=accept&_cookiemanager_WAR_cookiemanager_pageUrl=https%3A%2F%2Fwww.msci.com%2Fend-of-day-data-country")

# Find the button by its text, ID, or class and click it
button = driver.find_element(By.XPATH, "//a[@onclick='CommonUtils.downloadDataCountry()']")

button.click()

# Wait for the download to complete (optional)
import time
time.sleep(5)

driver.quit()

# Clean up the temporary user data directory after use
import shutil
shutil.rmtree(user_data_dir, ignore_errors=True)