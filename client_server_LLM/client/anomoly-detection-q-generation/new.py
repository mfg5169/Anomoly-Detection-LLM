from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Set up Chrome options to control downloads
download_path = os.path.expanduser("~/Downloads")  # Change if needed
chrome_options = webdriver.ChromeOptions()
chrome_options.add_experimental_option("prefs", {
    "download.default_directory": download_path,
    "download.prompt_for_download": False,  
    "download.directory_upgrade": True,
    "safebrowsing.enabled": True
})


# Start WebDriver
driver = webdriver.Chrome(options=chrome_options)
driver.get("https://www.msci.com/end-of-day-data-country?_cookiemanager_WAR_cookiemanager_6AZTm7sLbwpQZrd=accept&_cookiemanager_WAR_cookiemanager_pageUrl=https%3A%2F%2Fwww.msci.com%2Fend-of-day-data-country")

# Locate the `<a>` tag and click it
#download_button = driver.find_element(By.XPATH, "//a[@onclick='CommonUtils.downloadDataCountry()']")
wait = WebDriverWait(driver, 10)  # Wait up to 10 seconds

download_button = wait.until(EC.presence_of_element_located((By.XPATH, "//a[contains(text(), 'United Kindgom')]")))
download_button.click()
download_button = wait.until(EC.presence_of_element_located((By.XPATH, "//a[contains(text(), 'Download Data')]")))
download_button.click()
time.sleep(2)
print("Clicking download button...")
download_button.click()

# Wait for download to complete
time.sleep(5)

# Verify download
downloaded_files = os.listdir(download_path)
print("Downloaded files:", downloaded_files)

# Close browser
driver.quit()
