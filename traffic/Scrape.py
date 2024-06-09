from selenium import webdriver
from browsermobproxy import Server
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time

def getUrl(url):
    # Specify paths to BrowserMob Proxy and ChromeDriver
    bmp_path = r"D:\school\TrafficDetection\venv\browsermob-proxy\browsermob-proxy-2.1.4\bin\browsermob-proxy.bat"
    chrome_driver_path = "D:\\school\\TrafficDetection\\venv\\ChromeDriver\\chromedriver.exe"

    # Start BrowserMob Proxy
    server = Server(bmp_path)
    server.start()
    proxy = server.create_proxy()

    # ChromeDriver setup with proxy
    chrome_options = Options()
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument(f"--proxy-server={proxy.proxy}")
        
    service = Service(executable_path=chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    
    # Open the target webpage and handle consent
    driver.get(url)
    driver.maximize_window()

    try:
        # Attempt to click the consent button if it appears
        consent_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//p[contains(text(), 'Consent')]"))
        )
        consent_button.click()
        print("Consent button clicked.")
    except TimeoutException:
        # If the consent button doesn't appear within the timeout, proceed without clicking
        print("Consent button not found, proceeding without clicking.")

    # Attempt to click the play button
    try:
        play_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "svg.poster-icon")))
        play_button.click()
    except Exception as e:
        print("Play button not found or other error:", e)

    # Start capturing network traffic
    proxy.new_har("skylinewebcams", options={'captureHeaders': True, 'captureContent': True})

    # Add a delay for the stream to start and for any dynamic content to load
    time.sleep(10)

    # Retrieve the HAR data and search for the live stream URL
    har_data = proxy.har
    stream_url = None
    for entry in har_data['log']['entries']:
        _url = entry['request']['url']
        if "live.m3u8" in _url:
            stream_url = _url
            break

    if stream_url:
        print(f"Stream URL found: {stream_url}")
        return stream_url
    else:
        print("Stream URL not found.")
        return "Not Found"

    # Cleanup WebDriver and BrowserMob Proxy
    driver.quit()
    server.stop()