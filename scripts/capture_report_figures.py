"""Automated capture of high-resolution figures for the project report.
Uses headless Chrome with Selenium.
"""
import time
from pathlib import Path
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "report" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

def setup_driver(width=1400, height=1600):
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument(f"--window-size={width},{height}")
    driver = webdriver.Chrome(options=options)
    driver.set_window_size(width, height)
    return driver

def capture_streamlit_figures():
    print("[*] Launching browser to capture Streamlit dashboard figures...")
    driver = setup_driver(1400, 1600)
    
    try:
        driver.get("http://localhost:8501")
        time.sleep(6)
        
        # 1. Global Trends (Tab 0)
        tabs = driver.find_elements(By.CSS_SELECTOR, '[role="tab"]')
        print(f"[*] Found {len(tabs)} tabs in Streamlit.")
        
        if tabs:
            tabs[0].click()
            time.sleep(3)
            driver.save_screenshot(str(FIGS / "01-trends.png"))
            print("[OK] Saved 01-trends.png")
            
            # 2. Mumbai Trends
            # Click Mumbai radio in sidebar
            radios = driver.find_elements(By.XPATH, "//label[contains(., 'Mumbai')]")
            if radios:
                radios[0].click()
                time.sleep(3)
                driver.save_screenshot(str(FIGS / "01-trends-mumbai.png"))
                print("[OK] Saved 01-trends-mumbai.png")
                
                # 3. Mumbai Sentiment
                tabs = driver.find_elements(By.CSS_SELECTOR, '[role="tab"]')
                if len(tabs) > 1:
                    tabs[1].click()
                    time.sleep(3)
                    driver.save_screenshot(str(FIGS / "02-sentiment-mumbai.png"))
                    print("[OK] Saved 02-sentiment-mumbai.png")
                
                # Switch back to Global
                global_radios = driver.find_elements(By.XPATH, "//label[contains(., 'Global')]")
                if global_radios:
                    global_radios[0].click()
                    time.sleep(3)
            
            # 4. Global Sentiment (Tab 1)
            tabs = driver.find_elements(By.CSS_SELECTOR, '[role="tab"]')
            if len(tabs) > 1:
                tabs[1].click()
                time.sleep(3)
                driver.save_screenshot(str(FIGS / "02-sentiment.png"))
                print("[OK] Saved 02-sentiment.png")
                
            # 5. Outbreak Anomalies (Tab 2)
            if len(tabs) > 2:
                tabs[2].click()
                time.sleep(3)
                driver.save_screenshot(str(FIGS / "07-anomalies.png"))
                print("[OK] Saved 07-anomalies.png")
                
            # 6. Topics (Tab 4: NMF Topic Modeling)
            if len(tabs) > 4:
                tabs[4].click()
                time.sleep(3)
                driver.save_screenshot(str(FIGS / "03-topics.png"))
                print("[OK] Saved 03-topics.png")
                
            # 7. Keywords (Tab 5: Keywords & Co-occurrence)
            if len(tabs) > 5:
                tabs[5].click()
                time.sleep(3)
                driver.save_screenshot(str(FIGS / "04-keywords.png"))
                print("[OK] Saved 04-keywords.png")
                
            # 8. Cases Correlation (Tab 6: Lead/Lag Correlation)
            if len(tabs) > 6:
                tabs[6].click()
                time.sleep(3)
                driver.save_screenshot(str(FIGS / "05-cases-correlation.png"))
                print("[OK] Saved 05-cases-correlation.png")
                
            # 9. Try the model & XAI (Tab 7: Live Model & XAI)
            if len(tabs) > 7:
                tabs[7].click()
                time.sleep(3)
                driver.save_screenshot(str(FIGS / "06-try-the-model.png"))
                print("[OK] Saved 06-try-the-model.png")
                
    finally:
        driver.quit()

def capture_flux_web():
    print("[*] Capturing FLUX Web Command Center at http://localhost:8000...")
    driver = setup_driver(1400, 1100)
    try:
        driver.get("http://localhost:8000")
        time.sleep(4)
        
        # Click Enter Command Center to go to dashboard
        enter_btn = driver.find_elements(By.XPATH, "//button[contains(., 'Command Center')]")
        if enter_btn:
            enter_btn[0].click()
            time.sleep(3)
            
        driver.save_screenshot(str(FIGS / "08-flux-web.png"))
        print("[OK] Saved 08-flux-web.png")
    finally:
        driver.quit()

if __name__ == "__main__":
    capture_streamlit_figures()
    capture_flux_web()
    print("[OK] All report figures captured successfully!")
