from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options 
from selenium.webdriver.common.by import By

def pic():
    for i in range(0,1):
        chrome_options = Options()
        chrome_options.add_argument("--headless") #不開視窗，在背景下處理
        driver = webdriver.Chrome(options=chrome_options) 
        driver.get("https://www.ris.gov.tw/apply-idCard/app/idcard/IDCardReissue/main") #爬網址
        ######
        scroll_width = driver.execute_script('return document.body.parentNode.scrollWidth')
        scroll_height = driver.execute_script('return document.body.parentNode.scrollHeight')
        driver.set_window_size(scroll_width, scroll_height)
        driver.save_screenshot('./cap_pic/fullpage.jpg')
        # 網頁中圖片驗證碼的位置
        element = driver.find_element(by=By.XPATH, value='//*[@id="captchaImage_captcha-refresh"]')  # 抓取html裡圖的位置
        left = element.location['x']
        right = element.location['x'] + element.size['width']
        top = element.location['y']
        bottom = element.location['y'] + element.size['height']
        img = Image.open('./cap_pic/fullpage.jpg')
        img = img.crop((left, top, right, bottom))
        img = img.convert("RGB")
        img.save(f'./pic/{i}.jpg')
        print(f"已儲存:{i}.jpg")