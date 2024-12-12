import os
import time

import requests
from selenium.webdriver import Chrome, ChromeOptions, Remote
from selenium.webdriver.common.by import By

from driver import Driver


class Driver:
    def __init__(
        self, remote=False, loading_seconds=3, executor="http://localhost:4444/wd/hub"
    ):
        self.loading_seconds = loading_seconds

        options = ChromeOptions()
        if remote:
            self.driver = Remote(command_executor=executor, options=options)
        else:
            self.driver = Chrome(options=options)

        self.driver.maximize_window()

    def get(self, url):
        self.driver.get(url)
        time.sleep(self.loading_seconds)

    def close(self):
        self.driver.close()

    def scroll_down(self):
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(self.loading_seconds)

    def find_element(self, selector):
        return self.driver.find_element(By.CSS_SELECTOR, selector)

    def find_elements(self, selector):
        return self.driver.find_elements(By.CSS_SELECTOR, selector)

    def open_new_tab(self, url):
        self.driver.execute_script(f"window.open('{url}', '_blank');")
        self.driver.switch_to.window(self.driver.window_handles[-1])
        time.sleep(self.loading_seconds)

    def close_current_tab(self):
        self.driver.close()
        self.driver.switch_to.window(self.driver.window_handles[-1])


def get_category_images(
    keys: list[str],
    category: str,
    total=5000,
    url="https://www.amazon.com/s?k={}&s=exact-aware-popularity-rank&page={}",
):
    folder = os.path.join(os.path.dirname(__file__), category)
    if not os.path.exists(folder):
        os.mkdir(folder)

    driver = Driver(loading_seconds=0.5)
    try:
        imgs = set()
        count = 0
        keys.sort()
        for page in range(1, 20):
            zero_count = 0
            zero_keys = set()
            for key in zero_keys:
                keys.remove(key)
            for key in keys:
                formatted_url = url.format(key, page)
                driver.get(formatted_url)
                driver.scroll_down()

                items = driver.find_elements(
                    "div.s-result-item.s-asin > div > div > span > div > div > div > span > a > div > img.s-image"
                )
                if len(items) == 0:
                    items = driver.find_elements(
                        "div.s-result-item.s-asin > div > div > span > div > div > div > div > div > div > div > span > a > div > img.s-image"
                    )
                print(
                    "Found {} items in page {} in URL: {}".format(
                        len(items), page, formatted_url
                    )
                )
                if len(items) == 0:
                    zero_count += 1
                    zero_keys.add(key)

                for item in items:
                    src = item.get_attribute("src")
                    name = src.split("/")[-1].split(".")[0]
                    if name in imgs:
                        continue
                    img = requests.get(src).content
                    imgs.add(name)
                    path = os.path.join(folder, f"{name}.jpg")
                    with open(path, "wb") as file:
                        file.write(img)
                        count += 1
                    if count == total:
                        break

                print(f"Category {category} - Key {key} - Page {page} done!")
                if count == total:
                    break

            if zero_count == len(keys) or count == total:
                print("Done")
                break
    finally:
        driver.close()


if __name__ == "__main__":
    get_category_images(
        keys=[
            "refrigerator",
            "microwave",
            "oven",
            "dishwasher",
            "washer",
            "dryer",
            "vacuum",
            "air+conditioner",
            "fan",
            "heater",
            "iron",
            "blender",
            "toaster",
            "coffee+maker",
            "kettle",
            "juicer",
            "mixer",
            "food+processor",
            "grill",
        ],
        category="home-appliances",
    )
