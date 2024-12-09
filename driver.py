import time

from selenium.webdriver import Chrome, ChromeOptions, Remote
from selenium.webdriver.common.by import By


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
