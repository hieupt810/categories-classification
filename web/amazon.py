import os

import requests

from driver import Driver


def get_category_images(
    keys,
    category,
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

                if count == total:
                    break

                print(f"Category {category} - Key {key} - Page {page} done!")
            if zero_count == len(keys):
                print("Exhausted")
                break
    finally:
        driver.close()


if __name__ == "__main__":
    # get_category_images(
    #     keys=[
    #         "laptop",
    #         "tablet",
    #         "camera",
    #         "headphones",
    #         "speaker",
    #         "monitor",
    #         "keyboard",
    #         "computer+mouse",
    #         "printer",
    #         "projector",
    #         "smartwatch",
    #         "iphone",
    #         "samsung+phone",
    #         "xiaomi",
    #         "huawei",
    #         "desktop+computer",
    #         "router",
    #         "modem",
    #     ],
    #     category="electronics",
    # )
    # get_category_images(
    #     keys=[
    #         "t-shirt",
    #         "jeans",
    #         "shoes",
    #         "hat",
    #         "belt",
    #         "sunglasses",
    #         "dress",
    #         "jacket",
    #         "watch",
    #         "bag",
    #         "scarf",
    #         "gloves",
    #         "socks",
    #         "sweater",
    #         "hoodie",
    #         "shorts",
    #         "skirt",
    #         "suit",
    #     ],
    #     category="fashion",
    # )
    # get_category_images(
    #     keys=[
    #         "sofa",
    #         "table",
    #         "chair",
    #         "bed",
    #         "desk",
    #         "cabinet",
    #         "shelf",
    #         "lamp",
    #         "dresser",
    #         "rug",
    #         "mirror",
    #         "bench",
    #         "stool",
    #         "ottoman",
    #         "couch",
    #         "bookcase",
    #         "nightstand",
    #         "wardrobe",
    #     ],
    #     category="furniture",
    # )
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
