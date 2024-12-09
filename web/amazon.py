import os
from uuid import uuid4

import requests

from driver import Driver


def get_category_images(
    keys,
    category,
    url="https://www.amazon.com/s?k={}&s=exact-aware-popularity-rank&page={}",
):
    folder = os.path.join(os.path.dirname(__file__), category)
    if not os.path.exists(folder):
        os.mkdir(folder)

    driver = Driver(remote=True, loading_seconds=2)
    try:
        keys.sort()
        for key in keys:
            for page in range(1, 4):
                formatted_url = url.format(key, page)
                driver.get(formatted_url)
                driver.scroll_down()

                items = driver.find_elements(
                    ".a-section.a-spacing-none.puis-padding-right-small.s-title-instructions-style > .a-size-mini.a-spacing-none.a-color-base.s-line-clamp-2 > .a-link-normal.s-underline-text.s-underline-link-text.s-link-style.a-text-normal"
                )
                if len(items) == 0:
                    items = driver.find_elements(
                        ".a-section.a-spacing-small.puis-padding-left-small.puis-padding-right-small .a-link-normal.s-underline-text.s-underline-link-text.s-link-style.a-text-normal"
                    )

                print(
                    "Found {} items in page {} in URL: {}".format(
                        len(items), page, formatted_url
                    )
                )
                for item in items:
                    products = item.get_attribute("href")
                    driver.open_new_tab(products)
                    src = driver.find_element("#imgTagWrapperId img").get_attribute(
                        "src"
                    )

                    img = requests.get(src).content
                    path = os.path.join(folder, f"{uuid4()}.jpg")
                    with open(path, "wb") as file:
                        file.write(img)

                    driver.close_current_tab()

            print(f"Category {category} - Key {key} done!")
    finally:
        driver.close()


if __name__ == "__main__":
    get_category_images(
        keys=[
            "laptop",
            "tablet",
            "camera",
            "headphones",
            "speaker",
            "monitor",
            "keyboard",
            "computer+mouse",
            "printer",
            "projector",
            "smartwatch",
            "iphone",
            "samsung+phone",
            "xiaomi",
            "huawei",
            "desktop+computer",
            "router",
            "modem",
        ],
        category="electronics",
    )
    get_category_images(
        keys=[
            "t-shirt",
            "jeans",
            "shoes",
            "hat",
            "belt",
            "sunglasses",
            "dress",
            "jacket",
            "watch",
            "bag",
            "scarf",
            "gloves",
            "socks",
            "sweater",
            "hoodie",
            "shorts",
            "skirt",
            "suit",
        ],
        category="fashion",
    )
    get_category_images(
        keys=[
            "sofa",
            "table",
            "chair",
            "bed",
            "desk",
            "cabinet",
            "shelf",
            "lamp",
            "dresser",
            "rug",
            "mirror",
            "bench",
            "stool",
            "ottoman",
            "couch",
            "bookcase",
            "nightstand",
            "wardrobe",
        ],
        category="furniture",
    )
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
