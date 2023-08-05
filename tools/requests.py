import requests


def get_image_from_url(url: str, save_path: str) -> bool:
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            file.write(response.content)
        return True
    else:
        return False
