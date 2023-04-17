from io import BytesIO
from tqdm import tqdm
import requests


def download_file_from_url(url):
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)

    binary_file_object = BytesIO()
    for data in response.iter_content(block_size):
        progress_bar.update(len(data))
        binary_file_object.write(data)

    progress_bar.close()

    # Set the position of the BytesIO object back to the beginning
    binary_file_object.seek(0)

    return binary_file_object
