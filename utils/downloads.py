"""
Download utils (Reference: ultralytics/yolov5/utils/downloads.py)
"""

import os
import platform
import subprocess
import time
import urllib
import requests
import torch
from pathlib import Path


##########################################################################
# Attempt file download from url or url2
##########################################################################
def safe_download(file, url, url2=None, min_bytes=1E0, error_msg=''):
    # Attempts to download file from url or url2, checks and removes incomplete downloads < min_bytes
    file = Path(file)
    assert_msg = f"Downloaded file '{file}' does not exist or size is < min_bytes={min_bytes}"
    try:  # url1
        print(f'Downloading {url} to {file}...')
        torch.hub.download_url_to_file(url, str(file))
        assert file.exists() and file.stat().st_size > min_bytes, assert_msg  # check
    except Exception as e:  # url2
        file.unlink(missing_ok=True)  # remove partial downloads
        print(f'ERROR: {e}\nRe-attempting {url2 or url} to {file}...')
        os.system(f"curl -L '{url2 or url}' -o '{file}' --retry 3 -C -")  # curl download, retry and resume on fail
    finally:
        if not file.exists() or file.stat().st_size < min_bytes:  # check
            file.unlink(missing_ok=True)  # remove partial downloads
            print(f"ERROR: {assert_msg}\n{error_msg}")
        print('')

##########################################################################
# Attempt file download from YOLOv5-Repo (if does not exist)
##########################################################################
def attempt_download_yolov5(file, repo='ultralytics/yolov5'):
    file = Path(str(file).strip().replace("'", ''))

    if not file.exists():
        name = Path(urllib.parse.unquote(str(file))).name
        file.parent.mkdir(parents=True, exist_ok=True)  # make parent dir (if required)

        # GitHub Assets, i.e. ['yolov5s.pt', 'yolov5m.pt', ...]
        try:
            response = requests.get(f'https://api.github.com/repos/{repo}/releases/latest').json()  # GitHub API
            assets = [x['name'] for x in response['assets']]
            tag = response['tag_name']  # i.e. 'v1.0'
        except:  # fallback plan
            assets = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
                      'yolov5s6.pt', 'yolov5m6.pt', 'yolov5l6.pt', 'yolov5x6.pt']
            try:
                tag = subprocess.check_output('git tag', shell=True, stderr=subprocess.STDOUT).decode().split()[-1]
            except:
                tag = 'v5.0'

        if name in assets:
            safe_download(file,
                          url=f'https://github.com/{repo}/releases/download/{tag}/{name}',
                          # url2=f'https://storage.googleapis.com/{repo}/ckpt/{name}',  # Backup URL (optional)
                          min_bytes=1E5,
                          error_msg=f'{file} missing, try downloading from https://github.com/{repo}/releases/')

    return str(file)