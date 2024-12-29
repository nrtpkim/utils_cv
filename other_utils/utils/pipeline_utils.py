from typing import Dict, Any
import time
import os
import requests
import logging as logger


def track_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(
            f"Time taken to execute {func.__name__}: {end_time - start_time} seconds"
        )
        return result

    return wrapper


def convert_keys_to_int(original_dict: Dict[str, Any]) -> Dict[int, Any]:
    return {int(key): value for key, value in original_dict.items()}


def get_latest_folder(
    bucket_name: str,
    task: str,
    poc_experiments_path: str = "assets/poc_experiments",
):
    dirs = os.path.join(f"/gcs/{bucket_name}", poc_experiments_path)
    dirs_list = [dir for dir in os.listdir(dirs) if dir.startswith(task)]
    if len(dirs_list) != 0:
        sorted_folder_names = sorted(dirs_list, key=lambda x: int(x[len(task) :]))
        return sorted_folder_names[-1]
    return ""


def discord_noti(
    msg,
    name,
    webhook_url="https://discord.com/api/webhooks/1224640673513472052/j3iTspQS-zi5XDp0yPURzV-NrABbIbByt1_P3D5L59QD8FKcBZ55l3Fg4m9L9lFU129-",
):

    payload = {
        "username": name,
        "content": msg,
    }
    response = requests.post(webhook_url, data=payload)
    if response.status_code in [200, 204]:
        logger.info("Message sent successfully.")
    else:
        logger.error("Failed to send the message. Status code:", response.status_code)
    return
