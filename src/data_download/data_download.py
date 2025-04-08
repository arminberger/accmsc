import requests
import shutil
import constants.project_constants as project_constants
import constants.dataset_constants as dataset_constants
import os
import zipfile
import subprocess

def get_dataset(name):
    dataset_path = project_constants.GET_DATASET_DOWNLOAD_PATH(name)
    url, is_zip = dataset_constants.DATASET_DOWNLOAD_URLS[name]
    if is_zip:
        filepath = os.path.join(dataset_path, f'{name}.zip')


        response = requests.get(url, stream=True)

        with open(filepath, 'wb') as out_file:
            print(f'Downloading {name} dataset to {filepath}...')
            shutil.copyfileobj(response.raw, out_file)
        del response


        print(f'Extracting zip file of {name} dataset...')
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        # Delete zip file
        os.remove(filepath)
    else:
        command = ['wget', '-r', url]
        subprocess.run(command, cwd=dataset_path)