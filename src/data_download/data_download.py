import requests
import shutil
import os
import zipfile
import subprocess

def get_dataset(name, download_dir, is_zip, url):
    """
    Downloads and extracts a dataset from a given URL.
    
    Args:
        name: Name of the dataset
        download_dir: Directory to download the dataset to
        is_zip: Whether the dataset is a zip file
        url: URL to download the dataset from
    """
    # Create dataset directory and build full path
    dataset_path = os.path.join(download_dir, name)
    os.makedirs(dataset_path, exist_ok=True)
    
    if is_zip:
        # Handle zip file downloads
        filepath = os.path.join(dataset_path, f'{name}.zip')
        
        # Download the zip file using requests
        print(f'Downloading {name} dataset to {filepath}...')
        response = requests.get(url, stream=True)
        with open(filepath, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

        # Extract the contents and clean up
        print(f'Extracting zip file of {name} dataset...')
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(dataset_path)
        os.remove(filepath)
    else:
        # For non-zip files, use wget to recursively download
        subprocess.run(['wget', '-r', url], cwd=dataset_path)