import requests
import os
from bs4 import BeautifulSoup
from tqdm import tqdm

import zipfile
import glob
import shutil
import json
import mailparser
import argparse

# Add utilities
def empty_dir(directory_path):
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            # if the current item is a file, remove it
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # if the current item is a directory, remove it recursively using shutil.rmtree()
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def process_file(file_addr):
    message = ""
    id = os.path.basename(file_addr)
    mail = mailparser.parse_from_file(file_addr)
    subject = mail.subject
    message = mail.body
    return id, {"subject" : subject, "content" : message.strip()}


# Step 1: Send a request to the webpage
url = 'https://lamp-benchmark.github.io/download'
response = requests.get(url)

# Step 2: Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Step 3: Find all anchor tags
anchors = soup.find_all('a')

# Step 4: Extract href attribute from each anchor tag
urls = []
for anchor in anchors:
    href = anchor.get('href')
    if 'ciir' in href:
        urls.append(href)

# Target directory to save the downloaded JSON files
base_target_dir = 'raw_data'

# Create the base target directory if it doesn't exist
os.makedirs(base_target_dir, exist_ok=True)

# Download each JSON file and save it to the target directory
for i, url in enumerate(urls):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        # Extract the filename from the URL
        filename = os.path.basename(url)
        
        # Extract the directory path from the URL
        dir_path = os.path.dirname(url).replace('https://ciir.cs.umass.edu/downloads/LaMP/', '')

        # Remove the split name since this is in the file name (train,val,etc)
        last_slash_index = dir_path.rfind('/')
        dir_path = dir_path[:last_slash_index]

        # If the url does not contain the word time, then it is user split.
        # Since this is not specified by URL, we add it explicitly for structuring the data folder
        if 'time' not in dir_path:
            dir_path = 'user/'+dir_path
        
        # Construct the full target directory path
        target_dir = os.path.join(base_target_dir, dir_path)
        
        # Create the target directory if it doesn't exist
        os.makedirs(target_dir, exist_ok=True)
        
        # Calculate the total size of the file (if available)
        total_size = int(response.headers.get('content-length', 0))
        
        # Create a progress bar
        progress_bar = tqdm(total=total_size, unit='B', unit_scale=True, desc=filename)

        # Check if file exists, and skip it if so
        file_exists = os.path.isfile(os.path.join(target_dir, filename))
        if file_exists:
            print(f'{filename} already exists.')
        
        # Save the file to the target directory with progress bar
        filepath = os.path.join(target_dir, filename)
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
        
        progress_bar.close()
        print(f'Successfully downloaded {filename}')
    else:
        print(f'Failed to download {url} (status code: {response.status_code})')

# Step 5: Prepare avocado dataset (TODO)
# avocado_files_dir = ""
# extract_addr = ""
# output_dir = ""
# input_question_file_train = ""
# input_question_file_dev = ""
# input_question_file_test = ""
# time_based_separation = ""

# with open(input_question_file_train) as file:
#     input_questions_train = json.load(file)
# with open(input_question_file_dev) as file:
#     input_questions_dev = json.load(file)
# with open(input_question_file_test) as file:
#     input_questions_test = json.load(file)

# all_required_files = set()
# for sample in input_questions_train + input_questions_dev + input_questions_test:
#     all_required_files.add(sample['input'])
#     for p in sample['profile']:
#         all_required_files.add(p['text'])
    
# zip_addrs = glob.glob(os.path.join(avocado_files_dir, "*"))
# os.makedirs(extract_addr, exist_ok=True)
# database = dict()
# for zip_addr in tqdm.tqdm(zip_addrs):
#     with zipfile.ZipFile(zip_addr, 'r') as zobj:
#         zobj.extractall(path = extract_addr)
#         extracted_files_addrs = glob.glob(os.path.join(extract_addr, "*/*"))
#         for file_addr in extracted_files_addrs:
#             if os.path.basename(file_addr) in all_required_files:
#                 id, obj = process_file(file_addr)
#                 database[id] = obj
#     empty_dir(extract_addr)
    
# os.makedirs(output_dir, exist_ok=True)

# inps_train, outs_train = [], []
# for sample in input_questions_train:
#     id = sample['input']
#     sample['input'] = f"Generate a subject for the following email: {database[id]['content']}"
#     sample['output'] = database[id]['subject']
#     for p in sample['profile']:
#         pid = p['text']
#         p['text'] = database[pid]['content']
#         p['title'] = database[pid]['subject']
#     if time_based_separation:
#         inps_train.append({"id" : sample['id'], "input" : sample['input'], "profile" : sample['profile'], "user_id" : sample['user_id']})
#     else:
#         inps_train.append({"id" : sample['id'], "input" : sample['input'], "profile" : sample['profile']})
#     outs_train.append({"id" : sample['id'], "output" : sample['output']})

# inps_dev, outs_dev = [], []
# for sample in input_questions_dev:
#     id = sample['input']
#     sample['input'] = f"Generate a subject for the following email: {database[id]['content']}"
#     sample['output'] = database[id]['subject']
#     for p in sample['profile']:
#         pid = p['text']
#         p['text'] = database[pid]['content']
#         p['title'] = database[pid]['subject']
#     if time_based_separation:
#         inps_dev.append({"id" : sample['id'], "input" : sample['input'], "profile" : sample['profile'], "user_id" : sample['user_id']})
#     else:
#         inps_dev.append({"id" : sample['id'], "input" : sample['input'], "profile" : sample['profile']})
#     outs_dev.append({"id" : sample['id'], "output" : sample['output']})

    
# inps_test= []
# for sample in input_questions_test:
#     id = sample['input']
#     sample['input'] = f"Generate a subject for the following email: {database[id]['content']}"
#     for p in sample['profile']:
#         pid = p['text']
#         p['text'] = database[pid]['content']
#         p['title'] = database[pid]['subject']
#     if time_based_separation:
#         inps_test.append({"id" : sample['id'], "input" : sample['input'], "profile" : sample['profile'], "user_id" : sample['user_id']})
#     else:
#         inps_test.append({"id" : sample['id'], "input" : sample['input'], "profile" : sample['profile']})
        
# with open(os.path.join(output_dir, "train_questions.json"), "w") as file:
#     json.dump(inps_train, file)

# with open(os.path.join(output_dir, "train_outputs.json"), "w") as file:
#     json.dump({"task":"LaMP_6","golds":outs_train}, file)

# with open(os.path.join(output_dir, "dev_questions.json"), "w") as file:
#     json.dump(inps_dev, file)

# with open(os.path.join(output_dir, "dev_outputs.json"), "w") as file:
#     json.dump({"task":"LaMP_6","golds":outs_dev}, file)

# with open(os.path.join(output_dir, "test_questions.json"), "w") as file:
#     json.dump({"task":"LaMP_6","golds":inps_test}, file)