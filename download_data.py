import requests
import os
from bs4 import BeautifulSoup
from tqdm import tqdm

# Step 2: Send a request to the webpage
url = 'https://lamp-benchmark.github.io/download'
response = requests.get(url)

# Step 3: Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Step 4: Find all anchor tags
anchors = soup.find_all('a')

# Step 5: Extract href attribute from each anchor tag
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