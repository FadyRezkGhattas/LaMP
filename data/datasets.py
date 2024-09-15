from torch.utils.data import Dataset
import json
import datasets
import torch

def get_all_labels(task):
    if task == "LaMP-1":
        return ["[1]","[2]"]
    elif task == "LaMP-2":
        return ['sci-fi', 'based on a book', 'comedy', 'action', 'twist ending', 'dystopia', 'dark comedy', 'classic', 'psychology', 'fantasy', 'romance', 'thought-provoking', 'social commentary', 'violence', 'true story']
    elif task == "LaMP-3":
        return ["1", "2", "3", "4", "5"]
    elif task == "LaMP-4":
        return []
    elif task == "LaMP-5":
        return []
    elif task == "LaMP-6":
        return []
    elif task == "LaMP-7":
        return []

def create_preprocessor(tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        return model_inputs
    return preprocess_dataset

def create_preprocessor_scores(tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        model_inputs['id_1'] = examples['id_1']
        model_inputs['id_2'] = examples['id_2']
        return model_inputs
    return preprocess_dataset

def create_preprocessor_scores_seq(tokenizer, max_length):
    def preprocess_dataset(examples):
        inputs = [example for example in examples["source"]]
        targets = [example for example in examples["target"]]
        model_inputs = tokenizer(inputs, text_target=targets, max_length=max_length, truncation=True)
        model_inputs['id'] = examples['id']
        return model_inputs
    return preprocess_dataset

def convert_to_hf_dataset(dataset, cache_dir):
    def gen():
        for idx in range(len(dataset)):
            yield dataset[idx]
    return datasets.Dataset.from_generator(gen, cache_dir = cache_dir)

def get_io_keys(task):
    if task == "LaMP-1":
        return None # TODO
    elif task == "LaMP-2":
        return 'description', 'tag'
    elif task == "LaMP-3":
        return 'text', 'score'
    elif task == "LaMP-4":
        return 'text', 'title'
    elif task == "LaMP-5":
        return 'abstract', 'title'
    elif task == "LaMP-6":
        return None # TODO
    elif task == "LaMP-7":
        return None # TODO

class GeneralSeq2SeqProfileDataset(Dataset):
    def __init__(self, task, create_prompt, val=False, user_id=None, data=None, data_addr=None) -> None:
        """
        Loads dataset for specified task and user.

        Args:
            - **task** (str) : The name of the task.
            - **create_prompt** (function) : Function that creates a prompt for each sample.
            - **val** (bool) : If True, return datapoints for user input; otherwise, return profile data for training.
            - **user_id** (int) : ID of the user to load data from. Defaults to None.
            - **data** (a list of dict) : Data to use for this dataset. If given, it's assumed to be a single user.
            - **data_addr** (str) : Path to the data file.
        
        Returns:
            A dictionary containing the loaded dataset and its metadata

        Example usage:

            >>> from data.datasets import GeneralSeq2SeqProfileDataset
            >>> from prompts.prompts import create_prompt_generator as f
            >>> dataset = GeneralSeq2SeqProfileDataset('LaMP-2', f, user_id=125, data_addr='./data_raw/user/LaMP_2/train_questions.json')
        """
        super().__init__()
        self.task = task
        self.create_prompt = create_prompt
        self.val = val

        assert (data is None and data_addr != '') or (data != '' and data_addr is None), "Either data or data_addr must not be empty."
        if data_addr is not None:
            assert user_id is not None, "User id must be provided when using data_addr."
            with open(data_addr) as f:
                data = json.load(f)
            self.data = data[user_id]
        elif data is not None:
            self.data = data
        self.i_key, self.o_key = get_io_keys(self.task)

    def __getitem__(self, index):
        if not self.val:
            return {
                "id" : self.data['profile'][index]['id'],
                "source" : self.create_prompt(self.data['profile'][index][self.i_key], self.task),
                "target" : self.data['profile'][index][self.o_key]
            }
        else:
            return {
                "id" : self.data['id'],
                "source" : self.data['input'],
                "target" : self.data['output']
            }
    
    def __len__(self):
        if not self.val:
            return len(self.data['profile'])
        else:
            return 1

class GeneralSeq2SeqDataset(Dataset):

    def __init__(self, data_addr, use_profile, task, create_prompt = None) -> None:
        super().__init__()
        with open(data_addr) as file:
            self.data = json.load(file)
        self.use_profile = use_profile
        self.task = task
        assert not (use_profile ^ (create_prompt != None)), "You should provide a prompt maker function when you use profile"
        self.create_prompt = create_prompt

    def __getitem__(self, index):
        if self.use_profile:
            return {
                "id" : self.data[index]['id'],
                "source" : self.create_prompt(self.data[index]['input'], self.data[index]['profile'], self.task),
                "target" : self.data[index]['output']
            }
        else:
            return {
                "id" : self.data[index]['id'],
                "source" : self.data[index]['input'],
                "target" : self.data[index]['output']
            }
    
    def __len__(self):
        return len(self.data)

class GeneralSeq2SeqForScoreGenerationDataset(Dataset):

    def __init__(self, data_addr, use_profile, task, create_prompt = None, max_prof_size = -1) -> None:
        super().__init__()
        with open(data_addr) as file:
            self.data = json.load(file)
        self.use_profile = use_profile
        self.task = task
        assert not (use_profile ^ (create_prompt != None)), "You should provide a prompt maker function when you use profile"
        self.create_prompt = create_prompt
        self.max_prof_size = max_prof_size
        self.size = 0
        self.index_dict = dict()
        for i, x in enumerate(self.data):
            for j, y in enumerate(x['profile']):
                if max_prof_size == -1 or j < self.max_prof_size:
                    self.index_dict[self.size] = (i, j)
                    self.size += 1

    def __getitem__(self, index):
        self.use_profile = True
        i, j = self.index_dict[index]
        if self.use_profile:
            return {
                "source" : self.create_prompt(self.data[i]['input'], [self.data[i]['profile'][j]], self.task),
                "target" : self.data[i]['output'],
                "id_1" : self.data[i]['id'],
                "id_2" : self.data[i]['profile'][j]['id']
            }
        else:
            return {
                "source" : self.data[index]['input'],
                "target" : self.data[index]['output']
            }
    
    def __len__(self):
        return self.size