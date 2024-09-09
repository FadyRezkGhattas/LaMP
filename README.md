# PEFT For Large Language Models Personalization (LaMP)

## Data

### LaMP Tasks 1-5 & 7
Run `download_data.py`. Data will be downloaded to `./data_raw/` folder. Time and user splits will be in separate folders as `data_raw/time` and `data_raw/user` respectively.

### LaMP 6: Personalized Email Subject Generation (Avocado dataset)

The [Avocado](https://catalog.ldc.upenn.edu/LDC2015T03) dataset is not publicly accessible. However, we provided the samples' id and the code we used to generate our dataset. Therefore, if you get access to the dataset, you can quickly generate the dataset with the same format as the other datasets in LaMP using the following code:

```
python data/avocado/create_avocado_dataset.py \
    --avocado_files_dir \*Address to the directory containing zip files for avocado dataset 'avocado-1.0.2/data/text'*\ \
    --extract_addr \*A temp dir to extract the files for creating dataset*\ \
    --output_dir \*The directory to generate the final dataset*\ \
    --input_question_file_train \*The address to the train_questions.json file we provided in LaMP*\ \
    --input_question_file_dev \*The address to the dev_questions.json file we provided in LaMP*\ \
    --input_question_file_test \*The address to the test_questions.json file we provided in LaMP*\
```

### Pre-Processing Ranks
The first step is to sort items in each user profile based on the input for the task:
```
python rank_profiles.py \
    --input_data_addr /*input questions for one of the LaMP tasks*/ \
    --output_ranking_addr /*output address for the generated ranking file*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
    --ranker /*the ranking model to be used [bm25, contriever, recency]*/ \
    [optional] --use_date /*the batch size for ranking*/ \
    [optional] --use_date \ /*if used, it adds time to the text of each profile item*/
    [optional] --contriever_checkpoint /*address to the Contriever checkpoint to be used*/ \
```
Alernatively, we provide a script ``ranking_data.sh``. The script runs all ranking pre-processing commands for train, dev, and test questions for all LaMP tasks for both user and time splits. The best performing ranker is used per task as provied in tables 6 and 8 in the original [LaMP paper](https://arxiv.org/abs/2304.11406). This is using contriever for all tasks except (1) LaMP-3T and LaMP-4T using Recency, and (2) LaMP-5U, and LaMP-6U using BM25.

If multiple GPUs are available, the script commands can be manually pushed into different devices.

## Credit
[LaMP: When Large Language Models Meet Personalization](https://arxiv.org/abs/2304.11406)

```
@misc{salemi2023lamp,
      title={La{MP}: When Large Language Models Meet Personalization}, 
      author={Alireza Salemi and Sheshera Mysore and Michael Bendersky and Hamed Zamani},
      year={2023},
      eprint={2304.11406},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```