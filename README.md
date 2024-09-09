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