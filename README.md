# PEFT For Large Language Models Personalization (LaMP)

## Data

### LaMP Tasks 1-5 & 7
Run `download_data.py`. Data will be downloaded to `./raw_data/` folder. Time and user splits will be in separate folders as `raw_data/time` and `raw_data/user` respectively.

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

## Evaluation

The instructions for evaluating your results on the test set are provided [here](https://lamp-benchmark.github.io/leaderboard). In order to evaluate your results on the dev set, we provided an evaluation script that can be found here:


Evaluate all tasks together:

```
python eval/eval_all.py \
    --golds_zip /*Address to all gold labels for all tasks zipped in a file*/ \
    --preds_zip /*Address to all predictions for all tasks zipped in a file*/ \
    --temp_dir /*Address to a temp dir for extracting files*/ \
    --output_file /*Address to the results file*/ \
```

Evaluate one task:

```
python eval/eval_task.py \
    --golds_json /*Address to gold labels for the task as a json file*/ \
    --preds_json /*Address to predictions for the task as a json file*/ \
    --task_name /*Name of the task [LaMP_1, LaMP_2, LaMP_3, LaMP_4, LaMP_5, LaMP_6, LaMP_7]*/
    --output_file /*Address to the results file*/ \
```

The pred files should follow the exact same format as the gold files:

```
{
    "task" : "/*task name*/",
    "golds" : [
        {
            "id" : "/*sample 1 id*/",
            "output" : "/*output of the model for the first sample*/"
        },
        ...,
        {
            "id" : "/*sample n id*/",
            "output" : "/*output of the model for the n'th sample*/"
        }
    ]
}
```

## Personalizing LLMs with RAG (LaMP)

You first need to create an environment for this using the following script:

```
python3 -m venv lamp_venv
source lamp_venv/bin/activate
pip install -r LaMP/requirements.txt
```

### Ranking Profiles based on the Input

The first step is to sort items in each user profile based on the input for the task:

```
cd LaMP
python rank_profiles.py \
    --input_data_addr /*input questions for one of the LaMP tasks*/ \
    --output_ranking_addr /*output address for the generated ranking file*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
    --ranker /*the ranking model to be used [bm25, contriever, recency]*/ \
    [optional] --use_date /*the batch size for ranking*/ \
    [optional] --use_date \ /*if used, it adds time to the text of each profile item*/
    [optional] --contriever_checkpoint /*address to the Contriever checkpoint to be used*/ \

```

After that, use the following script to sort the profiles in the dataset based on the ranking file:

```
cd LaMP
python utils/merge_with_rank.py \
    --lamp_questions_addr /*address to the LaMP task inputs file*/ \
    --lamp_output_addr /*address to the LaMP task outputs file*/ \
    --profile_ranking_addr /*address to the generated ranking file from the previous script*/
    --merged_output_addr /*address to the sorted dataset using the provided ranking file*/ \

```

### Training LLM with RAG

The next step is to train the LLM on a LaMP task:

```
cd LaMP
python train_llm.py \
    --train_data /*address to sorted training data using the previous step*/ \
    --validation_data /*address to sorted validation data using the previous step*/ \
    [optional] --test_data /*address to sorted test data using the previous step*/ \
    --model_name /*address to the model that should be used for initialization of the LLM*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
    --output_dir /*output directory to save results and checkpoints*/ \
    --retriever /*the ranking model to be used [bm25, contriever, recency]*/ \
    --use_profile \ /*used to perfrom personalization with RAG */
    --is_ranked \ /*used if you pre-ranked the profiles based on the provided retrieval model*/
    --num_retrieved /*number of items to be retrieved from the user profile*/ \ 
```

### Zero-shot Evaluation of LLM with RAG

You can also evaluate the LLMs with the following script:

```
cd LaMP
python evaluate_llm.py \
    --validation_data /*address to sorted validation data using the previous step*/ \
    --model_addr /*address to the model that should be used for initialization of the LLM*/ \
    --task /*name of the task [LaMP-1, LaMP-2, ..., LaMP-7]*/ \
    --output_dir /*output directory to save results */ \
    --use_profile \ /*used to perfrom personalization with RAG */
    --retriever /*the ranking model to be used [bm25, contriever, recency]*/ \
    --is_ranked \ /*used if you pre-ranked the profiles based on the provided retrieval model*/
    --num_retrieved /*number of items to be retrieved from the user profile*/ \ 
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