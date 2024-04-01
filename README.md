# Replication Package for PTM4Tag+: Tag Recommendation of Stack Overflow Posts with Pre-trained Models


This is the codebase for the EMSE submission: "PTM4Tag+: Tag Recommendation of Stack Overflow Posts with Pre-trained Models"


## Data

The data used in thia paper can be download from this [link](https://zenodo.org/record/5604548#.YXoG7NZBw1I). 


```shell
    # Download train data
    wget https://zenodo.org/record/5604548/files/train.tar.gz
    # Download test data
    wget https://zenodo.org/record/5604548/files/test.tar.gz
    # tag file
    wget https://zenodo.org/record/5604548/files/_1_commonTags.csv
    # unzip file, change the filepath to your actual filepath
    tar -xvf file.tar.gz 

```


## Requirements
Our experiments were conducted under Ubuntu 18.04. We have made a ready-to-use docker image for this experiment.

```
docker pull dennishe9707/ptm4tag:emse
```

Then, assuming you have Nvidia GPUs, you can create a container using this docker image. An example:

```
docker run --name=ptm4tag_plus --gpus all -it --mount type=bind,src=/media/codebases,dst=/workspace dennishe9707/ptm4tag:emse
```

## Overview

```
.
├── README.md
├── preprocess
│   ├── 01_process_to_tensor.py
│   └── example.sh
└── src
    ├── data_structure
    │   ├── __init__.py
    │   └── question.py
    ├── model
    │   ├── __init__.py
    │   ├── loss.py
    │   └── model.py
    ├── rq1
    │   ├── test_triplet.py
    │   ├── train.py
    │   └── train_triplet.py
    ├── rq2
    │   ├── 02_1_train_no_title.bash
    │   ├── 02_2_train_no_text.bash
    │   ├── 02_3_train_no_code.bash
    │   ├── __init__.py
    │   ├── test_triplet_csv.py
    │   ├── test_triplet_no_code.py
    │   ├── test_triplet_no_text.py
    │   ├── test_triplet_no_title.py
    │   ├── train.py
    │   ├── train_no_code.py
    │   ├── train_no_text.py
    │   └── train_no_title.py
    └── util
        ├── __init__.py
        ├── data_util.py
        ├── eval_util.py
        └── util.py
```

`preprocess/`: Contains scripts related to the initial processing (tokenization) of the dataset.


`data_structure/`: Defines the data structure of the Stack Overflow posts used in the dataset.
question.py: Contains the Question class definition, representing a Stack Overflow question.


`model/`: Contains the implementation for the PTM4Tag+ framework.

`rq1/` and `rq2/`: Directories for the code related to specific research questions.

`test_triplet.py`: Script for testing the PTM4Tag+ model.
`train_triplet.py`: Script for training the PTM4Tag model.

`util/`: Utility functions supporting data handling, evaluation, and other common tasks.

# Tokenization

Before training the model, we need to tokenize the dataset to convert the raw text into a format that's compatible with the pre-trained models. Tokenization involves splitting the text into tokens (words or subwords), which can then be converted to numerical IDs. For more details, you can refer to this [tutorial](https://huggingface.co/learn/nlp-course/en/chapter2/4).



For this project, we have utilize various pre-trained models, we provided an example using the [BERTOverflow model](https://huggingface.co/jeniya/BERTOverflow), which is optimized for programming-related text found on Stack Overflow.

Step-by-Step Guide for Tokenization:

Prepare your dataset: Download the data. We assume the data files are in the format of pickle files, and we assume your dataset is under the `processed_train` directory.

We have included an example script, process/example.sh, to demonstrate how to run the tokenization process. Refer to this script as a template for the following steps.

```
cd /preprocesss

. example.sh
```

Here's the command for executing the tokenization script: Run the 01_process_to_tensor.py script with the necessary arguments to tokenize your dataset. 

```
python 01_process_to_tensor.py \

    --input_dir ../data/processed_train \
    --out_dir ../data/bertoverflow/ \
    --model_type jeniya/BERTOverflow \
    --title_max 50 \
    --text_max 50 \
    --code_max 50
```

--input_dir: The directory containing your raw dataset.
--out_dir: The directory where the tokenized dataset will be saved.
--model_type: Specifies the pre-trained model used for tokenization. For this example, we use jeniya/BERTOverflow.
--title_max, --text_max, --code_max: Maximum length for the title, text, and code segments of the posts, respectively. Posts longer than these values will be truncated.




## PTM4Tag+ Training and Testing

### RQ1
navigate to `/src/rq1/`


#### Model Train

```python
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch \
    --nproc_per_node=2 train_triplet.py \
    --data_folder ../../data/bertoverflow \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 64 \
    --logging_steps 100 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --code_bert jeniya/BERTOverflow \
    --learning_rate 7e-5
```

Key Arguments:
`--data_folder`: Specifies the directory containing the pre-processed data formatted for training. 

`--output_dir`: The output directory where the model checkpoints and predictions will be written.
`--vocab_file`: Specifies the path containing the tags. 
--per_gpu_train_batch_size 64: Sets the batch size to 64 for each GPU. This means each GPU will process 64 examples in each training step.

`--code_bert`: Indicates the pre-trained model to be used for initialization before training. You can the correct model name from the huggingface website. 




#### Model Test
```python

python -u test_triplet.py \
    --data_dir ../../data/test \
    --model_path /path/to/your/model_checkpoint.pt
    --test_batch_size 64 \
    --mlb_latest
```

Make sure to replace /path/to/your/model_checkpoint.pt with the actual file path to your trained model checkpoint.


### RQ2

In RQ2, we conduct the ablation study on the effect of different post components. 
Navigate to `src/rq2/`


```bash
# remove title
. 02_1_train_no_title.bash
# remove description
. 02_2_train_no_text.bash
# remove code
. 02_3_train_no_code.bash
```

Testing
```python
CUDA_VISIBLE_DEVICES=6,7 python -u test_triplet_no_code.py \
    --data_dir ../../data/test_tensor \
    --test_batch_size 500 \
    --code_bert microsoft/codebert-base \
    --model_path tbert.pt \
    --mlb_latest 2>&1| tee ./logs/test_trinity_no_code0926.log
    
```