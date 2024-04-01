# Replication Package for PTM4Tag+: Tag Recommendation of Stack Overflow Posts with Pre-trained Models

## Data

https://zenodo.org/record/5604548#.YXoG7NZBw1I

## Download Data

### Train

```shell
    wget https://zenodo.org/record/5604548/files/train.tar.gz
```

### Test

```shell
    wget https://zenodo.org/record/5604548/files/test.tar.gz
```
### Tag

```shell
    wget https://zenodo.org/record/5604548/files/_1_commonTags.csv
```

### Unzip
```
   tar -xvf file.tar.gz 
```

- Train File: ./bert/triplet/train_trinity.bash
- Test File: ./bert/triplet/test_triplet.py


# Tokenization
use `process/example.sh` as the example to tokenize the data first


# Training 
```python
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python train_trinity.py \
    --data_folder ../../data/train \
    --output_dir ../../data/results \
    --per_gpu_train_batch_size 2 \
    --logging_steps 100 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 3 \
    --learning_rate 1e-5
```

# Test
```python

python -u test_triplet.py \
    --data_dir ../../data/test \
    --code_bert Salesforce/codet5-base \
    --test_batch_size 64 \
    --mlb_latest
```