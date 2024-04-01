python 02_process_to_tensor.py \
    --input_dir ../data/processed_train \
    --out_dir ../data/bertoverflow/ \
    --model_type jeniya/BERTOverflow 2>&1| tee ./bertoverflow.log