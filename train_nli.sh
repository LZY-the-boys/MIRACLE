source ./.model.sh
TASK=dialogue-nli

WANDB_PROJECT=attribute_dialogue WANDB_NAME=$TASK \
accelerate launch  --mixed_precision fp16 \
train_classify.py \
--task-name $TASK --type 'nli' \
--micro-batch 16 --total-batch 128 --learning-rate 5e-5 --num-epoch 25 -w 0.05 \
--model-path $CLASSIFY --data-path 'dataset/nli.jsonl' --dev-data-path 'dataset/dev_nli.jsonl'  \
$CMD --no-use-wandb
