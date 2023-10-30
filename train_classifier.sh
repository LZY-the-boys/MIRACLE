source .model.sh
if [ ! "$ATTR" ];then
   echo 'you have to set the $ATTR'
   exit -1
fi
TASK=$ATTR-text

WANDB_PROJECT=attribute_dialogue WANDB_NAME=$TASK \
accelerate launch  --mixed_precision fp16  --dynamo_backend eager --num_processes 1 --num_machines 1 \
train_classify.py --task-name $TASK --type 'text' --attr $ATTR \
--micro-batch 256 --total-batch 1024 --learning-rate 5e-5 --num-epoch 25 -w 0.05 \
--model-path $CLASSIFY \
--no-use-wandb $CMD