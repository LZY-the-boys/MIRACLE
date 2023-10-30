source .model.sh

if [ ! "$TASK" ];then
   echo 'you have to set the $TASK'
   exit -1
fi

source .cmd.sh

export WANDB_PROJECT=attribute_dialogue 
export WANDB_NAME=$TASK 

accelerate launch train_cvae.py --task-name $TASK  \
--micro-batch 4 \
--encoder-path $ENCODE --decoder-path $DECODE --num-epoch 20 \
--no-use-wandb \
--output-path outs/attribute-dialogue/cvae-6all/$TASK \
$CMD