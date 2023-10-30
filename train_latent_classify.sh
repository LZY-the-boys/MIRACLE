source .model.sh
if [ ! "$TASK" ];then
   echo 'you have to set the $TASK'
   exit -1
fi
source .cmd.sh
if [ ! "$ATTR" ];then
   echo 'you have to set the $ATTR'
   exit -1
fi
LATENTTASK=$TASK-$ATTR-latent

# depend on VAE
CHECKPOINT=outs/attribute-dialogue/$TASK
echo "encoder $ENCODE; decoder $DECODE; checkpoint $CHECKPOINT"

WANDB_PROJECT=attribute_dialogue WANDB_NAME=$LATENTTASK \
accelerate launch \
train_classify.py --task-name $LATENTTASK --attr $ATTR --type 'latent' \
--micro-batch 256 --total-batch 1024 --learning-rate 5e-3 --num-epoch 50 -w 0 \
--encoder-path $ENCODE --decoder-path $DECODE --checkpoint-dir $CHECKPOINT \
--no-use-wandb $CMD