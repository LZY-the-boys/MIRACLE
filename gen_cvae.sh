source .model.sh
source .cmd.sh

echo "Task $TASK"

WANDB_PROJECT=attribute_dialogue WANDB_NAME=$TASK-gen-$atol-$rtol \
python gen_cvae.py -t $TASK \
--encoder-path $ENCODE --decoder-path $DECODE \
--checkpoint-dir outs/attribute-dialogue/cvae-6all/$TASK \
--postfix checkpoint-11315/pytorch_model.bin \
$CMD
