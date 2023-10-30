export PYTHONPATH=.
IP=$(hostname -I | cut -d' ' -f1)
echo "host $IP"

CLASSIFY="bert-base-uncased"
ENCODE="bert-base-uncased"
DECODE="microsoft/DialoGPT-medium"
BATCH=4

if [ ! "$CUDA_VISIBLE_DEVICES" ];then
    export CUDA_VISIBLE_DEVICES=0
fi