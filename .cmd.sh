if [ ! "$TASK" ];then
   echo 'you have to set the $TASK'
   exit -1
fi

CMD=' '
# for train_cvae
if [[ $TASK == *"-context"* ]]; then
    CMD+=" --use-context "
fi
if [[ $TASK == *"-full-decoder"* ]]; then 
    CMD+=" --full-decoder "
fi
if [[ $TASK == *"-share-encoder"* ]]; then
    CMD+=" --share-encoder "
fi
if [[ $TASK == *"-kl"* ]]; then
    if [[ $TASK =~ kl([0-9.]+) ]]; then
        kl_extracted=${BASH_REMATCH[1]}
        CMD+=" --kl-ratio $kl_extracted"
    else
        echo 'you set wrong $KL'
        exit -1
    fi
fi
if [[ $TASK == *"-la"* ]]; then
    if [[ $TASK =~ la([0-9]+) ]]; then
        la_extracted=${BASH_REMATCH[1]}
        CMD+=" --latent-size $la_extracted"
    else
        echo 'you set wrong $LA'
        exit -1
    fi
fi
if [[ $TASK == *"-cls"* ]]; then
    if [[ $TASK =~ cls([0-9.]+) ]]; then
        extracted=${BASH_REMATCH[1]}
        CMD+=" --attr-cls-ratio $extracted"
    else
        echo 'you set wrong $CLS'
        exit -1
    fi
fi
if [[ $TASK == *"-gap"* ]]; then
    if [[ $TASK =~ gap([0-9.]+) ]]; then
        extracted=${BASH_REMATCH[1]}
        CMD+=" --attr-gap-ratio $extracted"
    else
        echo 'you set wrong $GAP'
        exit -1
    fi
fi
if [[ $TASK == *"-prior-cls"* ]]; then 
    CMD+=" --prior-cls "
fi
if [[ $TASK == *"-prior-gap"* ]]; then 
    CMD+=" --prior-gap "
fi
if [[ $TASK == *"-negap"* ]]; then 
    CMD+=" --negap "
fi

# for gen_attr train_attr
if [ "$VAE_ID" ]; then
    CMD+=" --ids $VAE_ID " 
fi

if [ "$VAE_ATTR" ]; then 
    CMD+=" --attrs $VAE_ATTR "
fi
if [ "$ALGO" ]; then 
    CMD+=" --algorithm $ALGO "
fi
echo "commmand $CMD"