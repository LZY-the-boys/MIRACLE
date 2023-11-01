export VAE_ATTR='0,1,2'
export VAE_ID='0,0,0;0,0,1;0,1,0;0,1,1;1,0,0;1,0,1;1,1,0;1,1,1'

cls=0.05
gap=0.002
kl=0.001

TASK=cls$cls-gap$gap-kl$kl
export TASK=cvae-6all-mul-attr-full-decoder-la768-$TASK
bash train_cvae.sh
ALGO=ODE bash gen_cvae.sh 
