python train_imgreid_xent.py \
       -s dukemtmcreid \
       -t dukemtmcreid \
       --height 256 \
       --width 128 \
       --optim amsgrad \
       --lr 0.0003 \
       --max-epoch 100 \
       --stepsize 20 40 \
       --train-batch-size 32 \
       --test-batch-size 100 \
       -a resnet50_fc512 \
       --save-dir log/resnet50-dukemtmcreid-xent-amsgrad \
       --gpu-devices 0 \
       --start-eval 59 \
       --eval-freq 10 \
       #--label-smooth \ # label smoothing regularizer
