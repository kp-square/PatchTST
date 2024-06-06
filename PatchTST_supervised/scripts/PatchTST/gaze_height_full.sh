if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=24
label_len=12
model_name=PatchTST

root_path_name=./dataset/
data_path_name=chattahoochee_1hr.csv
model_id_name=gaze_height_full
data_name=gaze_height_full
frequency=1h
pred_len=3

random_seed=2021
for seq_len in 24
do
    python -u run_longExp.py \
      --target 'gaze_height'\
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --freq $frequency \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 4 \
      --n_heads 4 \
      --d_model 8 \
      --d_ff 256 \
      --dropout 0.2\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 8\
      --stride 4\
      --des 'Exp' \
      --train_epochs 50\
      --embed 'fixed'\
      --itr 1 --batch_size 512 --learning_rate 0.002 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done