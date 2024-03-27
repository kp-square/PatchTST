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
data_path_name=gaze_height_CHATTAHOOCHEE_02336000.csv
model_id_name=gaze_height
data_name=gaze_height
frequency=1h

random_seed=2021
for pred_len in 1
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
      --features S \
      --freq $frequency \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --e_layers 3 \
      --n_heads 4 \
      --d_model 16 \
      --d_ff 128 \
      --dropout 0.3\
      --fc_dropout 0.3\
      --head_dropout 0\
      --patch_len 16\
      --stride 8\
      --des 'Exp' \
      --train_epochs 100\
      --itr 1 --batch_size 512 --learning_rate 0.0001 >logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 
done