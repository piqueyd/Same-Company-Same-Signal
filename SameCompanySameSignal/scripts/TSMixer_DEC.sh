export CUDA_VISIBLE_DEVICES=0

# Define model name and data paths
model_name="TSMixer"
task_name="ts_earnings"


year_options=(2019 2020 2021 2022 2023)
quarter_options=("first" "second" "third" "fourth")
datasets=("DEC")
dataset_paths=("DEC.csv")
prediction_window_options=(3 7 15 30)
seq_len_options=(22)
label_len_options=(22)

seq_lens=()
lab_lens=()
model_ids=()
prediction_windows=()
test_years=()
test_quarters=()


for prediction_window in "${prediction_window_options[@]}"; do
  for seq_len in "${seq_len_options[@]}"; do
    for year in "${year_options[@]}"; do
      for quarter in "${quarter_options[@]}"; do
        # Skip known invalid combinations
        if [[ "$year" == "2019" && "$quarter" == "first" ]]; then
          continue
        fi

        # Generate experiment ID
        model_id="m${model_name}_d${datasets[0]}_seq${seq_len}"
        model_ids+=("$model_id")
        seq_lens+=("$seq_len")    
        prediction_windows+=("$prediction_window")
        test_years+=("$year")
        test_quarters+=("$quarter")
      done
    done
  done
done


# Iterate over indices
# {# ... }: This is a parameter expansion that returns the length of the value inside it.
# [@]: This is a syntax used to reference all elements of the array
for ((i=0; i<${#model_ids[@]}; i++)); do
  python -u run.py \
    --task_name "$task_name" \
    --is_training 1 \
    --root_path ./dataset \
    --data_path "${dataset_paths[0]}" \
    --model_id "${model_ids[i]}" \
    --ts_pattern "volatility" \
    --model "$model_name" \
    --data "$task_name" \
    --features S \
    --freq b \
    --seq_len "${seq_lens[i]}" \
    --label_len "${seq_lens[i]}" \
    --pred_len 1 \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 1 \
    --dec_in 1 \
    --c_out 1 \
    --batch_size 16 \
    --d_model 512 \
    --des 'Exp' \
    --itr 1 \
    --learning_rate 0.001 \
    --loss 'MSE' \
    --prediction_window "${prediction_windows[i]}" \
    --test_year "${test_years[i]}" \
    --test_quarter "${test_quarters[i]}"
done
