#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Define model and task names
model_name="TMLP"
task_name="text_earnings"

# Arrays to hold generated experiment parameters
model_ids=()
prediction_windows=()
embedding_files=()
test_years=()
test_quarters=()

# Define available configurations
year_options=(2019 2020 2021 2022 2023)
quarter_options=("first" "second" "third" "fourth")
datasets=("DEC")
dataset_paths=("DEC.csv")
prediction_window_options=(3 7 15 30)
embedding_file_options=("DEC" "DECRandomTicker" "DECRandomAll")

# Generate combinations of experimental setups
for prediction_window in "${prediction_window_options[@]}"; do
  for embedding_file in "${embedding_file_options[@]}"; do
    for year in "${year_options[@]}"; do
      for quarter in "${quarter_options[@]}"; do
        # Skip known invalid combinations
        if [[ "$year" == "2019" && "$quarter" == "first" ]]; then
          continue
        fi

        # Generate experiment ID
        model_id="m${model_name}_d${datasets[0]}_e(${embedding_file})"

        # Append configuration to arrays
        model_ids+=("$model_id")
        prediction_windows+=("$prediction_window")
        embedding_files+=("$embedding_file")
        test_years+=("$year")
        test_quarters+=("$quarter")
      done
    done
  done
done

# Execute training for each experiment
for ((i = 0; i < ${#model_ids[@]}; i++)); do
  python -u run.py \
    --task_name "$task_name" \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path "${dataset_paths[0]}" \
    --model_id "${model_ids[i]}" \
    --model "$model_name" \
    --data "$task_name" \
    --features S \
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
    --emb_dim 3072 \
    --emb_file "${embedding_files[i]}" \
    --prediction_window "${prediction_windows[i]}" \
    --test_year "${test_years[i]}" \
    --test_quarter "${test_quarters[i]}"
done