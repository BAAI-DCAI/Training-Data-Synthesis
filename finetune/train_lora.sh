# export CACHE_ROOT="ROOT PATH"
# export MODEL_NAME="${CACHE_ROOT}/huggingface/models--runwayml--stable-diffusion-v1-5/snapshots/39593d5650112b4cc580433f6b0435385882d819"
export MODEL_NAME="PATH TO SD"
versions=('v1')
methods=("gt_dm")
length=${#versions[@]}
# Loop through the array and set the OUTPUT_DIR variable accordingly
for method in "${methods[@]}"; do
  for ((i=0; i<$length; i++)); do 
      version="${versions[$i]}"
      export OUTPUT_DIR="./LoRA/checkpoint/${method}_${version}/all"
      export DATASET_NAME="./LoRA/train/"
      export LOG_DIR="./LoRA/train/logs"
      echo "Current OUTPUT_DIR: $OUTPUT_DIR, $i"
      if [ -f "$OUTPUT_DIR/pytorch_lora_weights.bin" ] || [ -f "$OUTPUT_DIR/pytorch_lora_weights.safetensors" ]; then
        echo "Folder exists. Skipping script execution."
      else
        if [ "$method" == "gt_dm" ]; then
          echo "script execution. ${method}"
          accelerate launch --mixed_precision="fp16" ./finetune/train_text_to_image_lora.py \
          --pretrained_model_name_or_path=$MODEL_NAME \
          --train_data_dir $DATASET_NAME --caption_column="text" \
          --report_to=tensorboard \
          --resolution=512 --random_flip \
          --train_batch_size=8 \
          --num_train_epochs=100 --checkpointing_steps=500 \
          --learning_rate=1e-04 --lr_scheduler="constant" \
          --seed=42 \
          --output_dir=${OUTPUT_DIR} \
          --snr_gamma=5 \
          --guidance_token=8 \
          --dist_match=0.003 \
          --logging_dir $LOG_DIR \
          --exp_id ${i} \
        else
          echo "Method not implemented"
        fi
        wait
        echo "All processes completed"
      fi
  done
done