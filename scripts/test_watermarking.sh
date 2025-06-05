# For the "test" argument input the UDH watermark trained folder path, 
python3 main_watermark_modify.py \
  --imageSize 128 \
  --bs_secret 44 \
  --num_training 1 \
  --num_secret 1 \
  --num_cover 1 \
  --channel_cover 3 \
  --channel_secret 3 \
  --norm 'batch' \
  --loss 'l2' \
  --beta 0.75 \
  --remark 'main_watermark_modify' \
  --test '2025-06-01_H01-21-15_256_1_1_44_2_batch_l2_0.75_1colorIn1color_main_watermarking' \
  --testPics '模型文件/'\