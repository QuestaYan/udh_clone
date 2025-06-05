# For the "test" argument input the UDH trained folder path, 
# for the "test_diff" argument input the DDH trained folder path. 
python3 main_modify.py \
  --imageSize 256 \
  --bs_secret 44 \
  --num_training 1 \
  --num_secret 1 \
  --num_cover 1 \
  --channel_cover 3 \
  --channel_secret 3 \
  --norm 'batch' \
  --loss 'l2' \
  --beta 0.75 \
  --remark 'main' \
  --test '/2025-06-04_H01-13-26_256_1_1_120_3_batch_l2_0.75_1colorIn1color_main_udh' \
  --testPics '模型文件/'\