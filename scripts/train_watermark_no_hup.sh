python3 main_watermark_modify.py \
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
  --jpeg_quality 50 \
  --remark 'main_watermarking' \
  > nohup.log 2>&1 &
  
  # 等待 Python 脚本运行完成
wait

# 执行关机命令
shutdown -h now