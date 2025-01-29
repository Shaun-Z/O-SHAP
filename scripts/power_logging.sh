# 启动两个命令并将它们放在后台运行
python train.py --config config/ResNet50onEuroSAT/train.yaml &
pid1=$!
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit --format=csv -l 1 > gpu_full_log.csv &
pid2=$!

# 等待其中一个命令结束
wait -n $pid1 $pid2

# 终止另一个命令
kill $pid1 $pid2

