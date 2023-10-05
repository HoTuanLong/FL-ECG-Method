cd ../main/

export CUDA_VISIBLE_DEVICES=0
nohup python server.py --server_port=9980 --dataset="PhysioNet" > ../../ckps/PhysioNet/server.out &
sleep 30
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9980 --dataset="PhysioNet" --subdataset="CHA" --num_classes=30 > ../../ckps/PhysioNet/CHA/log.out &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9980 --dataset="PhysioNet" --subdataset="CPS" --num_classes=22 > ../../ckps/PhysioNet/CPS/log.out &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9980 --dataset="PhysioNet" --subdataset="G12" --num_classes=25 > ../../ckps/PhysioNet/G12/log.out &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9980 --dataset="PhysioNet" --subdataset="PTB" --num_classes=24 > ../../ckps/PhysioNet/PTB/log.out &
sleep 3