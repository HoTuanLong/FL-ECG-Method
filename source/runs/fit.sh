cd ../main/

export CUDA_VISIBLE_DEVICES=0
nohup python server.py --server_port=9985 --dataset="PhysioNet" > ../../ckps/PhysioNet/server.out &
sleep 30
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9985 --dataset="PhysioNet" --subdataset="CHA" > ../../ckps/PhysioNet/CHA/log.out &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9985 --dataset="PhysioNet" --subdataset="CPS" > ../../ckps/PhysioNet/CPS/log.out &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9985 --dataset="PhysioNet" --subdataset="G12" > ../../ckps/PhysioNet/G12/log.out &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9985 --dataset="PhysioNet" --subdataset="PTB" > ../../ckps/PhysioNet/PTB/log.out &
sleep 3