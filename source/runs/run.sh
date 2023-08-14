cd ../main/

export CUDA_VISIBLE_DEVICES=0
nohup python server.py --server_port=9999 --dataset="PhysioNet" --num_rounds=2 --num_epochs=2 > ../../ckps/PhysioNet/server.out &
sleep 30
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9999 --dataset="PhysioNet" --subdataset="CHA" &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9999 --dataset="PhysioNet" --subdataset="CPS" &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9999 --dataset="PhysioNet" --subdataset="G12" &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python client.py --server_port=9999 --dataset="PhysioNet" --subdataset="PTB" &
sleep 3