cd ../main/

export CUDA_VISIBLE_DEVICES=0
nohup python server.py --server_port=9999 --dataset="PhysioNet" --subdataset="CHA"  > ../../ckps/PhysioNet/CHA/result.out &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python server.py --server_port=9999 --dataset="PhysioNet" --subdataset="CPS"  > ../../ckps/PhysioNet/CPS/result.out &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python server.py --server_port=9999 --dataset="PhysioNet" --subdataset="G12"  > ../../ckps/PhysioNet/G12/result.out &
sleep 3
export CUDA_VISIBLE_DEVICES=0
nohup python server.py --server_port=9999 --dataset="PhysioNet" --subdataset="PTB"  > ../../ckps/PhysioNet/PTB/result.out &
sleep 3