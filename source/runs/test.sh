cd ../main/

export CUDA_VISIBLE_DEVICES=0
python server.py --server_port=9999 --dataset="PhysioNet" --subdataset="CHA"
sleep 3
export CUDA_VISIBLE_DEVICES=0
python server.py --server_port=9999 --dataset="PhysioNet" --subdataset="CPS"
sleep 3
export CUDA_VISIBLE_DEVICES=0
python server.py --server_port=9999 --dataset="PhysioNet" --subdataset="G12"
sleep 3
export CUDA_VISIBLE_DEVICES=0
python server.py --server_port=9999 --dataset="PhysioNet" --subdataset="PTB"
sleep 3