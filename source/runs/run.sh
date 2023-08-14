cd ../main/

export CUDA_VISIBLE_DEVICES=0
nohup python server.py --server_port=9999 --dataset="PACS" --subdataset="0" > ../../ckps/PACS/0/0.out &
sleep 30