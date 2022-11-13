


CUDA_VISIBLE_DEVICES=3 python run_train.py --model=snn_ta --checkpointDir=ckpt_dir_snnta

CUDA_VISIBLE_DEVICES=3 python run_train.py --model=snn --checkpointDir=snn

CUDA_VISIBLE_DEVICES=3 python run_train.py --model=snnta --max_seq_length=20 --checkpointDir=snnta_20
CUDA_VISIBLE_DEVICES=4 python run_train.py --model=snnta --max_seq_length=30 --checkpointDir=snnta_30
CUDA_VISIBLE_DEVICES=7 python run_train.py --model=snnta --max_seq_length=50 --checkpointDir=snnta_50


CUDA_VISIBLE_DEVICES=3 python run_eval.py --model=snnta --max_seq_length=10 --checkpointDir=snnta_10
CUDA_VISIBLE_DEVICES=0 python run_train.py --model=snnta --max_seq_length=30 --checkpointDir=snnta_30
CUDA_VISIBLE_DEVICES=7 python run_eval.py --model=snnta --max_seq_length=50 --checkpointDir=snnta_50


CUDA_VISIBLE_DEVICES=
CUDA_VISIBLE_DEVICES=3 python run_eval.py --model=snn --max_seq_length=10 --checkpointDir=snn_10


CUDA_VISIBLE_DEVICES=7 python run_eval.py --model=snnta --max_seq_length=50 --checkpointDir=snnta_50



CUDA_VISIBLE_DEVICES=0 python run_train.py --model=snn --max_seq_length=10 --checkpointDir=snn_10

CUDA_VISIBLE_DEVICES=0 python run_eval.py --model=snn --max_seq_length=5 --checkpointDir=snn_5



CUDA_VISIBLE_DEVICES=2 python run_train.py --model=snn --max_seq_length=3 --checkpointDir=snn_3
CUDA_VISIBLE_DEVICES=2 python run_eval.py --model=snn --max_seq_length=3 --checkpointDir=snn_3


CUDA_VISIBLE_DEVICES=0 python run_train.py --model=snn --max_seq_length=8 --checkpointDir=snn_8_wv
CUDA_VISIBLE_DEVICES=1 python run_train.py --model=snn --max_seq_length=8 --checkpointDir=snn_8_wv_1
CUDA_VISIBLE_DEVICES=2 python run_train.py --model=snn --max_seq_length=8 --checkpointDir=snn_8_wv_2

CUDA_VISIBLE_DEVICES=7 python run_train.py --model=dnn --checkpointDir=dnn

