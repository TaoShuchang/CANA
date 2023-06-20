

# ogbn-products
CUDA_VISIBLE_DEVICES=3 nohup python -u run_flag.py --dropout 0.3 --perturb_size 0.01 --dataset ogbproducts  --suffix final > logs/ogbproducts.log 2>&1 &

# reddit
CUDA_VISIBLE_DEVICES=2 nohup python -u run_flag.py --dropout 0 --perturb_size 0.01 --dataset reddit --suffix final > logs/reddit.log 2>&1 &

# ogbn-arxiv
CUDA_VISIBLE_DEVICES=0 nohup python -u run_flag.py --dropout 0.3 --perturb_size 0.002 -m 30 --dataset ogbarxiv --suffix final > logs/ogbarxiv.log 2>&1 &