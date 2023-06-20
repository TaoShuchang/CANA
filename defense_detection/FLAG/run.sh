# Evaluate the attacks by FLAG

# ogbn-products
# Use the attacked graphs downloaded from https://drive.google.com/file/d/17SWRJx9IT-7ZHkkcf5yDe2IcFSCGXUM4/view?usp=sharing
CUDA_VISIBLE_DEVICES=0 nohup python -u run_flag.py --dataset ogbproducts  --dropout 0.3 --perturb_size 0.01  --suffix final > logs/ogbproducts_final.log 2>&1 &

# Use the generated attacked graphs. 
CUDA_VISIBLE_DEVICES=0 nohup python -u run_flag.py --dataset ogbproducts  --dropout 0.3 --perturb_size 0.01  --suffix attacked > logs/ogbproducts_attacked.log 2>&1 &



# reddit
# Use the attacked graphs downloaded from https://drive.google.com/file/d/17SWRJx9IT-7ZHkkcf5yDe2IcFSCGXUM4/view?usp=sharing
CUDA_VISIBLE_DEVICES=2 nohup python -u run_flag.py --dataset reddit --dropout 0 --perturb_size 0.01 --suffix final > logs/reddit_final.log 2>&1 &

# Use the generated attacked graphs. 
CUDA_VISIBLE_DEVICES=2 nohup python -u run_flag.py --dataset reddit --dropout 0 --perturb_size 0.01 --suffix attacked > logs/reddit_attacked.log 2>&1 &



# ogbn-arxiv
# Use the attacked graphs downloaded from https://drive.google.com/file/d/17SWRJx9IT-7ZHkkcf5yDe2IcFSCGXUM4/view?usp=sharing
CUDA_VISIBLE_DEVICES=0 nohup python -u run_flag.py --dataset ogbarxiv --dropout 0.3 --perturb_size 0.002 -m 30 --suffix final > logs/ogbarxiv_final.log 2>&1 &

# Use the generated attacked graphs. 
CUDA_VISIBLE_DEVICES=0 nohup python -u run_flag.py --dataset ogbarxiv --dropout 0.3 --perturb_size 0.002 -m 30 --suffix attacked > logs/ogbarxiv_attacked.log 2>&1 &
