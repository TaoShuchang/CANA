
# Evaluate the attacks by detection methods.

# ogbn-products
# Use the attacked graphs downloaded from https://drive.google.com/file/d/17SWRJx9IT-7ZHkkcf5yDe2IcFSCGXUM4/view?usp=sharing
nohup python -u run_detection.py --dataset ogbproducts --suffix final  --gpu 0 > logs/ogbproducts_final.log 2>&1 &  

# Use the generated attacked graphs. 
nohup python -u run_detection.py --dataset ogbproducts --suffix attacked  --gpu 0 > logs/ogbproducts_attacked.log 2>&1 &  


# reddit
# Use the attacked graphs downloaded from https://drive.google.com/file/d/17SWRJx9IT-7ZHkkcf5yDe2IcFSCGXUM4/view?usp=sharing
nohup python -u run_detection.py --dataset reddit --suffix final  --gpu 1 > logs/reddit_final.log 2>&1 &  

# Use the generated attacked graphs. 
nohup python -u run_detection.py --dataset reddit --suffix attacked  --gpu 0 > logs/reddit_attacked.log 2>&1 &  


# ogbn-arxiv
# Use the attacked graphs downloaded from https://drive.google.com/file/d/17SWRJx9IT-7ZHkkcf5yDe2IcFSCGXUM4/view?usp=sharing
nohup python -u run_detection.py --dataset ogbarxiv --suffix final  --gpu 0 > logs/oogbarxiv_final.log 2>&1 &  

# Use the generated attacked graphs. 
nohup python -u run_detection.py --dataset ogbarxiv --suffix attacked  --gpu 0 > logs/ogbarxiv_attacked.log 2>&1 &  


