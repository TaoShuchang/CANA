
# Evaluate the attacks by detection methods.

# ogbproducts
# Use the attacked graphs downloaded from https://drive.google.com/file/d/17SWRJx9IT-7ZHkkcf5yDe2IcFSCGXUM4/view?usp=sharing
nohup python -u run_detection.py --suffix final  --gpu 0 --dataset ogbproducts > logs/ogbproducts_final.log 2>&1 &  

# Use the generated attacked graphs. 
nohup python -u run_detection.py --suffix attacked  --gpu 0 --dataset ogbproducts > logs/ogbproducts_attacked.log 2>&1 &  


# reddit
# Use the attacked graphs downloaded from https://drive.google.com/file/d/17SWRJx9IT-7ZHkkcf5yDe2IcFSCGXUM4/view?usp=sharing
nohup python -u run_detection.py --suffix final  --gpu 3 --dataset reddit > logs/reddit_final.log 2>&1 &  

# Use the generated attacked graphs. 
nohup python -u run_detection.py --suffix attacked  --gpu 0 --dataset reddit > logs/reddit_attacked.log 2>&1 &  


# ogbarxiv
# Use the attacked graphs downloaded from https://drive.google.com/file/d/17SWRJx9IT-7ZHkkcf5yDe2IcFSCGXUM4/view?usp=sharing
nohup python -u run_detection.py --suffix final  --gpu 4 --dataset ogbarxiv > logs/oogbarxiv_final.log 2>&1 &  

# Use the generated attacked graphs. 
nohup python -u run_detection.py --suffix attacked  --gpu 0 --dataset ogbarxiv > logs/ogbarxiv_attacked.log 2>&1 &  


