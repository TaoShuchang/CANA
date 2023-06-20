# TDGIA+CANA

# ogbn-products
nohup python -u run_tdgia_cana.py --dataset ogbproducts --suffix tdgia+cana --alpha 5 --Dopt 100 --lr_D 1e-3 --lr 0.01 --epochs 101 --step 0.1 --gpu 3 > logs/ogbproducts_tdgia+cana.log 2>&1 &

# reddit
nohup python -u run_tdgia_cana.py --dataset reddit --suffix tdgia+cana --alpha 10 --Dopt 100 --lr 0.01 --epochs 101 --step 0.1 --gpu 4 > logs/reddit_tdgia+cana.log 2>&1 &

# ogbn-arxiv
nohup python -u run_tdgia_cana.py --dataset ogbarxiv --suffix tdgia+cana  --alpha 50 --Dopt 1 --lr 1e-3 --lr_D 1e-4 --epochs 31 --step 0.05 --gpu 4 > logs/ogbarxiv_tdgia+cana.log 2>&1 &
