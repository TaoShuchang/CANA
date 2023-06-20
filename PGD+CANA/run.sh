# PGD+CANA

# ogbn-products
CUDA_VISIBLE_DEVICES=3 nohup python -u run_pgd_cana.py  --dataset ogbproducts --batch_size 2099 --suffix pgd+cana  --alpha 10 --lr 1e-2 --lr_D 1e-3 --Dopt 1 > logs/ogbproducts.log 2>&1 &

# reddit
CUDA_VISIBLE_DEVICES=0 nohup python -u run_imppgd.py  --dataset 12k_reddit --suffix pgd+cana --alpha 20 --lr 1e-1 --lr_D 1e-2 --Dopt 20 > logs/reddit.log 2>&1 &

# ogbn-arxiv
CUDA_VISIBLE_DEVICES=1 nohup python -u run_pgd_cana.py  --dataset ogbarxiv --batch_size 1800 --suffix pgd+cana --alpha 50 --lr 1e-2 --lr_D 1e-3 --Dopt 4 > logs/ogbarxiv.log 2>&1 &
