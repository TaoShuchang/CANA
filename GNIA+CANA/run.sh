# GNIA+CANA

# ogbn-products
nohup python -u run_gnia_cana.py --dataset ogbproducts --suffix 1e-3 --batchsize 128 --alpha 0.5 --beta 0.01 --Dopt 10 --lr_G 1e-3 --lr_D 1e-3 --gpu 6 > logs/ogbproducts_1e-3_bs128.log 2>&1 &

# Reddit
nohup python -u run_gnia_cana.py --dataset reddit --alpha 0.2 --beta 0.01 --lr_G 1e-3 --lr_D 1e-3 --gpu 5 > logs/reddit.log 2>&1 &


# ogbn-arxiv
nohup python -u run_gnia_cana.py --dataset ogbarxiv --alpha 2 --beta 0.01 --Dopt 5 --lr_G 1e-3 --lr_D 1e-3 --gpu 3 > logs/ogbarxiv.log 2>&1 &

