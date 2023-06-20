# GNIA+CANA

# ogbn-products
nohup python -u run_gnia_cana.py --dataset ogbproducts --suffix gnia+cana --alpha 0.5 --beta 0.01 --Dopt 10 --lr_G 1e-3 --lr_D 1e-3 --gpu 0 > logs/ogbproducts_gnia+cana.log 2>&1 &

# Reddit
nohup python -u run_gnia_cana.py --dataset reddit --suffix gnia+cana --alpha 0.2 --beta 0.01 --lr_G 1e-3 --lr_D 1e-3 --gpu 1 > logs/reddit.log 2>&1 &


# ogbn-arxiv
nohup python -u run_gnia_cana.py --dataset ogbarxiv --suffix gnia+cana --alpha 2 --beta 0.01 --Dopt 5 --lr_G 1e-3 --lr_D 1e-3 --gpu 1 > logs/ogbarxiv.log 2>&1 &

