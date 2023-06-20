# Adversarial Camouﬂage for Node Injection Attack on Graphs

**This repository is our Pytorch implementation of our paper:**

[Adversarial Camouﬂage for Node Injection Attack on Graphs](https://arxiv.org/abs/2208.01819)

By Shuchang Tao, Qi Cao, Huawei Shen, Yunfan Wu, Liang Hou, Fei Sun, and Xueqi Cheng

## Introduction

In this paper, we find that the malicious nodes generated by existing node injection attack methods are prone to failure in practical situations, since defense and detection methods can easily distinguish and remove the injected malicious nodes from the original normal nodes.

Figure 1 shows  the distribution of attributes of injected nodes and original normal nodes for state-of-the-art node injection attack methods, i.e., G-NIA [39] and TDGIA [59], and heuristic imperceptible constraint HAO. Node attributes of **injected nodes (red) look different from the normal ones (blue)**. The defects weaken the eﬀectiveness of such attacks in practical scenarios where defense/detection methods are commonly used.

![motivation](./README_figures/motivation.jpg)

## CANA

We first formulate the camouﬂage on graphs as the distribution similarity between the ego networks centering around the injected nodes and the ego networks centering around the normal nodes, characterizing both network structures and node attributes. Then we propose an **adversarial camouﬂage framework for node injection attacks**, namely **CANA**, to improve the camouﬂage of injected nodes through an adversarial paradigm. CANA is a general framework, which could be attached to any existing node injection attack methods (G), improving node camouﬂage while inheriting the performance of existing node injection attacks.

Further details can be found in our [paper](https://arxiv.org/abs/2208.01819).


![CANA](./README_figures/CANA.jpg)

## Results

Extensive experiments demonstrate that CANA can signiﬁcantly improve the attack performance under defense/detection methods with higher camouﬂage or imperceptibility.

![results](./README_figures/results.jpg)

## Datasets and splits

Download ogbarxiv, ogbproducts (the subgraph in our paper), Reddit (the subgraph in our paper) from [Here](https://drive.google.com/file/d/1R0BGShORJdjaLDPSPLv9FVDHJMQOUswe/view?usp=drive_link).

Unzip the `datasets_CANA.zip` and put the folder `datasets` in the root directory.

## Environment

- Python >= 3.6
- pytorch >= 1.6.0
- scikit-learn >= 0.24.2
- matplotlib >= 3.3.4
- pyod >= 1.0.4
- scipy==1.5.4
- pandas >= 1.15

## Reproduce the results

- Inject nodes and Generate the attacked graphs by CANA

  - **Running scripts and parameters for all the datasets are given in `PGD+CANA/run.sh`, `TDGIA+CANA/run.sh`, `GNIA+CANA/run.sh`**

    Example Usage:

    ```
    cd GNIA+CANA
    mkdir logs
    nohup python -u run_gnia_cana.py --dataset ogbproducts --suffix cana --alpha 0.5 --beta 0.01 --Dopt 10 --lr_G 1e-3 --lr_D 1e-3 --gpu 0 > logs/ogbproducts_cana.log 2>&1 &
    ```

    Put the attacked graphs (e.g., `GNIA+CANA/new_graphs/ogbproducts_cana.npz`) into the directory  `final_graphs/ogbproducts`.

  - Please note that you can also **directly download attacked graphs used in our paper from [Here](https://drive.google.com/file/d/17SWRJx9IT-7ZHkkcf5yDe2IcFSCGXUM4/view?usp=sharing).** 
    Unzip `final_graphs.zip`, and put the `final_graphs` folder in the root directory.

- Evaluate the attack performance by detection and defense methods

  **Running scripts and parameters for all the datasets are given in `defense_detection/Detection/run.sh`, `defense_detection/FLAG/run.sh`, `defense_detection/GNNGuard/run.sh`**

  - Detections

    - Use the attacked graphs downloaded from the above link. Example usage:

      ```
      cd defense_detection/Detection
      mkdir logs
      nohup python -u eval_detect.py --suffix final  --gpu 0 --dataset ogbproducts > log/ogbproducts_final.log 2>&1 &  
      ```

      The accuracy can be found in `logs/ogbproducts/ogbproducts_final.csv`.

    - Use the generated attacked graphs. Example usage:

      ```
      cd defense_detection/Detection
      mkdir logs
      nohup python -u eval_detect.py --suffix attacked  --gpu 0 --dataset ogbproducts > log/ogbproducts_attacked.log 2>&1 &  
      ```

      The accuracy can be found in `logs/ogbproducts/ogbproducts_attacked.csv`.

  - FLAG

    Train FLAG model and Evaluate the attacked graphs by FLAG model:

    ```
    cd defense_detection/FLAG
    mkdir logs
    CUDA_VISIBLE_DEVICES=0 nohup python -u run_flag.py --dropout 0.3 --perturb_size 0.01 --dataset ogbproducts  --suffix final > logs/ogbproducts.log 2>&1 &
    ```

  - GNNGuard

    Train GNNGuard model and Evaluate the attacked graphs by GNNGuard model:

    ```
    cd defense_detection/GNNGuard
    mkdir logs
    CUDA_VISIBLE_DEVICES=0 nohup python -u run_gnnguard.py --dataset ogbproducts  --dropout 0.3 --suffix final > logs/ogbproducts_final.log 2>&1 &
    ```

    