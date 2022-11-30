# Dependencies
This project depends on the following python packages:
1. pytorch 1.7.1+
2. torchvision 0.8.2+
3. numpy 1.19.2+
4. scikit-learn 0.23.2+
5. matplotlib 3.3.2+

# How to Run
## Construct datasets
Launch python from the current directory and type
``` python
from dataloader import BuildFedCifar10Dataset, BuildFedMNISTDataset, BuildFedFashionMNISTDataset

BuildFedCifar10Dataset(500, sort_by='label', similarity=0.0)
BuildFedMNISTDataset(500, sort_by='label', similarity=0.0)
BuildFedFashionMNISTDataset(500, sort_by='label', similarity=0.0)
```

## Run algorithms on the AUC maximization task
``` bash
bash ./test_scripts/test_auc_mnist_n_500_best.sh  # test algorithms on MNIST
bash ./test_scripts/test_auc_cifar10_n_500_best.sh  # test algorithms on CIFAR-10
```
The output files are stored in `./data/results_auc_mnist_best_1234/` and `./data/results_auc_cifar10_best_1234/`.

## Run algorithms on the robust adversarial neural network training task
``` bash
bash ./test_scripts/test_robustnn_mnist_n_500_best.sh  # test algorithms on MNIST
bash ./test_scripts/test_robustnn_fashion_mnist_n_500_best.sh  # test algorithms on Fashion MNIST
```
The output files are stored in `./data/results_robustnn_mnist_best_1234/` and `./data/results_robustnn_fashion_mnist_best_1234/`.

## Run algorithms on the GAN training task
1. Train models that will be used in the Inception score evaluation.
``` bash
python ./utils/train_inception_score_model.py --dataset_name=MNIST  # for MNIST
python ./utils/train_inception_score_model.py --dataset_name=FashionMNIST  # for Fashion MNIST
```

2. Run the algorithms on MNIST and Fashion MNIST.
``` bash
bash ./test_scripts/test_gan_mnist_n_500_best.sh  # test algorithms on MNIST
bash ./test_scripts/test_gan_fashion_mnist_n_500_best.sh  # test algorithms on Fashion MNIST

bash ./test_scripts/test_gan_mnist_n_500_varying_K.sh  # test CD-MA, CD-MAGE-MB, and CD-MAGE-VR with varying K on MNIST
bash ./test_scripts/test_gan_fashion_mnist_n_500_varying_K.sh  # test CD-MA, CD-MAGE-MB, and CD-MAGE-VR with varying K on Fashion MNIST
```

3. To run the algorithms on the CelebA dataset, we have to use 4 machines, each machine with 4 GPUs. First, modify line 15 in `test_script/test_gan_celeba_n_500_best.sh` to set the central node's IP, and then type the following 4 commands on the 4 machines, respectively.
``` bash
bash test_script/test_gan_celeba_n_500_best.sh nic_name 0 # on node 0 (the central node)
bash test_script/test_gan_celeba_n_500_best.sh nic_name 1 # on node 1
bash test_script/test_gan_celeba_n_500_best.sh nic_name 2 # on node 2
bash test_script/test_gan_celeba_n_500_best.sh nic_name 3 # on node 3
```
In the above commands, `nic_name` denotes the network interface card name (e.g., eno2) of the current machine.

The output files are stored in `./data/results_gan_mnist_best_1234/`, `./data/results_gan_fashion_mnist_best_1234/`, `./data/results_gan_celeba_best_1234/`.
