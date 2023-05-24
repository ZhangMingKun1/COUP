# COUP
Implementation code for paper "Classifier-Confidence Guided Adversarial Purification with Diffusion Models"

We follow the requirements of DiffPure: https://github.com/NVlabs/DiffPure/tree/master



To get the results of AutoAttack of our COUP:

+ Linf

  + ```	python
    cd run_scripts/cifar10
    bash run_cifar_stand_inf_guide.sh 121 0 # standard mode
    bash run_cifar_rand_inf_guide.sh 121 0 # rand mode
    ```

+ L2

  + ```	python
    cd run_scripts/cifar10
    bash run_cifar_stand_L2_guide.sh 121 0 # standard mode
    bash run_cifar_rand_L2_guide.sh 121 0 # rand mode
    ```

