# COUP
Implementation code for paper "Classifier-Confidence Guided Adversarial Purification with Diffusion Models"
# CLASSIFIER GUIDANCE ENHANCES DIFFUSION-BASED ADVERSARIAL PURIFICATION BY PRESERVING PREDICTIVE INFORMATION

![GitHub set up](https://github.com/ZhangMingKun1/COUP/blob/main/asserts/fig_COUP.png "Main Idea of COUP")
</center> <!--结束居中对齐-->

**Abstract**

Adversarial purification is one of the promising approaches to defend neural networks against adversarial attacks. Recently, methods utilizing diffusion probabilistic models have achieved great success for adversarial purification in image classification tasks. However, such methods fall into the dilemma of balancing the needs for noise removal and information preservation. This paper points out that existing adversarial purification methods based on diffusion models gradually lose sample information during the core denoising process, causing occasional label shift in subsequent classification tasks. As a remedy, we suggest to suppress such information loss by introducing guidance from the classifier confidence. Specifically, we propose Classifier-cOnfidence gUided Purification (COUP) algorithm, which purifies adversarial examples while keeping away from the classifier decision boundary. Experimental results show that COUP can achieve better adversarial robustness under strong attack methods.



**Requirements**

We follow the requirements of DiffPure: https://github.com/NVlabs/DiffPure/tree/master

- Python 3.8

- CUDA=11.0

- Installation of the required library dependencies with Docker:

  ```python
  docker build -f diffpure.Dockerfile --tag=diffpure:0.0.1 .
  docker run -it -d --gpus 0 --name diffpure --shm-size 8G -v $(pwd):/workspace -p 5001:6006 diffpure:0.0.1
  docker exec -it diffpure bash
  ```

  

**Dataset**

We use CIFAR-10 dataset which can be automatically download in the code.



**Checkpoint**

 You have to download the checkpoint and put it in the 'pretrained' directory.

- Diffusion model
  - We use the VP-SDE (vp/cifar10_ddpmpp_deep_continuous) of [Score SDE](https://drive.google.com/file/d/16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif/view).
- Classifier
  - We use both WideResNet-28-10(no need to download separately) and [WideResNet-70-16](https://drive.google.com/file/d/16_-Ahc6ImZV5ClUc0vM5Iivf8OJ1VSif/view)



## To get the results of AutoAttack of our COUP:

+ Linf

  ```	python
  cd run_scripts/cifar10
  bash run_cifar_stand_inf_guide.sh 121 0 # standard mode
  bash run_cifar_rand_inf_guide.sh 121 0 # rand mode
  ```

+ L2

  ```	python
  cd run_scripts/cifar10
  bash run_cifar_stand_L2_guide.sh 121 0 # standard mode
  bash run_cifar_rand_L2_guide.sh 121 0 # rand mode
  ```


