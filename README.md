# Multilinear Compressive Learning Framework 




What is MCL 
=============

In many application of Compressive Sensing, the signals often posess multi-dimensional structure such as MRI or HSI. Multilinear Compressive Learning is a framework which combines Multidimensional Compressive Sensing and Machine Learning to learn the inference tasks in an end-to-end manner. We demonstrate that MCL is both computationally efficient and accurate, outperforming the vector-based design proposed in [Adler et al](https://arxiv.org/abs/1610.09615). 

In this repository, we provide the full implementation that was used in our experiments in our work.

Dependencies
=============

The following dependencies should be installed prior to running our code:

* tensorflow
* keras
* tqdm
* numpy


Data
=====

The datasets, together with train/validation/test splits can be downloaded from [here](https://bit.ly/2Q6fe68). After cloning our repository and downloading the data, the data should be put in the directory named "data" at the same level as the "code" directory, i.e.,

MultilinearCompressiveLearningFramework/code
MultilinearCompressiveLearningFramework/data


Running Experiments
===================

After putting the data to the correct place, run::

	bash train.sh

to produce all experiment results
