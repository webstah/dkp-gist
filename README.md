# Direct Kolen Pollack

This repository is an implementation of the Direct Kolen Pollack(DKP) method, as well as Direct Feedback Alignment(DFA), using PyTorch. DKP is a combination of two alternative credit assignment learning algorithms: DFA and the Kolen Pollack(KP) method as adapted by Akrout et al. The network used for testing each method on CIFAR10 consists of two convolutional layers followed by two fully connected layers. For testing on CIFAR100 we use AlexNet with batch normalization after each convolutional layer.

*main.py* usage example:
```
python main.py --train-mode DKP --batch-size 50 --epochs 100
```

*alexnet/main.py* usage example:
```
python main.py -a alexnet --dist-url 'tcp://127.0.0.1:8080' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 /home/user01/datasets/ILSVRC2012 --dataset cifar100 --train-mode DFA
```
Code for *alexnet/main.py* and *alexnet/models/alexnet.py* borrows heavily from the repositories below:
* https://github.com/pytorch/examples/tree/master/imagenet
* https://github.com/pytorch/vision/tree/master/torchvision/models

### Direct Feedback Alignment vs Direct Kolen Pollack
Both **DFA** and **DKP** make use of direct connections from the output of the network to each layer in the backward path. This attribute allows for the parallelization of the backwards pass, also refered to as *backwards unlocking*, meaning that each layer's gradient can be calculated and updated in parallel with another.

In **DFA**, <img src="https://render.githubusercontent.com/render/math?math=B_{\ell}"> is a fixed random weight matrix that projects the gradient at the output of a network to the output of layer <img src="https://render.githubusercontent.com/render/math?math=\ell - 1">. Just as it is with backpropagation, the learning signal at the output of the network <img src="https://render.githubusercontent.com/render/math?math=\delta_{k}">, where <img src="https://render.githubusercontent.com/render/math?math=f'()"> is the derivative of the activation function, would be calculated in the following way.
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=\delta_{k} = error\odot f'(a_{k})"></p></br>

Then, the learning signal at some layer <img src="https://render.githubusercontent.com/render/math?math=\ell - 1">, as prescribed by DFA(and DKP), would be calculated in the following way.
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=\delta_{\ell-1} = \delta_{k}\cdot B_{\ell}\odot f'(a_{\ell-1})"></p></br>
  
As for **DKP**, the rules above remain the same, however <img src="https://render.githubusercontent.com/render/math?math=B_{\ell}"> is no longer a fixed matrix. We will adjust the backward matrices using the following update rule.
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=\nabla B_{\ell} = - \delta_{k}^T\cdot a_{\ell - 1}"></p>


- <a href="http://papers.nips.cc/paper/6441-direct-feedback-alignment-provides-learning-in-deep-neural-networks.pdf" target="_blank">Direct Feedback Alignment Provides Learning in
Deep Neural Networks, Nøkland</a>
- <a href="https://arxiv.org/pdf/1904.05391.pdf" target="_blank">Deep Learning without Weight Transport, Akrout et al.</a>
- <a href="https://ieeexplore.ieee.org/document/374486" target="_blank">Backpropagation without Weight Transport, Kolen et al.</a>
- <a href="https://arxiv.org/pdf/1906.04554.pdf" target="_blank">Principled Training of Neural Networks
with Direct Feedback Alignment, Launay et al.</a>
