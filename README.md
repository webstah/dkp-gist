# dkp_cifar10
This repository is an implementation of the Direct Kolen Pollack(DKP) method, as well as Direct Feedback Alignment(DFA), using PyTorch. The network used to test each method consists of two convolutional layers followed by two fully connected layers, and the dataset trained on is CIFAR10. DKP is a combination of the two alternative credit assignment learning algorithms DFA and the Kolen Pollack(KP) method as proposed by Akrout et al.

main.py usage example:
```
python main.py --train-mode DKP --batch-size 50 --epochs 100
```
### Direct Feedback Alignment vs Direct Kolen Pollack
Both **DFA** and **DKP** make use of direct connection from the output of the network to each layer in the backwards path. This attribute allows for the parallelization of the backwards pass, also refered to as *backwards unlocking*, meaning that each layer's gradient can be calculated and updated in parallel with another.

In **DFA**, <img src="https://render.githubusercontent.com/render/math?math=B_{\ell}"> is a fixed random weight matrix that projects the gradient at the output of a network to the output of the layer <img src="https://render.githubusercontent.com/render/math?math=\ell">. Just as it is with backpropagation, the learning signal at the output of the network <img src="https://render.githubusercontent.com/render/math?math=\delta_{k}">, where <img src="https://render.githubusercontent.com/render/math?math=f'()"> is the derivative of the activation function, would be calculated in the following way.
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=\delta_{k} = error\odot f'(a_{k})"></p></br>

Then, the learning signal at some layer <img src="https://render.githubusercontent.com/render/math?math=\ell - 1">, as prescribed by DFA(and DKP), would be calculated in the following way.
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=\delta_{\ell-1} = \delta_{k}\cdot B_{\ell}\odot f'(a_{\ell-1})"></p></br>
  
As for **DKP**, the rules above remain the same, however <img src="https://render.githubusercontent.com/render/math?math=B_{\ell}"> is no longer a fixed matrix. We will update the backward matrices using the following rule.
<p align="center"><img src="https://render.githubusercontent.com/render/math?math=\nabla B_{\ell} = - a_{\k}^T\cdot a_{\ell - 1}"></p>

### Top-1 Accuracy on CIFAR10

*Results below are an average of four randomly initialized trials trained for 50 epochs with default parameters.*

|               |  Test Accuracy  |
| ------------- | --------------- |
|      DKP      | 67.52% ± 0.002  |
|      DFA      | 61.02% ± 0.006  |
|      BP       | 70.47% ± 0.004  |


#### References

- <a href="http://papers.nips.cc/paper/6441-direct-feedback-alignment-provides-learning-in-deep-neural-networks.pdf" target="_blank">Direct Feedback Alignment Provides Learning in
Deep Neural Networks, Nøkland</a>
- <a href="https://arxiv.org/pdf/1904.05391.pdf" target="_blank">Deep Learning without Weight Transport, Akrout et al.</a>
