# dkp_cifar10
This repository is a convolutional implementation of the Direct Kolen Pollack method, as well as Direct Feedback Alignment, using PyTorch. DKP is a combination of the two alternative credit assignment learning algorithms Direct Feedback Alignment(DFA) and the Kolen Pollack(KP) method as proposed by Akrout et al.

main.py usage example:
```
python main.py --train-mode DKP --batch-size 50 --epochs 100
```

#### Top-1 Accuracy on CIFAR10:

*Results below are an average of four randomly initialized trials trained for 50 epochs. The learning rate on the forward weights is set to 5e-4 for all learning algorithms.

|               |  Test Accuracy  |
| ------------- | --------------- |
|      DKP      | 67.52% ± 0.002  |
|      DFA      | 61.02% ± 0.006  |
|      BP       | 70.47% ± 0.004  |


#### References

<a href="http://papers.nips.cc/paper/6441-direct-feedback-alignment-provides-learning-in-deep-neural-networks.pdf" target="_blank">Direct Feedback Alignment Provides Learning in
Deep Neural Networks, Nøkland</a>

<a href="https://arxiv.org/pdf/1904.05391.pdf" target="_blank">Deep Learning without Weight Transport, Akrout et al.</a>
