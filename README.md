# dkp_cifar10
This repository is a convolutional implementation of the Direct Kolen Pollack method and Direct Feedback Alignment trained on CIFAR10 using PyTorch. DKP is a combination of the two alternative credit assignment algorithms Direct Feedback Alignment(DFA) and the Kolen Pollack(KP) method as proposed by Akrout et al.

main.py usage example:
```
python main.py --train-mode DKP --batch-size 50 --epochs 100
```

<a href="http://papers.nips.cc/paper/6441-direct-feedback-alignment-provides-learning-in-deep-neural-networks.pdf" target="_blank">Direct Feedback Alignment Provides Learning in
Deep Neural Networks, Nøkland</a>

<a href="https://arxiv.org/pdf/1904.05391.pdf" target="_blank">Deep Learning without Weight Transport, Akrout et al.</a>
