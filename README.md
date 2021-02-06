# HopSkipJumpAttack 

(previously named as Boundary Attack++)
Code for [HopSkipJumpAttack: A Query-Efficient Decision-Based Adversarial Attack](https://arxiv.org/abs/1904.02144) by Jianbo Chen, Michael I. Jordan, Martin J. Wainwright.

## Clarification
If you are using the code to compare HSJA with other algorithms, you will need to **set your own counter for the number of model evaluations**. "max_num_evals" in hsja.py is the maximum number of evaluations for estimating gradient (for each iteration), **not the total number of model evaluations**. To increase the total number of model evaluations, set a larger num_iterations. 
 

## Dependencies
The code for HopSkipJumpAttack runs with Python and requires Tensorflow of version 1.2.1 or higher. Please `pip install` the following packages:
- `numpy`
- `tensorflow` 
- `keras`
- `scipy`

## Running in Docker, MacOS or Ubuntu
We provide as an example the source code to run HopSkipJumpAttack on a ResNet trained on CIFAR-10. Run the following commands in shell:

```shell
###############################################
# Omit if already git cloned.
git clone https://github.com/Jianbo-Lab/HSJA
cd HSJA
############################################### 
# Carry out L2 based untargeted attack on 5 samples.
python main.py --constraint l2 --attack_type untargeted --num_samples 5
# Carry out L2 based targeted attack on 5 samples.
python main.py --constraint l2 --attack_type targeted --num_samples 5
# Carry out Linf based untargeted attack on 5 samples.
python main.py --constraint linf --attack_type untargeted --num_samples 5
# Carry out Linf based targeted attack on 5 samples.
python main.py --constraint linf --attack_type targeted --num_samples 5

# Results are stored in cifar10resnet/figs/. 
# For each image, the left-hand side is the original example and 
# the right-hand side is its perturbed version.
```

See `main.py` and `hsja.py` for details. 
## Citation
If you use this code for your research, please cite our [paper](https://arxiv.org/abs/1904.02144):
```
@article{chen2019boundary,
  title={HopSkipJumpAttack: A Query-Efficient Decision-Based Adversarial Attack},
  author={Chen, Jianbo and Jordan, Michael I. and Wainwright, Martin J.},
  journal={arXiv preprint arXiv:1904.02144},
  year={2019}
}
```
