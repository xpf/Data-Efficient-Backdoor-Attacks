# Data-Efficient-Backdoor-Attacks (PyTorch)

[Data Efficient Backdoor Attacks]()

Pengfei Xia, Ziqiang Li, Wei Zhang, and Bin Li, *International Joint Conferences on Artificial Intelligence*, 2022.

>Abstract: *Recent studies have proven that deep neural networks are vulnerable to backdoor attacks. Specifically, by mixing a small number of poisoned samples into the training set, the behavior of the trained model can be maliciously controlled. Existing attack methods construct such adversaries by randomly selecting some clean data from the benign set and then embedding a trigger into them. However, this selection strategy ignores the fact that each poisoned sample contributes inequally to the backdoor injection, which reduces the efficiency of poisoning. In this paper, we formulate improving the poisoned data efficiency by the selection as an optimization problem and propose a Filtering-andUpdating Strategy (FUS) to solve it. The experimental results on CIFAR-10 and ImageNet-10 indicate that the proposed method is effective: the same attack success rate can be achieved with only 47% to 75% of the poisoned sample volume compared to the random selection strategy. More importantly, the adversaries selected according to one setting can generalize well to other settings, exhibiting strong transferability.*

## Searching

```python
# Use random selection to create 0.01 * 50000 = 500 poisoned samples for CIFAR-10
python search.py --data_path your_path --data_name cifar10 --ratio 0.01 --n_iter 0

# Use FUS to create 0.01 * 50000 = 500 poisoned samples for CIFAR-10
python search.py --data_path your_path --data_name cifar10 --ratio 0.01 --n_iter 10 --alpha 0.5
```

### Transfering

```python
# Use random selected poisoned samples for training a new model to
python transfer.py --data_path your_path --data_name cifar10 --ratio 0.01 --n_iter 0

# Use filtering-and-updating strategy to create 0.01 * 50000 = 500 poisoned samples for CIFAR-10
python search.py --data_path your_path --data_name cifar10 --ratio 0.01 --n_iter 10 --alpha 0.5
```
