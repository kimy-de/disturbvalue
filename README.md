[![PyPI:dvreg1.0](http://img.shields.io/badge/PyPI-dvreg1.0.-B31B1B.svg)](https://pypi.org/project/dvreg/)
# DisturbValue
*Very easy and efficient regularization technique for any regression tasks*

**DisturbValue** injects noise into a portion of target values at random to alleviate the overfitting problem. The reference paper shows that the method outperforms L2 regularization and Dropout and that the best performance is achieved in more than half the datasets by combining our methods with either L2 regularization or Dropout.

*[Yongho Kim, Hanna Lukashonak, Paweena Tarepakdee, Klavdiia Zavalich, and Mofassir ul Islam Arif (2021) Disturbing Target Values for Neural Network Regularization, arXiv:2110.05003](https://arxiv.org/abs/2110.05003)* 

## Install
`pip install dvreg`

## Hyperparameters
`alpha`: maximum disturbance rate in [0,1] (When alpha=0, there is no disturbance.)

`sigma`: standard deviation to generate Gaussian noise

## Usage
```python
import torch
from disturbvalue import disturbvalue 

target = torch.ones(100,1)
reg = disturbvalue.DisturbValue(alpha=.3, sigma=1e-2)
num_epochs = 1000

for i in range(num_epochs):

    ...
    
    for data, targets in dataloader:
    
        ...
        
        pred = model(data)
        dtargets = reg.dv(targets)
        loss = criterion(pred, dtargets)
        
        ...
        
```
