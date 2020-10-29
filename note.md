# paper reading
- **[Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295.pdf)**
  >这篇文章证明了传统的FEDAVG算法等同于对于聚合后的伪梯度(pseudo-gredient)进行SGD下降，因此提出**并不直接采用聚合后的梯度更新网络**，而是在此基础上在服务器上采用诸如ADAM等优化算法来更新网络。此外，文章还提出两点结论:  
  >1. clients上面的模型更新学习率最好使用衰减策略;  
  >2. server上面的学习率与clients上面的学习率应该成反比; 