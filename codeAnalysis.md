---
title: 实验代码解析文档
---

鉴于代码结构并不很简单，写一个文档将每个py文件的内容进行整理，对模块的接口进行说明：

# 2020/10.30

## aggretator.py:

### def aggregate_grads(grads, backend):

实现梯度聚合

> 输入名：grads
> 解释：多个模型的梯度参数
> 格式：
>
> ```
> data: a list of grads' information(dict stracture)
>     item format:
>         {
>             'n_samples': xxx,
>             'named_grads': xxx,
>         }
> ```
>
> ---
>
> 输入名：backend
> 解释：操作后端，提供一些操作
> 格式：NumpyBackend / PytorchBackend，详见之后的解释

---

> 输出名： gradients
> 解释：聚合梯度
> 格式：
>
> ```
> Return:
>     named grads: {
>         'layer_name1': grads1,
>         'layer_name2': grads2,
>         ...
>     }
> ```

## context.py



