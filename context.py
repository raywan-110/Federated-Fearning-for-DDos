from abc import ABC
from abc import abstractmethod
import os

import numpy as np
import torch

from aggretator import aggregate_grads

# 产生 2n或者2n-1 位的随机16进制数的字符串
def random_str(n):
    return hex(int.from_bytes(os.urandom(n), byteorder='big'))[2:]

# 基础类，被PytorchModel继承
class ModelBase(ABC):
    def __init__(self, **kwargs):
        for k in kwargs:
            setattr(self, k, kwargs[k])  # setattr 对属性k赋值

    @abstractmethod
    def update_grads(self, grads):
        pass

    @abstractmethod
    def load_model(self, path):
        pass

    @abstractmethod
    def save_model(self, path):
        pass

# 继承自ModelBase,有
class PytorchModel(ModelBase):
    def __init__(self,
                 torch,
                 model_class,
                 init_model_path: str = '',  # 变量注释，没啥用处
                 lr: float = 0.001,
                 optim_name: str = 'Adam',
                 cuda: bool = False):
        """Pytorch 封装.

        参数：
            torch: torch 库
            model_class: 训练模型类
            init_model_path: 初始模型路径
            lr: 学习率
            optim_name: 优化器类名称
            cuda: 是否需要使用cuda
        """

        self.torch = torch
        self.model_class = model_class
        self.init_model_path = init_model_path
        self.lr = lr
        self.optim_name = optim_name
        self.cuda = cuda

        self._init_params()

    def _init_params(self):
        self.model = self.model_class()
        if self.init_model_path:
            self.model.load_state_dict(self.torch.load(self.init_model_path))

        if self.cuda and self.torch.cuda.is_available():
            self.model = self.model.cuda()

        self.optimizer = getattr(self.torch.optim,
                                 self.optim_name)(self.model.parameters(),
                                                  lr=self.lr)

    def update_grads(self, grads):  # method1: update the model via aggregate_grads
        self.optimizer.zero_grad()

        for k, v in self.model.named_parameters():
            v.grad = grads[k].type(v.dtype)

        self.optimizer.step()

    def update_params(self, params):  # method2: update the model via gived parameters

        for k, v in self.model.named_parameters():
            v[:] = params[k]

        return self.model

    def load_model(self, path, force_reload=False):
        if force_reload is False and self.load_from_path == path:
            return

        self.load_from_path = path
        self.model.load_static_dict(self.torch.load(path))

    def save_model(self, path):
        base = os.path.dirname(path)
        if not os.path.exists(base):
            os.makedirs(base)

        self.torch.save(self.model.state_dict(), path)

        return path


class BaseBackend(ABC):
    @abstractmethod
    def mean(self, data):
        data = np.array(data)

        return data.mean(axis=0)  # get mean each column


class NumpyBackend(BaseBackend):
    def mean(self, data):
        return super().mean(data=data)


class PytorchBackend(BaseBackend):
    """ define many methods like a kit
        init: input (torch, cuda=False) no return
        mean: input (data, dim=0) return tensor.mean(dim)
        sum:  input (data, dim=0) return tensor.sum(dim)
        update_grads: input (model, grads) no return
        update_params: input (model, params) return the updated model
        load_model: input (model, path, forced_reloaded=False) no return
        save_model: input (model, path) return path
    """
    def __init__(self, torch, cuda=False):
        self.torch = torch
        if cuda:
            if self.torch.cuda.is_available():
                self.cuda = True
        else:
            self.cuda = False

    def mean(self, data, dim=0):
        return self.torch.tensor(
            data,
            device=self.torch.cuda.current_device() if self.cuda else None,
        ).mean(dim=dim)  # dim (int) – the dimension to reduce

    def sum(self, data, dim=0):
        return self.torch.tensor(
            data,
            device=self.torch.cuda.current_device() if self.cuda else None,
        ).sum(dim=dim)

    def _check_model(self, model):
        if not isinstance(model, PytorchModel):
            raise ValueError(
                "model must be type of PytorchModel not {}".format(
                    type(model)))

    def update_grads(self, model, grads):
        self._check_model(model=model)
        return model.update_grads(grads=grads)

    def update_params(self, model, params):
        self._check_model(model=model)
        return model.update_params(params=params)

    def load_model(self, model, path, force_reload=False):
        self._check_model(model=model)
        return model.load_model(path=path, force_reload=force_reload)

    def save_model(self, model, path):
        self._check_model(model=model)
        return model.save_model(path)


class Aggregator(object):
    def __init__(self, model, backend):
        self.model = model  # the model
        self.backend = backend  # the backend to control


class FederatedAveragingGrads(Aggregator):
    def __init__(self, model, framework=None):
        self.framework = framework or getattr(model, 'framework')

        if framework is None or framework == 'numpy':
            backend = NumpyBackend
        elif framework == 'pytorch':
            backend = PytorchBackend(torch=torch)
        else:
            raise ValueError(
                'Framework {} is not supported!'.format(framework))

        super().__init__(model, backend)

    def aggregate_grads(self, grads, leader=-1):
        """Aggregate model gradients to models.

        Args:
            data: a list of grads' information
                item format:
                    {
                        'n_samples': xxx,
                        'named_grads': xxx,
                    }
        """
        bias = [1 for n in range(len(grads))] # for each member has differ bias
        if leader != -1:
            bias[leader] = int(len(grads))
        self.backend.update_grads(self.model,
                                  grads=aggregate_grads(grads=grads,
                                                        backend=self.backend,bias=bias))

    def save_model(self, path):
        return self.backend.save_model(self.model, path=path)

    def load_model(self, path, force_reload=False):
        return self.backend.load_model(self.model,
                                       path=path,
                                       force_reload=force_reload)

    def __call__(self, grads, leader=-1):
        """Aggregate grads.

        Args:
            grads -> list: grads is a list of either the actual grad info
            or the absolute file path  of grad info.
        """
        if not grads:
            return

        if not isinstance(grads, list):
            raise ValueError('grads should be a list, not {}'.format(
                type(grads)))

        actual_grads = grads

        return self.aggregate_grads(grads=actual_grads,leader=leader)
