from datetime import datetime
import os
import shutil
import unittest

import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn.functional as F

from context import FederatedAveragingGrads
from context import PytorchModel
from learning_model import FLModel
from preprocess import get_test_loader
from preprocess import UserRoundData
from train import user_round_train
from platform import system

class ParameterServer(object):
    def __init__(self, init_model_path, testworkdir, lr=0.001):
        self.round = 0
        self.rounds_info = {}
        self.rounds_model_path = {}  # path to the dir of model
        self.current_round_grads = []  # a list to store grads
        self.init_model_path = init_model_path
        self.model = PytorchModel(torch=torch,
                               model_class=FLModel,
                               lr=lr,
                               init_model_path=self.init_model_path,
                               optim_name='Adam')
        self.aggr = FederatedAveragingGrads(
            model=self.model,
            framework='pytorch',
        )

        self.testworkdir = testworkdir
        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

    def change_lr(self, lr):
        self.model.lr = lr

    def get_latest_model(self):
        if not self.rounds_model_path:
            return self.init_model_path

        if self.round in self.rounds_model_path:
            return self.rounds_model_path[self.round]

        return self.rounds_model_path[self.round - 1]

    def receive_grads_info(self, grads):
        self.current_round_grads.append(grads)  # grads from each node

    def aggregate(self, leader=-1):
        if leader != -1:  # leader exist
            self.aggr(self.current_round_grads)  # update the model in server
            self.round += 1
            self.current_round_grads = []  # clear the grads buffer in server
        else:  # need to select a leader
            self.aggr(self.current_round_grads)
            self.current_round_grads = []
        # define the path where the model saved to
        path = os.path.join(self.testworkdir,
                            'round-{round}-model.md'.format(round=self.round))
        self.save_model(path) # save model to a file for members to get

    def save_model(self, path):
        # update the info: the updated model's path
        self.rounds_model_path[self.round] = path
        # remove the outdated model
        if (self.round - 1) in self.rounds_model_path:
            if os.path.exists(self.rounds_model_path[self.round - 1]):
                os.remove(self.rounds_model_path[self.round - 1])

        info = self.aggr.save_model(path=path)

        return info


class FedAveragingGradsTestSuit(unittest.TestCase):
    RESULT_DIR = 'result'
    N_VALIDATION = 10000
    if system() == 'Windows':  # for local test
        TEST_BASE_DIR = './tmp/'
    else:                      # for cloud platform to test
        TEST_BASE_DIR = '/tmp/'

    def setUp(self):
        self.seed = 0
        self.use_cuda = False
        self.batch_size = 1600
        self.test_batch_size = 1000
        self.lr = 0.01
        self.n_max_rounds = 10000
        self.log_interval = 10
        self.n_round_samples = 1600
        self.testbase = self.TEST_BASE_DIR
        self.testworkdir = os.path.join(self.testbase, 'competetion-test')  # set all the hyperparameters

        if not os.path.exists(self.testworkdir):
            os.makedirs(self.testworkdir)

        self.init_model_path = os.path.join(self.testworkdir, 'init_model.md')
        torch.manual_seed(self.seed)

        if not os.path.exists(self.init_model_path):
            torch.save(FLModel().state_dict(), self.init_model_path)

        self.ps = ParameterServer(init_model_path=self.init_model_path,
                                  testworkdir=self.testworkdir, lr=self.lr)

        if not os.path.exists(self.RESULT_DIR):
            os.makedirs(self.RESULT_DIR)

        self.urd = UserRoundData()
        self.n_users = self.urd.n_users

    def _clear(self):
        shutil.rmtree(self.testworkdir)

    def tearDown(self):
        self._clear()

    def test_federated_averaging(self):
        torch.manual_seed(self.seed)
        device = torch.device("cuda" if self.use_cuda else "cpu")

        training_start = datetime.now()
        bestModel = FLModel()
        highestACC = 0
        leader = self.select_Leader(10,device)
        anger = 0        # to calculate how many round there is no improvement with current leader
        for r in range(1, self.n_max_rounds + 1):
            start = datetime.now()
            for u in range(0, self.n_users):
                path = self.ps.get_latest_model()
                model = FLModel()
                model.load_state_dict(torch.load(path))
                model = model.to(device)
                x, y = self.urd.round_data(
                    user_idx=u,
                    n_round=r,
                    n_round_samples=self.n_round_samples,
                )
                grads = user_round_train(X=x, Y=y, model=model, device=device, batch_size=self.batch_size)  # obtain grads from each user
                self.ps.receive_grads_info(grads=grads)

            self.ps.aggregate(leader=leader)  # aggregate the gradients and update the model
            print('\nRound {} cost: {}, total training cost: {}'.format(
                r,
                datetime.now() - start,
                datetime.now() - training_start,
            ))

            # test the model and select the best
            if r % 5 == 0:
                path = self.ps.get_latest_model()
                model = FLModel()
                model.load_state_dict(torch.load(path))
                model = model.to(device)
                _, acc = self.predict(model,
                             device,
                             self.urd.uniform_random_loader(self.N_VALIDATION),
                             prefix="Train")
                if acc > highestACC:  # select the better model
                    bestModel = model
                    highestACC = acc  # update the limit
                    anger -= 1
                    print('****** best model update *******')
                else:
                    anger += 1
                print(f'anger: {anger}')
                if system() != 'Windows':  # for platform test
                    self.save_testdata_prediction(model=model, device=device)
                if anger > 5:   # the lead is too junk and should be replaced
                    leader = self.select_Leader(10,device,last_leader=leader,ACC=acc) # exchange the leader
                    anger = 0

        model = bestModel()
        if model is not None:
            self.save_testdata_prediction(model=model, device=device)

    def select_Leader(self, probation, device, last_leader=0, ACC=0.0):
        HACC = ACC
        delta = 0
        leader = last_leader
        for u in range(0, self.n_users):
            for r in range(1, probation + 1):
                path = self.ps.get_latest_model()
                model = FLModel()
                model.load_state_dict(torch.load(path))
                model = model.to(device)
                x, y = self.urd.round_data(
                    user_idx=u,
                    n_round=r,
                    n_round_samples=self.n_round_samples,
                )
                grads = user_round_train(X=x, Y=y, model=model, device=device, batch_size=self.batch_size)  # obtain grads from each user
                self.ps.receive_grads_info(grads=grads)
                self.ps.aggregate()  # aggregate the gradients and update the model
            path = self.ps.get_latest_model()
            model = FLModel()
            model.load_state_dict(torch.load(path))
            model = model.to(device)
            print(f"******** selecting Leader: member {u} **********")
            _, acc = self.predict(model,
                                  device,
                                  self.urd.uniform_random_loader(self.N_VALIDATION),
                                  prefix="Train")
            if acc-HACC > delta:
                if u != last_leader:
                    leader = u
                delta = acc - HACC
        print(f"*********** new leader is member {leader} *********** ")
        return leader

    def save_prediction(self, predition):
        if isinstance(predition, (np.ndarray, )):
            predition = predition.reshape(-1).tolist()

        with open(os.path.join(self.RESULT_DIR, 'result.txt'), 'w') as fout:
            fout.writelines(os.linesep.join([str(n) for n in predition]))

    def save_testdata_prediction(self, model, device):
        loader = get_test_loader(batch_size=1000)
        prediction = []
        with torch.no_grad():
            for data in loader:
                pred = model(data.to(device)).argmax(dim=1, keepdim=True)
                prediction.extend(pred.reshape(-1).tolist())

        self.save_prediction(prediction)

    def predict(self, model, device, test_loader, prefix=""):
        model.eval()
        test_loss = 0
        correct = 0
        prediction = []
        real = []
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                target = target.long()
                output = model(data)
                test_loss += F.nll_loss(
                    output, target,
                    reduction='sum').item()  # sum up batch loss
                pred = output.argmax(
                    dim=1,
                    keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                prediction.extend(pred.reshape(-1).tolist())
                real.extend(target.reshape(-1).tolist())

        test_loss /= len(test_loader.dataset)
        acc = 100. * correct / len(test_loader.dataset)
        print(classification_report(real, prediction))
        print(
            '{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                prefix, test_loss, correct, len(test_loader.dataset), acc), )
        return test_loss, acc


def suite():
    suite = unittest.TestSuite()
    suite.addTest(FedAveragingGradsTestSuit('test_federated_averaging'))
    return suite


def main():
    runner = unittest.TextTestRunner()
    runner.run(suite())


if __name__ == '__main__':
    main()
