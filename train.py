import torch
import torch.nn.functional as F

from preprocess import CompDataset


def user_round_train(user, X, Y, model, device,batch_size=1600, debug=False):
    data = CompDataset(X=X, Y=Y)
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,  # modify it so that every nodes just train one epoch
        shuffle=True,
    )
    label_num = len(set(Y))  # calculate the label numbers of this user
    # print('user:',user,' label numbers:',label_num)
    model.train()  # set the model as train status
    correct = 0
    prediction = []
    real = []
    total_loss = 0
    model = model.to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # import ipdb
        # ipdb.set_trace()
        # print(data.shape, target.shape)
        output = model(data)
        loss = F.nll_loss(output, target.long())
        total_loss += loss
        loss.backward()
        pred = output.argmax(
            dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred).long()).sum().item()
        prediction.extend(pred.reshape(-1).tolist())
        real.extend(target.reshape(-1).tolist())

    grads = {'label_nums': label_num,'n_samples': data.shape[0], 'named_grads': {}}  # add the label_nums to the dict
    for name, param in model.named_parameters():
        grads['named_grads'][name] = param.grad.detach().cpu().numpy()

    if debug:
        print('user:',user,' Training Loss: {:<10.2f}, accuracy: {:<8.2f}'.format(
            total_loss, 100. * correct / len(train_loader.dataset)))

    return grads
