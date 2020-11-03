def aggregate_grads(grads, backend):
    """Aggregate model gradients to models.

    Args:
        data: a list of grads' information(dict stracture)
            item format:
                {
                    'label_nums': xxx,
                    'n_samples': xxx,
                    'named_grads': xxx,
                }
    Return:
        named grads: {
            'layer_name1': grads1,
            'layer_name2': grads2,
            ...
        }
    """
    total_grads = {}
    n_total_samples = 0  # all models' samples' num
    n_total_label_nums = 0  # all models' labels' num
    '''
    for i in range(len(grads)):
        print('user', i,' label_num ', grads[i]['label_nums'])
    '''
    for gradinfo in grads:  # each model's gradinfo
        n_label_nums = gradinfo['label_nums']
        n_samples = gradinfo['n_samples']
        for k, v in gradinfo['named_grads'].items():  # a model's named_grads, k is the layer's name, v is the variables
            if k not in total_grads:
                total_grads[k] = []
            total_grads[k].append(v * n_samples)
            # total_grads[k].append(v * n_samples * n_label_nums)  # weighting the grads from users i according to its samples amount and label nums
        # n_total_label_nums += n_label_nums
        n_total_samples += n_samples
        # n_total_samples += n_samples * n_label_nums
    gradients = {}
    for k, v in total_grads.items():
        # gradients[k] = backend.sum(v, dim=0) / (n_total_samples * n_total_label_nums)  # is equivalent to SGD for each sample
        gradients[k] = backend.sum(v, dim=0) / (n_total_samples)
    return gradients
