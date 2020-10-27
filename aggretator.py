def aggregate_grads(grads, backend):
    """Aggregate model gradients to models.

    Args:
        data: a list of grads' information
            item format:
                {
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
    for gradinfo in grads:  # each model's gradinfo
        n_samples = gradinfo['n_samples']
        for k, v in gradinfo['named_grads'].items():  # a model's named_grads, k is the layer's name, v is the variables
            if k not in total_grads:
                total_grads[k] = []

            total_grads[k].append(v * n_samples)
        n_total_samples += n_samples

    gradients = {}
    for k, v in total_grads.items():
        gradients[k] = backend.sum(v, dim=0) / n_total_samples

    return gradients
