def get_name(opts, func='search'):
    if func == 'search':
        name = '{}_{}_{}_{}_{}_{}_{}'.format(
            opts.data_name,
            opts.model_name,
            opts.attack_name,
            opts.trigger,
            opts.target,
            opts.ratio,
            opts.n_iter
        )
        if opts.n_iter != 0:
            name = name + '_{}'.format(opts.alpha)
    elif func == 'transfer':
        name = '{}_{}_{}_{}_{}_{}'.format(
            opts.data_name,
            opts.model_name,
            opts.attack_name,
            opts.trigger,
            opts.target,
            opts.samples_idx,
        )
    return name
