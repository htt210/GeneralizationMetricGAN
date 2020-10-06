import yaml


def load_configs(path):
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.CLoader)
    return config


def update_config_kv(config, *params):
    """
    update a config recursively: config[key1][key2][...] = value

    :param config: a dictionary of config
    :param params: array of the form [key1, key2, ..., keyn, value]
    :return: None
    """

    print(params)
    print(config)
    print()
    if len(params) < 2:
        raise AttributeError('params should contain keys and a value')
    if len(params) == 2:
        config[params[0]] = params[1]
    else:
        update_config_kv(config[params[0]], *params[1:])


def update_config(base_config, new_config):
    for k, v in new_config.items():
        bv = base_config.get(k, None)
        if bv is not None:  # only updating, not adding new keys
            if isinstance(bv, dict):
                update_config(bv, v)
            else:
                base_config[k] = v


# if __name__ == '__main__':
#     config = {}
#     config[0] = {}
#     config[0][1] = 'a'
#
#     update_configs(config, *[0, 1, 'b'])
#     print(config)
