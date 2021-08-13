def map_nested_dicts(d, func, k=None):
    if isinstance(d, dict):
        return {k: map_nested_dicts(v, func, k) for k, v in d.items()}
    else:
        return func(k, d)