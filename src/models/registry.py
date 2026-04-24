from .lse_hf_lt import LSEHFLTModel


def build_model(cfg_model):
    name = cfg_model['name']
    if name == 'lse_hf_lt':
        return LSEHFLTModel(cfg_model['text_encoder'], cfg_model['num_labels'])
    raise ValueError(f'Unknown model: {name}')
