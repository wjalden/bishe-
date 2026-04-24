from .lse_hf_lt import LSEHFLTModel
from .papers import PaperAliasModel


def build_model(cfg_model):
    name = cfg_model['name']
    if name == 'lse_hf_lt':
        return LSEHFLTModel(cfg_model['text_encoder'], cfg_model['num_labels'])

    # paper reproduction entries (initial scaffold)
    if name in {'hpt', 'hitin', 'hcl', 'hybrid_embed', 'hb2m'}:
        return PaperAliasModel(cfg_model['text_encoder'], cfg_model['num_labels'], paper_tag=name)

    raise ValueError(f'Unknown model: {name}')
