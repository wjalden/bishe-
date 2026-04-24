from src.models.lse_hf_lt import LSEHFLTModel


class PaperAliasModel(LSEHFLTModel):
    """Alias wrapper used to keep unified runner while filling paper-specific logic incrementally."""

    def __init__(self, model_name: str, num_labels: int, paper_tag: str):
        super().__init__(model_name=model_name, num_labels=num_labels)
        self.paper_tag = paper_tag
