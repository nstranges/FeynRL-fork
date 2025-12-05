import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class GeneralDataLoader(Dataset):
    '''
        This is a general dataloader that works for any source data that has the following formant:
        {
            "prompt": "this is a prompt",
            "answer": "this is an answer",
            ...
        }
        The data should be in a parquet format.
    '''
    def __init__(self, 
                prompt_key,
                answer_key, 
                max_seq_len,
                tokenizer=None, 
                data_path="", 
                ):
        assert prompt_key != "", "prompt_key cannot be empty"
        assert answer_key != "", "answer_key cannot be empty"
        assert max_seq_len > 0, "max_seq_len must be greater than 0"
        assert tokenizer is not None, "tokenizer cannot be None"
        assert data_path != "", "data_path cannot be empty"

        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.data_path = data_path

    def __getitem__(self, idx):
        text = " this is a dummy text to test our dataloader"
        tokenized_text = self.tokenizer(
                                        text,
                                        return_tensors="pt",
                                        padding="max_length",
                                        truncation=True,
                                        max_length=self.max_seq_len,
                                    )
        return tokenized_text

    def __len__(self):
        return self.len_data