import torch
import torch.nn.functional as F
import numpy as np
from typing import Set
import random
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer, util
from munch import Munch

INSTRUCTIONS = [
    "Please provide a reinterpretation of the preceding table. \n[TABLE]\n```\n",
    "Could you give me a different version of the table above? \n[TABLE]\n```\n",
    "After uppacking the table above, we got: \n[TABLE]\n```\n",
    "Please offer a restatement of the table I've just read. \n[TABLE]\n```\n"
]
class SamplePreprocessor:
    def __init__(
        self,
        tokenizer,
        beacon_size: int = 1,
        max_length: int = 4096,
    ):
        self.tokenizer = tokenizer
        self.beacon_size = beacon_size
        self.max_length = max_length
        self.beacon_token = self.tokenizer.eos_token
        self.beacon_token_id = self.tokenizer.convert_tokens_to_ids(self.beacon_token)

        # self.max_length = max_length
        # self.dataset_type = dataset_type
        # if dataset_type.upper() not in dir(DatasetType):
        #     raise ValueError(f"Unsupported dataset type: {dataset_type}")


    def __call__(self, sample, **kwargs):

        df = sample['df'] # pd.DataFrame

        df_aug = df.map(lambda x: str(x) + self.beacon_token * self.beacon_size)
        rows = df_aug.values.tolist()

        # prefix + header
        input_ids = self.tokenizer.encode("[TABLE]\n```\n", add_special_tokens=False)
        header_ids = self.tokenizer.encode(','.join(df.columns.to_list())+"\n", add_special_tokens=False)
        input_ids.extend(header_ids)
        segment_ids = [0] * len(input_ids)
        is_beacon = [0] * len(input_ids)

        # table
        length = len(input_ids)
        row_nums = 0
        for row_idx, row in enumerate(rows):
            row_input_ids, row_segment_ids, row_is_beacon = self.tokenize_row(row, row_idx)
            if length + len(row_input_ids) > self.max_length:
                break
            input_ids.extend(row_input_ids)
            segment_ids.extend(row_segment_ids)
            is_beacon.extend(row_is_beacon)
            length += len(row_input_ids)
            row_nums += 1

        # postfix
        postfix_ids = self.tokenizer.encode("```\n" + random.choice(INSTRUCTIONS), add_special_tokens=False)
        input_ids.extend(postfix_ids)
        segment_ids.extend([0] * len(postfix_ids))
        is_beacon.extend([0] * len(postfix_ids))

        # labels
        labels = df.iloc[:row_nums].to_csv(index=False, header=True)
        label_ids = self.tokenizer.encode(labels, add_special_tokens=False)
        input_ids.extend(label_ids)
        segment_ids.extend([0] * len(label_ids))
        is_beacon.extend([0] * len(label_ids))

        # attention_mask & position_ids
        # attention_mask = self.make_segment_mask(torch.tensor(segment_ids, dtype=torch.int64).unsqueeze(0), torch.tensor(is_beacon, dtype=torch.int64).unsqueeze(0))

        del sample['df']
        sample['input_ids'] = input_ids
        sample['segment_ids'] = segment_ids
        sample['is_beacon'] = is_beacon
        # sample['attention_mask'] = attention_mask.tolist()
        sample['label_ids'] = label_ids
        return sample
    
    def tokenize_row(self, row, row_idx):
        """
        Tokenize a single row of the DataFrame.
        """
        row_input_ids = []
        row_segment_ids = []
        row_is_beacon = []
        for idx, cell in enumerate(row):
            # Tokenize each cell, add a comma if not the last cell
            if idx == len(row) - 1:
                cell_text = str(cell) + ","
                cell_ids = self.tokenizer.encode(cell_text, add_special_tokens=False)
                row_is_beacon.extend([0] * (len(cell_ids) - 1 - self.beacon_size) + [1] * self.beacon_size + [0])
            else:
                cell_text = str(cell)
                cell_ids = self.tokenizer.encode(cell_text, add_special_tokens=False)
                row_is_beacon.extend([0] * (len(cell_ids) - self.beacon_size) + [1] * self.beacon_size)
            row_input_ids.extend(cell_ids)
            row_segment_ids.extend([row_idx] * len(cell_ids))
        return row_input_ids, row_segment_ids, row_is_beacon

class CHADataCollator:

    def make_segment_mask(self, segment_ids: torch.Tensor, is_beacon: torch.Tensor, dtype=torch.bfloat16) -> torch.Tensor:
        """
        segment_ids: LongTensor[B, L] - >0 表示属于某段；==0 表示 global super
        is_beacon:   BoolTensor[B, L] - 标记哪些 token 是 beacon
        return: attention bias mask [B, L, L]
        """
        if segment_ids.dim() == 1:
            segment_ids = segment_ids.unsqueeze(0)

        B, L = segment_ids.shape
        device = segment_ids.device
        MINF = float("-inf")

        is_beacon = is_beacon.bool()
        is_global_super = segment_ids == 0
        is_super = is_beacon | is_global_super
        is_token = (segment_ids > 0) & (~is_beacon)

        # === broadcast indices
        i = torch.arange(L, device=device).view(1, L, 1)
        j = torch.arange(L, device=device).view(1, 1, L)

        seg_i = segment_ids.unsqueeze(2)  # [B, L, 1]
        seg_j = segment_ids.unsqueeze(1)  # [B, 1, L]
        same_seg = (seg_i == seg_j) & (seg_i > 0)

        is_token_i = is_token.unsqueeze(2)
        is_token_j = is_token.unsqueeze(1)
        is_beacon_i = is_beacon.unsqueeze(2)
        is_beacon_j = is_beacon.unsqueeze(1)
        is_super_i = is_super.unsqueeze(2)
        is_super_j = is_super.unsqueeze(1)

        # === Rule definitions ===

        # 1. 普通 token：看段内过去 token（普通+beacon），看之前 super token（beacon+global）
        token_to_seg = is_token_i & same_seg & (j <= i) & (is_token_j | is_beacon_j)
        token_to_super = is_token_i & is_super_j & (j < i)

        # 2. beacon token：看段内过去 token（普通+beacon），看之前 super token（不包括自己后面的）
        beacon_to_seg = is_beacon_i & same_seg & (j <= i) & (is_token_j | is_beacon_j)
        beacon_to_super = is_beacon_i & is_super_j & (j < i)

        # 3. global super token：看之前所有 super token（包括 beacon）（causal）
        super_to_super = is_global_super.unsqueeze(2) & is_super_j & (j <= i)

        # === Final visibility mask
        visible = (
            token_to_seg |
            token_to_super |
            beacon_to_seg |
            beacon_to_super |
            super_to_super
        )

        # === Attention bias
        attn_bias = torch.full((B, L, L), MINF, dtype=dtype, device=device)
        attn_bias.masked_fill_(visible, 0.0)
        return attn_bias

    def __call__(self, batch):
        """
        batch is a batch of samples. Now we only support batch size = 1
        """
        if len(batch) > 1:
            raise ValueError("Now we can only support batch_size equal to 1 !!!")
        
        sample = batch[0]
        
        input_ids = torch.LongTensor(sample['input_ids']).unsqueeze(0)
        segment_ids = torch.LongTensor(sample['segment_ids']).unsqueeze(0)  # (1, L)
        is_beacon = torch.LongTensor(sample['is_beacon']).unsqueeze(0)  # (1, L)
        attention_mask = self.make_segment_mask(segment_ids, is_beacon).unsqueeze(0)  # (1, L, L)
        position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64).unsqueeze(0)
        label_ids = torch.LongTensor(sample['label_ids']).unsqueeze(0)  # (1, L)
        # print(f"Input IDs shape: {input_ids.shape}, Attention mask shape: {attention_mask.shape}, Position IDs shape: {position_ids.shape}, Label IDs shape: {label_ids.shape}, Is beacon shape: {is_beacon.shape}")

        return Munch({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'label_ids': label_ids,
            "is_beacon": is_beacon,
        })

# class CompressionStrategy:
#     RANDOM = "random"
#     PREDEFINED = "predefined"
#     DETERMINISTIC = "deterministic"

# class SHADataCollator:
#     def __init__(self,
#                 ) -> None:
        
#     def __call__(self, batch):
#         """
#         batch is a batch of samples. Now we only support batch size = 1
#         """
#         # TODO: add eval mode
#         if len(batch) > 1:
#             raise ValueError("Now we can only support batch_size equal to 1 !!!")
#         sample = batch[0]
#         sample = self.sample_processor(sample)

#         segments = sample['segments']
#         reference_sent = sample.get('reference_sent', None)
#         instruction = sample.get('instruction', None)
#         answer = sample.get('answer', None)
#         question = sample.get("question", "\nQuestion: " + reference_sent + "\nAnswer: " if reference_sent != instruction else "\nSummary: ")
#         budgets = sample.get('budgets', None)
    
        
#         # TODO
#         task = TaskType.FINETUNE if reference_sent else TaskType.PRETRAIN

#         if task == TaskType.FINETUNE:
#             input_ids = self.lm_model_tokenizer.encode(instruction, add_special_tokens=False)
#         else:
#             input_ids = []

#         for idx, sent in enumerate(segments):
#             sent_ids = self.lm_model_tokenizer.encode(sent+' ', add_special_tokens=False)
#             input_ids.extend(sent_ids)

#         if task == TaskType.FINETUNE:
#             input_ids.extend(self.lm_model_tokenizer.encode(question + answer, add_special_tokens=False))

#         input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)  # (1, L_total)

#         if task == TaskType.FINETUNE:
#             label_ids = self.lm_model_tokenizer.encode(answer, add_special_tokens=False) + [self.lm_model_tokenizer.eos_token_id]
#             label_ids = torch.tensor(label_ids, dtype=torch.long).unsqueeze(0)
#         elif task == TaskType.PRETRAIN:
#             label_ids = input_ids.clone()
    
#         attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
#         position_ids = torch.arange(input_ids.shape[1], dtype=torch.int64, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)

#         return Munch({
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "position_ids": position_ids,
#             "label_ids": label_ids,
#         })