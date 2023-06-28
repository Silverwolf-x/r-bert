# data_load
import torch
from torch.utils.data import Dataset
import re

def label_coder(length = False):
    """encoder-decoder for label
    - `length=True` to return classifier_length,
    default to return `num_to_label`, `label_to_num`
    """
    num_to_label=['Cause-Effect(e1,e2)',
    'Cause-Effect(e2,e1)',
    'Component-Whole(e1,e2)',
    'Component-Whole(e2,e1)',
    'Content-Container(e1,e2)',
    'Content-Container(e2,e1)',
    'Entity-Destination(e1,e2)',
    'Entity-Destination(e2,e1)',
    'Entity-Origin(e1,e2)',
    'Entity-Origin(e2,e1)',
    'Instrument-Agency(e1,e2)',
    'Instrument-Agency(e2,e1)',
    'Member-Collection(e1,e2)',
    'Member-Collection(e2,e1)',
    'Message-Topic(e1,e2)',
    'Message-Topic(e2,e1)',
    'Product-Producer(e1,e2)',
    'Product-Producer(e2,e1)',
    'Other']
    label_to_num={label:num for num, label in enumerate(num_to_label)}
    return len(num_to_label) if length else (num_to_label, label_to_num)

def clean_data(txtfile,pretry=None):
    """get data needed for training
    - `txtfile`: original data file
    - `pretry`: data to have a test for coding
    - return `{'data','label'}`"""
    with open(txtfile,'r') as f:
        lines=f.readlines()
    data,label=[],[]
    # each 4 is a group
    if pretry:
        lines=lines[:pretry*4]
    for i,line in enumerate(lines):
        line=line.strip()
        # replace the special tokens into which are used in the paper
        if i%4==0:
            _,line_data=line.split('\t')
            line_data=(line_data
                .replace('<e1>','$ ')
                .replace('</e1>',' $')
                .replace('<e2>','# ')
                .replace('</e2>',' # ')
                .replace('"','')
            )
            data.append(line_data)
        if i%4==1:
            label.append(line)
    _,label_to_num=label_coder()
    label=list(map(lambda x: label_to_num[x], label))
    return {'data':data,'label':label}

class TextDataset(Dataset):
    """
    remember to process data when initialize, otherwise 
    - return `{'input_ids','token_type_ids','attention_mask'}`,  `label`, `e` with tensor"""

    def find_pos(self, input_id, text,e = '$' or '#'):
        """
        use regular expression to locate the entity after tokenization
        - e only accepts '$' or '#'"""
        assert e in ('$','#')
        if e == '$':
            match = re.search(r'\$ ([\w\-]+(?: [\w\-]+)*) \$', text).group()
        else :
            match = re.search(r'\# ([\w\-]+(?: [\w\-]+)*) \#', text).group()
        encoded_match = self.tokenizer.encode(match, add_special_tokens=False)

        n = len(encoded_match)
        start_pos, end_pos = -1, -1
        for pos in range(len(input_id) - n + 1):
            # input_id is a tensor
            if input_id[pos:pos+n].tolist() == encoded_match:
                start_pos = pos
                end_pos = start_pos + n - 1
                break
        result= [start_pos, end_pos]
        return torch.tensor(result,dtype=torch.int)

    def __init__(self,config,tokenizer,data,label):
        super(TextDataset).__init__()
        self.tokenizer = tokenizer 
        # token during __init__ to accelerate training
        # __getitem__ will waste time for unnessary calculation
        tokens = tokenizer(data
            ,padding="max_length"
            ,max_length=config.max_length
            ,return_tensors="pt")
        batch_input_ids = tokens['input_ids'].unbind(dim=0)
        e1_find = map(lambda x, y: self.find_pos(x, y, e='$'), batch_input_ids, data)
        e1 = torch.stack(list(e1_find)) 
        e2_find = map(lambda x, y: self.find_pos(x, y, e='#'), batch_input_ids, data)
        e2 = torch.stack(list(e2_find)) 

        self.tokens = tokens
        self.e = torch.cat((e1,e2), dim=1)
        self.label = torch.tensor(label,dtype=torch.int)

    def __len__(self):
        return len(self.label)
    def __getitem__(self,idx):
        input_ids = self.tokens['input_ids'][idx] # [max_length, ]
        token_type_ids = self.tokens['token_type_ids'][idx]
        attention_mask = self.tokens['attention_mask'][idx]
        e = self.e[idx]
        label= self.label[idx]
        return {'input_ids':input_ids,'token_type_ids':token_type_ids,'attention_mask':attention_mask},label,e
