import os
import torch

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import transformers

from trainer import trainer
from data_load import clean_data,TextDataset,label_coder
from model import RBERT
from scorer import report,cm

class config:
    '''parameter setting，use `print(pd.DataFrame([config.__dict__]))` to see the detail'''
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.seed = 6666

        #-==Important Hyperparameters===
        self.batch_size = 16
        self.early_stop = 5
        self.learning_rate = 2e-5
        self.n_epoches = 5
        self.max_length = 128
        self.dropout_rate = 0.1

    def same_seed(self):
        """fixed the seed to reproduct the training """
        seed=self.seed
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # accelerate
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

def get_pretrain(model_folder='./model/'):
    """
    - download bert_model to model_folder from Huggingface
    - return tokenizer, bert
    """
    transformers.logging.set_verbosity_error()
    if not os.path.exists(model_folder):
        # use tuna mirror to connect and speed up downloading without proxy
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased",mirror='tuna')
        bert = AutoModel.from_pretrained("bert-base-uncased",mirror='tuna')
        tokenizer.save_pretrained(model_folder)
        bert.save_pretrained(model_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_folder,use_fast=True)
    bert = AutoModel.from_pretrained(model_folder)
    return tokenizer, bert


if __name__ == '__main__':

    config = config()
    pretry_mode = 0
    if pretry_mode:
        print('pretry_mode')
        # pretry = 1000
        pretry = None
        config.n_epoches = 1
        # config.device='cpu'
    else:
        pretry = None
    
    config.same_seed()
    print(config.device)

    tokenizer, bert = get_pretrain()

    TRAIN_FILE = 'SemEval2010_task8_all_data/SemEval2010_task8_training/TRAIN_FILE.TXT'
    TEXT_FILE = 'SemEval2010_task8_all_data/SemEval2010_task8_testing_keys/TEST_FILE_FULL.TXT'
    train_data = clean_data(TRAIN_FILE, pretry)
    test_data = clean_data(TEXT_FILE, pretry)

    train_set = TextDataset(config, tokenizer, **train_data)
    test_set = TextDataset(config, tokenizer, **test_data)

    train_loader = DataLoader(train_set,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=0)
    test_loader = DataLoader(test_set,
                             batch_size=config.batch_size,
                             shuffle=False,# both valid and predict
                             pin_memory=True,
                             num_workers=0)

    model = RBERT(label_coder(length=True), config, bert).to(config.device)
    record = trainer(train_loader, test_loader, model, config)  
    print(f'F1-score {record.score[-1]:.4f}')
    config.logger.info(f'Score | Macro-averaged F1-scores (excluding Other): {record.score[-1]:.4f}')
    result_cm = cm(record.labels,record.preds)
    config.logger.info(f'Report | \n{result_cm}')
    result_report = report(record.labels,record.preds)
    config.logger.info(f'Report | \n{result_report}')
