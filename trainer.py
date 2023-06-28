# trainer
import torch
from tqdm import tqdm
import numpy as np
import os
import time
import logging

from scorer import my_f1_score

def getLogger():
    """return `logger`, `logger_time`
    `logger_time`: keep the same as save_model_time
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d-%H.%M")
    sHandler = logging.StreamHandler()
    sHandler.setLevel(logging.WARNING)
    sHandler.setFormatter(formatter)
    logger.addHandler(sHandler)

    work_dir = f'./logs/{time.strftime("%Y-%m-%d",time.localtime())}'
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    logger_time = time.strftime("%Y-%m-%d-%H.%M", time.localtime())
    fHandler = logging.FileHandler(os.path.join(work_dir,f'{logger_time}.log'),mode='w')
    fHandler.setLevel(logging.DEBUG)
    fHandler.setFormatter(formatter)
    logger.addHandler(fHandler)

    return logger, logger_time

class record():
    """record training loss and acc for each epoch"""
    def __init__(self) -> None:
        self.train_loss = []
        self.train_acc = []
        self.valid_loss = []
        self.valid_acc = []
        self.save_path = None
        self.score= []
    def save_model(self, model, optimizer, epoch, logger_time):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'next_epoch': epoch + 1
        }
        if not os.path.exists('./run'):
            os.makedirs('./run')
        save_path = f'./run/{logger_time}.pth'
        self.save_path = save_path
        torch.save(checkpoint, save_path)

record=record()

def trainer(train_loader, valid_loader, model, config, checkpoint_path = None):
    """
    - remember `model.to(device)` before passing in trainer
    - return `record()`, which including List Like `train_loss`, `train_acc`, `valid_loss`, `valid_acc`, `score`, `save_path`
    - `score`: macro-F1 in valid data
    - `save_path`: path checkpoint/model for future prediction
    """
    #===prepare===
    config.logger, config.logger_time = getLogger()  # avoid trash logger before training
    logger = config.logger
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_loss = 10e5
    early_stop_count = 0

    #===checkpoint loading===
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['next_epoch']
        if start_epoch >= config.n_epoches:
            print(f'start_epoch {start_epoch} >= n_epoches {config.n_epoches}, hence STOP')
            logger.warning(f'Stop | Wrong loading for checkpoint')
            os.exit()
        logger.info(f"Checkpoint | Finish loading from {checkpoint_path}")
    else:
        start_epoch = 0

    for epoch in range(start_epoch, config.n_epoches):
        #===train mode===
        model.train()
        step_loss = []
        num_correct = 0
        train_loop = tqdm(train_loader, position=0, ncols=70, leave=False)
        for step, (data, label, e) in enumerate(train_loop):
            data = {key: value.to(config.device) for key, value in data.items()}
            label = label.to(config.device)
            e = e.to(config.device)

            output = model(data, e)
            loss = criterion(output, label.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            now_loss = loss.item() / len(train_loader)
            step_loss.append(now_loss)

            preds = output.argmax(dim=1)
            num_correct += torch.eq(preds, label).sum().float().item()
            now_acc = num_correct / len(train_loader.dataset)

            train_loop.set_description(f'Epoch [{epoch}/{config.n_epoches}]')
            train_loop.set_postfix({'loss': f'{now_loss:.2e}'})
            # train_loop.set_postfix({'acc': f'{now_acc:.6f}'})
            # if step % 100 == 0 or step + 1 == len(train_loader):
            if step + 1 == len(train_loader):
                logger.info(
                    f"Train | Epoch: {epoch}\t Step: {step+1}/{len(train_loader)}\t Loss: {now_loss:.2e}\t Acc: {now_acc:.4f}"
                )
        record.train_acc = now_acc
        record.train_loss.append(np.mean(step_loss))

        #===evaluate mode===
        model.eval()
        step_loss, step_preds, step_labels = [], [], []
        num_correct = 0
        valid_loop = tqdm(valid_loader,position=0,ncols=70,leave=False,desc=f'Epoch {epoch} validating')
        for data, label, e in valid_loop:
            data = {key: value.to(config.device) for key, value in data.items()}
            label = label.to(config.device)
            e = e.to(config.device)

            with torch.no_grad():
                output = model(data, e)
                loss = criterion(output, label.long())

                now_loss = loss.item() / len(valid_loader)
                step_loss.append(now_loss)

                preds = output.argmax(dim=1)
                num_correct += torch.eq(preds, label).sum().float().item()
                now_acc = num_correct / len(valid_loader.dataset)

                step_preds.extend(preds.cpu().data.numpy())
                step_labels.extend(label.cpu().data.numpy())

        record.valid_acc.append(now_acc)
        record.valid_loss.append(np.mean(step_loss))
        record.preds = step_preds
        record.labels = step_labels

        score = my_f1_score(step_labels, step_preds)
        record.score.append(score)
        logger.info(
            f"Valid | Loss: {record.valid_loss[-1]:.2e}\t Acc: {record.valid_acc[-1]:.4f}\t F1 Score:{score:.2f}"
        )

        #===early stopp===
        if record.valid_loss[-1] < best_loss:
            best_loss = record.valid_loss[-1]
            early_stop_count = 0
            record.save_model(model, optimizer, epoch, config.logger_time)
            print(
                f'Save model with valid_loss {best_loss:.2e}, valid_acc {record.valid_acc[-1]:.4f} at epoch {epoch}'
            )
        else:
            early_stop_count += 1
        if early_stop_count >= config.early_stop:
            logger.info(
                f"Stop | Epoch: {epoch}\t Model is not improving for {config.early_stop} epoches"
            )
            print(
                f'Model is not improving for {config.early_stop} epoches. The last epoch is {epoch}.'
            )
            break
    record.save_model(model, optimizer, epoch, config.logger_time)
    print("===Finish Training===")
    return record
