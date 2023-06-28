# Predict
import tqdm
import torch
import numpy as np
def predict(test_loader, model,config):
    model.eval()
    step_preds = []
    test_loop = tqdm(test_loader,position=0,ncols=70,leave=False,desc=f'Testing')
    for data, label, e in test_loop:
        data = {key: value.to(config.device) for key, value in data.items()}
        label = label.to(config.device)
        e = e.to(config.device)
        with torch.no_grad():
            output = model(data, e)
            preds = output.argmax(dim=1)
            step_preds.extend(preds.cpu().data.numpy())
    config.logger.info("Test | Finish!")
    print("===FInish Test===")
    with open('') as f:
        pass
    return step_preds
