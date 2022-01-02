import torch
import json
import logging
from utils import*
import pdb

logger = logging.getLogger("my")


def train(args, gpu, model, train_loader, optimizer):
    model.train()
    if gpu==0: logger.info("Train start")
    for iter, batch in enumerate(train_loader):
        optimizer.zero_grad()
        input_ids = batch['input']['input_ids'].to(f'cuda:{gpu}')
        labels = batch['target']['input_ids'].to(f'cuda:{gpu}')
        
        outputs = model(input_ids=input_ids, labels=labels)
        loss =outputs.loss
        loss.backward()
        optimizer.step()
    
        if (iter + 1) % 50 == 0 and gpu==0:
            logger.info('gpu {} step : {}/{} Loss: {:.4f}'.format(
                gpu,
                iter, 
                str(len(train_loader)),
                loss.detach())
            )
        

def valid(args, gpu, model, dev_loader):
    model.eval()
    loss_sum = 0
    logger.info("Validation start")
    with torch.no_grad():
        for iter,batch in enumerate(dev_loader):

            input_ids = batch['input']['input_ids'].to(f'cuda:{gpu}')
            labels = batch['target']['input_ids'].to(f'cuda:{gpu}')
            output = model(input_ids=input_ids, labels=labels)
            loss_sum += output.loss.detach()
            if (iter + 1) % 50 == 0 and gpu == 0:
                logger.info('step : {}/{} Loss: {:.4f}'.format(
                iter, 
                str(len(dev_loader)),
                output.loss.detach()
                ))
    return  loss_sum/iter



def test(args, model, test_loader):
    answer_list = []
    pred_list = []
    
    model.eval()
    if args.do_test:
        logger.info("Test start")
        with torch.no_grad():
            for iter,batch in enumerate(test_loader):
                outputs_text = model.generate(input_ids=batch['input']['input_ids'].to('cuda'))
                outputs_text = [args.tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in outputs_text]
                answer_text = [args.tokenizer.decode(o).replace('</s>','').replace('<pad>','').strip() for o in batch['target']['input_ids']]
                answer_list += answer_text
                pred_list+=outputs_text
                
                if (iter + 1) % 50 == 0:
                    logger.info('step : {}/{}'.format(
                    iter+1, 
                    str(len(test_loader)),
                    ))

        save_pickle("./logs/test_results.pickle", {'answer_list' : answer_list, 'pred_list': pred_list})

    
    saved_file  = load_pickle("./logs/test_results.pickle")
    answer_list, pred_list = saved_file['answer_list'], saved_file['pred_list']
    F1 = cal_f1(answer_list, pred_list)
    EM = cal_EM(answer_list, pred_list)
    return  EM, F1

        
        