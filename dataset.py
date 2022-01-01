import re
import pdb
import json
import torch
import pickle
from tqdm import tqdm
import logging
from log_conf import init_logger
from collections import defaultdict
import random
logger = logging.getLogger("my")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, args, data_path, data_type):
        random.seed(args.seed)
        self.data_type = data_type # 없어도 될까?
        
        self.tokenizer = args.tokenizer
        self.max_length = args.max_length
        self.data_type = data_type
        
        if args.do_short:
            data_path = f'./data/short.json'
            
            
        logger.info(f"load {self.data_type} raw file {data_path}")   
        raw_dataset = json.load(open(data_path , "r"))
        question, context, answer= self.seperate_data(raw_dataset)

        assert len(question) == len(context) == len(answer)
                    
        self.answer = answer # for debugging
        self.target = self.encode(answer)
        self.question = question
        self.context = context
            
    def encode(self, texts ,return_tensors="pt"):
        examples = []
        for i, text in enumerate(texts):
            tokenized = self.tokenizer.batch_encode_plus([text], max_length = self.max_length, padding=False, truncation=True, return_tensors=return_tensors) # TODO : special token
            examples.append(tokenized)
        return examples

    def __len__(self):
        return len(self.answer)

    def seperate_data(self, dataset):
        question, context, answer = [], [], []
        
        for raw_data in dataset:
            
            data = raw_data['paragraphs'][0]
            QA_text =data['qas']

            for QA in QA_text:
                context.append(data['context'])
                question.append(QA['question'])
                answer.append(QA['answers'][0]['text'])

        return question, context, answer

    def __getitem__(self, index):
              
        question = self.question[index]
        context = self.context[index]
        answer = self.answer[index] # for debugging
        
        target = {k:v.squeeze() for (k,v) in self.target[index].items()}
        
        return {"target": target, "context" : context, "answer" : answer, "question" : question}

    
    def make_DB(self, belief_state, activate):
        pass
    
    def collate_fn(self, batch):
        """
        The tensors are stacked together as they are yielded.
        Collate function is applied to the output of a DataLoader as it is yielded.
        context = self.context[index]
        belief_state = self.belief_state[index]
        """
        question = [x["question"] for x in batch]
        context = [x["context"] for x in batch]
        target = [x["target"] for x in batch]

        input_source = [f"question: {q} context: {c}" for (q,c) in  \
            zip(question, context)]
        
        source = self.encode(input_source)
        source_list = [{k:v.squeeze() for (k,v) in s.items()} for s in source]
            
        pad_source = self.tokenizer.pad(source_list,padding=True)
        pad_target = self.tokenizer.pad(target,padding=True)
        
        return {"input": pad_source, "target": pad_target}
        

if __name__ == '__main__':
    import argparse
    init_logger(f'data_process.log')
    logger = logging.getLogger("my")

    parser = argparse.ArgumentParser()

    parser.add_argument('--do_short' ,  type = int, default=1)
    parser.add_argument('--seed' ,  type = float, default=1)
    parser.add_argument('--max_length' ,  type = int, default=128)
    
    
    args = parser.parse_args()

    args.data_path = './data/dev.json'
    from transformers import T5Tokenizer
    args.tokenizer = T5Tokenizer.from_pretrained('t5-small')
    
    dataset = Dataset(args, args.data_path, 'train')
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=16, collate_fn=dataset.collate_fn)
        
    for batch in loader:
        pdb.set_trace()
    
    