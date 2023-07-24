import torch
import torch.nn as nn
from tqdm import tqdm 
import argparse 
from utils.hparams import HParam
from model.cbow import CBOW

def get_text(file_path):
    raw_text_file = open(file_path, 'r')
    raw_text = raw_text_file.read()
    return raw_text.strip().split()

def create_data(CONTEXT_SIZE, raw_text):
    '''
    each element is a tuple - (context, target)
    '''
    data = []
    for i in range(CONTEXT_SIZE, len(raw_text) - CONTEXT_SIZE):
        context = [raw_text[i - CONTEXT_SIZE], raw_text[i - CONTEXT_SIZE],
                raw_text[i + CONTEXT_SIZE], raw_text[i + CONTEXT_SIZE]]
        target = raw_text[i]
        data.append((context, target))
    return data 

def context_to_index(context, word_to_index):
    
    idxs = [word_to_index[w] for w in context]
    return torch.tensor(idxs, dtype=torch.long)

def train(hp, resume, load_from):
    # get device 
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    # get param
    CONTEXT_SIZE = hp.CONTEXT_SIZE
    EMBEDDING_DIM = hp.EMBEDDING_DIM 
    
    # get data
    path_data = hp.train
    raw_text = get_text(path_data)
    data = create_data(CONTEXT_SIZE, raw_text)
    
    # create vocab and vocab_dict
    vocab = set(raw_text)
    vocab_size = len(vocab)

    word_to_index = {word: index for index, word in enumerate(vocab)}
    index_to_word = {index: word for index, word in enumerate(vocab)}
    
    # select model 
    model = CBOW(vocab_size, EMBEDDING_DIM)
    
    # choose loss function and optimizer 
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    
    for epoch in tqdm(range(40)):
        total_loss = 0

        for context, target in data:
            context_vector = context_to_index(context, word_to_index)  

            log_probs = model(context_vector)

            total_loss += loss_function(log_probs, torch.tensor([word_to_index[target]]))

        #optimize at the end of each epoch
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # Test
    context = ['This', 'fantasy', 'manga', 'follows']
    context_vector = context_to_index(context, word_to_index)
    a = model(context_vector)

    #Print result
    print(f'Raw text: {" ".join(raw_text)}\n')
    print(f'Context: {context}\n')
    print(f'Prediction: {index_to_word[torch.argmax(a[0]).item()]}')
    
if __name__ == "__main__":
    # create an instance of class ArgumentParser()
    parser = argparse.ArgumentParser(description="config")
    
    # --config flag
    parser.add_argument("--config", 
                        required=True, 
                        help='yaml file for configuration')
    # --resume flag
    parser.add_argument("--resume",
                        help="path to latest checkpoint")
    # --load_from flag
    parser.add_argument("--load_from", 
                        help="path to latest checkpoint")
    
    # Parse all arguments of an instance parser 
    args = parser.parse_args()
    
    hp = HParam(args.config)
    
    train(hp, resume=args.resume, load_from=args.load_from)   