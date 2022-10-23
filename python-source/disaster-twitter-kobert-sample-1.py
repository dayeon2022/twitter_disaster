import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

# device = torch.device("cuda:0")

bertmodel, vocab = get_pytorch_kobert_model()

import snscrape.modules.twitter as sntwitter
from datetime import datetime, timedelta
import pandas as pd

df = pd.read_excel("2020년 지자체 사고 8종 마이크로데이터-수난.xlsx", sheet_name="수난")
# df = pd.read_csv("2020-disaster.csv")
df1 = df.iloc[:,[1,2,3,4,5,16,17]]

date_format = '%Y-%m-%d'
target_list = []
check_row = 0
search_date = ''
tweet_read = []

# For Bert
train_data = []
# test_data = []

for index, disaster in df1.iterrows():
    from_time = int(disaster[3][0:2])
    to_time = int(disaster[3][3:5])
    since_date = datetime.strptime(disaster[0],date_format)
    until_date = since_date + timedelta(days=1)
    
    print(since_date.strftime('%Y-%m-%d') + '~' + until_date.strftime('%Y-%m-%d'))
    
    if check_row % 10 == 0 :
        print('current row = ' + str(check_row))
    
    query = '태풍 since:'+since_date.strftime('%Y-%m-%d')+' until:' + until_date.strftime('%Y-%m-%d')
    
    if search_date != since_date :
        search_date = since_date
        tweet_read = []
        for tweet in sntwitter.TwitterSearchScraper(query).get_items():
            tweet_read.append(tweet)
            tweet_hour = tweet.date.hour
            if tweet_hour >= from_time and tweet_hour <= to_time :
                list = [disaster[0],disaster[1],disaster[2],disaster[3],disaster[4],disaster[5],disaster[6],disaster[7],tweet.content]
                target_list.append(list)
                train_data.append([tweet.content,disaster[7]])
    else:
        for tweet in tweet_read:
            tweet_hour = tweet.date.hour
            if tweet_hour >= from_time and tweet_hour <= to_time :
                list = [disaster[0],disaster[1],disaster[2],disaster[3],disaster[4],disaster[5],disaster[6],disaster[7],tweet.content]
                target_list.append(list)
                train_data.append([tweet.content,disaster[7]])
    
    check_row = check_row + 1

# 앞에서 만든 작업을 한번더 해야 함. 
# Not 태풍인것...... / 위험도 : 0 

target_df = pd.DataFrame(target_list, columns=['신고년월일','월','신고시각','시간분류','발생장소','사망자수','부상자수','위험도','트윗'])
target_df

# 실제 사용할 데이터 :  train_data

from sklearn.model_selection import train_test_split
dataset_train, dataset_test = train_test_split(train_data, test_size=0.25, random_state=0)

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))


max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate =  5e-5

tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

data_train = BERTDataset(dataset_train, 0, 1, tok, max_len, True, False)
data_test = BERTDataset(dataset_test, 0, 1, tok, max_len, True, False)

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=batch_size, num_workers=5)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=batch_size, num_workers=5)

class BERTClassifier(nn.Module):
    def __init__(self,
                 bert,
                 hidden_size = 768,
                 num_classes=2,   ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
                 
        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
    
    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        
        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)


model = BERTClassifier(bertmodel,  dr_rate=0.5)

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc
    
train_dataloader



for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(train_dataloader)):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))
    
    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm_notebook(test_dataloader)):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} test acc {}".format(e+1, test_acc / (batch_id+1)))
    


tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def predict(predict_sentence):

    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)
    
    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length= valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)


        test_eval=[]
        for i in out:
            logits=i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("날씨가 좋습니다.")
            elif np.argmax(logits) > 0:
                test_eval.append("날씨가 위험합니다.")
            
        print(">> 입력하신 내용에서 " + test_eval[0] + " 느껴집니다.")


