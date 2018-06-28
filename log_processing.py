
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import os
from tqdm import tqdm
import logging
from gensim.models import word2vec

pd.set_option('display.max_columns', 10)

file_path = r"D:\USERLOG"
save_path = r"D:\USERLOG\select_user"


userset01 = pickle.load(open(os.path.join(file_path, 'userset04.pkl'), 'rb'))
len(userset01)
userset01.head(20)
userset01.tail(20)
print(userset01.iloc[0,:].InTime.day)


######################
# TODO ::: except 03/27
######################
userset01.index = range(len(userset01))
userset01.iloc[0,:].InTime.date()
userset01.iloc[0,:].InTime.to_pydatetime()

drop_lst = [i for i in tqdm(range(len(userset01))) if userset01.iloc[i].OutTime.day == 27]
'''
#save
with open(os.path.join(save_path, 'user03_droplist_except27.pkl'), 'wb') as f:
    pickle.dump(drop_lst, f)

#load
with open(os.path.join(save_path, 'user01_droplist_except27.pkl'), 'rb') as f:
    drop_lst = pickle.load(f)
'''
select_day = userset01.drop(userset01.index[drop_lst])

# add calculated seconds
select_day.index = range(len(select_day))
def bet_second(dataset, timeind):
   sec = int(dataset.OutTime[timeind].value) - int(dataset.InTime[timeind].value)
   sec_ = sec / 1000000000
   return int(sec_)
bets = [bet_second(select_day, ind) for ind in tqdm(range(len(select_day)))]
select_day['bets'] = bets


#save
select_day.to_pickle(r'D:\USERLOG\select_user\user04_dayselect_except27.pkl')

#load
#select_day = pickle.load(open(r'D:\USERLOG\select_user\user01_dayselect_except27.pkl', 'rb'))



######################
# TODO ::: user select
######################

### log count by user
# user_count = select_day.ID.value_counts()
# user_sel = user_count.index[user_count >= 700]
# user_sel_count = user_count[user_sel]
# user_sel_series = pd.Series(user_sel)
# user_sel_series.index = user_sel_series
# user_sel_series = pd.DataFrame({'ID':user_sel_series})
# select_day_user = pd.merge(select_day, user_sel_series, how='right', on='ID')

### sum of second by onoff_sess
# user pre-filtering for cost reduction
user_count = select_day.ID.value_counts()
print(user_count.describe())
user_sel_ = user_count[user_count > 1000]
user_sel = user_sel_[user_sel_ < 5000]
user_sel_df = pd.DataFrame({'ID':user_sel.index}, index=user_sel.index)
select_day_user_filter = pd.merge(select_day, user_sel_df, how='right', on='ID')
# seconds by onoff_sess group
sess_sum_df = select_day_user_filter.groupby(["ID","OnOff_sess"]).sum().reset_index()
print(sess_sum_df.head(15))
sess_sum_df_60s = sess_sum_df[sess_sum_df.bets >= 60]
print(sess_sum_df_60s.head(15))
sess_sum_df_60s_ID = sess_sum_df_60s.bets.groupby([sess_sum_df_60s.ID]).count().reset_index()
print(sess_sum_df_60s_ID.head(15))
sess_sum_60s_100c_ID = sess_sum_df_60s_ID[sess_sum_df_60s_ID.bets >= 100]
print(sess_sum_60s_100c_ID.head(15))
user_sel_sess = pd.DataFrame({'ID':sess_sum_60s_100c_ID.ID})
print('Number of total user :', len(user_count), '\n', 'Number of filtered user :', len(user_sel_df), '\n', 'Number of selected user :', len(user_sel_sess))
# select user
select_day_user = pd.merge(select_day, user_sel_sess, how='right', on='ID')


######################
# TODO ::: add day (require to day-level in hierarchical structure)
# TODO ::: convert less than 1 minute channel to 'Jap'
######################
# add day
day_col = [select_day_user.OutTime[i].day for i in tqdm(range(len(select_day_user)))]
select_day_user['day'] = day_col

# < 60min delete (or Jap)
CH_m = []
for i in tqdm(range(len(select_day_user))):
    if select_day_user.bets[i] < 60:
        CH_m.append('Jap')
    else: CH_m.append(select_day_user.CH[i])
select_day_user['CH_m'] = CH_m

#save
select_day_user.to_pickle(r'D:\USERLOG\select_user\user04_dayuserselect_except27.pkl')
#load
select_day_user = pickle.load(open(r'D:\USERLOG\select_user\user01_dayuserselect_except27.pkl', 'rb'))



######################
# TODO ::: channel matching
######################

match_path = r"D:\USERLOG\select_user\tv_channel_info.csv"
match_data = pd.read_csv(match_path)
match_data.iloc[:,0] = [int(i) for i in match_data.iloc[:,0] if i != None]

match_dic = {}
for i in range(len(match_data)):
    match_dic['S{}'.format(match_data.loc[i][0])] = ['S{}'.format(match_data.loc[i][1]), match_data.loc[i][3]]

t_path = r'D:\USERLOG\select_user\total'
with open(os.path.join(t_path, 'match_dic.pkl'), 'wb') as f: pickle.dump(match_dic, f)
with open(os.path.join(t_path, 'match_dic.pkl'), 'rb') as f: match_dic = pickle.load(f)

CH_match = []
for i in range(len(select_day_user)):
    if select_day_user.loc[i]['day'] in [19,20,21,22,23,24,25,26,27]:
        if select_day_user.loc[i]['CH_m'] == 'Jap':
            CH_match.append(select_day_user.loc[i]['CH_m'])
        elif select_day_user.loc[i]['CH_m'] in list(match_dic.keys()):
            CH_match.append(match_dic[select_day_user.loc[i]['CH_m']][0])
        else:
            CH_match.append(select_day_user.loc[i]['CH_m'])
    else:
        CH_match.append(select_day_user.loc[i]['CH_m'])
    if i % 1000 == 0: print("{}/{}".format(i, len(select_day_user)))


select_day_user.loc[83125]
tmp = select_day_user.head(10).copy()
tmp.loc[1]['CH'] = 'S13111'




######################
# TODO ::: list generate
######################
# explore bets
print(select_day_user[select_day_user.CH_m != 'Jap'].bets.describe(percentiles=np.arange(0,1,.1)).apply(lambda x: format(x, 'f')))
# 1200second(10min), 1 japping
print(select_day_user.index)
unique_day = range(20, 27)
unique_user = select_day_user.ID.unique()


SEP = 60*20
#MAX_TIME = 3600*5

user_CH = defaultdict()
for i, user in enumerate(list(unique_user)):
    print('{}//{}'.format(i+1, len(unique_user)))
    userdf = select_day_user[select_day_user.ID == user]
    userdf.index = range(len(userdf))
    chlst = []
    for day in unique_day:
        userdaydf = userdf[userdf.day == day]
        userdaydf.index = range(len(userdaydf))
        daylst = ['SS']
        for j, ch in enumerate(list(userdaydf.CH_m)):
            if ch == 'Jap' and daylst[-1] == 'Jap':
                continue
            elif ch == 'Jap' and daylst[-1] != 'Jap':
                daylst.append('Jap')
            #elif userdaydf.bets[j] >= MAX_TIME:
            #    for _ in range(int(MAX_TIME/SEP)+1):
            #        daylst.append(ch)
            #    continue
            else:
                for _ in range(int(userdaydf.bets[j]/SEP)+1):
                    daylst.append(ch)
                continue
        daylst.remove('SS')
        chlst.append(daylst)
    user_CH[user] = chlst
'''
#test
tmp04 = select_day_user[select_day_user.ID == list(user_CH.keys())[:1][0]]
user_CH[list(user_CH.keys())[:1][0]][0][:20]
'''
#save
with open(r'D:\USERLOG\select_user\user04_dict_10m.pkl', 'wb') as f:
    pickle.dump(user_CH, f)

#load
#with open(r'D:\USERLOG\select_user\user01_dict_10m.pkl', 'rb') as f:
#    tmp = pickle.load(f)



# TODO :: emergency (rebuilding dictionary)
# delete 19 day
# unique_day = [20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31,  1,  2,  3,  4,  5]
em_path = r'D:\USERLOG\select_user'
em_dir = [file for file in os.listdir(em_path) if 'dayuserselect' in file]
for ind, pkl in enumerate(em_dir):
    user_pkl = pickle.load(open(os.path.join(em_path, pkl), 'rb'))
    user_pkl_ = user_pkl[(user_pkl.day != 19) & (user_pkl.day != 5)]
    unique_day_ = user_pkl_.day.unique()
    unique_user_ = user_pkl_.ID.unique()

    SEP = 60*20
    #MAX_TIME = 3600*5

    user_CH_ = defaultdict()
    for i, user in enumerate(list(unique_user_)):
        print('user{} {}//{}'.format(ind+1, i+1, len(unique_user_)))
        userdf_ = user_pkl_[user_pkl_.ID == user]
        userdf_.index = range(len(userdf_))
        chlst_ = []
        for day in unique_day_:
            userdaydf_ = userdf_[userdf_.day == day]
            userdaydf_.index = range(len(userdaydf_))
            daylst_ = ['SS']
            for j, ch in enumerate(list(userdaydf_.CH_m)):
                if ch == 'Jap' and daylst_[-1] == 'Jap':
                    continue
                elif ch == 'Jap' and daylst_[-1] != 'Jap':
                    daylst_.append('Jap')
                #elif userdaydf_.bets[j] >= MAX_TIME:
                #    for _ in range(int(MAX_TIME/SEP)+1):
                #        daylst_.append(ch)
                #    continue
                else:
                    for _ in range(int(userdaydf_.bets[j]/SEP)+1):
                        daylst_.append(ch)
                    continue
            daylst_.remove('SS')
            chlst_.append(daylst_)
        user_CH_[user] = chlst_

    with open(os.path.join(em_path, 'user{}_dict_10m.pkl'.format(ind+1)), 'wb') as f:
        pickle.dump(user_CH_, f)


#load
#with open(r"D:\USERLOG\select_user\user1_dict_10m.pkl", 'rb') as f: tmp = pickle.load(f)


######################
# TODO ::: channel vectorization
# TODO ::: (!!! have to work on all the data)
# TODO ::: build vocabulary
######################
# dictionary load
dic_path = r'D:\USERLOG\select_user'
dic_dir = [file for file in os.listdir(dic_path) if 'dic' in file]
user_dic_list = []
for dic in dic_dir:
    with open(os.path.join(dic_path, dic), 'rb') as f:
        user_dic_list.append(pickle.load(f))

# collection in dictionary
user_dic = defaultdict()
for i, user_set in enumerate(user_dic_list):
    print('{}-{}'.format(i+1, 10))
    for j in range(len(user_set)):
        key = 'userset{}_'.format(i+1) + list(user_set.keys())[j]
        user_dic[key] = user_set[list(user_set.keys())[j]]
print(len(list(user_dic.keys())), '_users')

# delete empty list
user_fdic = {}
for user, ch_lst in user_dic.items():
    ch_lst_del = list(filter(None, ch_lst))
    user_fdic[user] = ch_lst_del

# save, load
t_path = r'D:\USERLOG\select_user\total'
with open(os.path.join(t_path, 'total_user_dic_10m.pkl'), 'wb') as f:
    pickle.dump(user_fdic, f)
with open(os.path.join(t_path, 'total_user_dic_10m.pkl'), 'rb') as f:
    user_fdic = pickle.load(f)

'''
# TODO ::: channel matching
match_path = r"D:\USERLOG\select_user\tv_channel_info.csv"
match_data = pd.read_csv(match_path)
match_data.iloc[:,0] = [int(i) for i in match_data.iloc[:,0] if i != None]

match_dic = {}
for i in range(len(match_data)):
    match_dic['S{}'.format(match_data.loc[i][0])] = ['S{}'.format(match_data.loc[i][1]), match_data.loc[i][3]]

with open(os.path.join(t_path, 'match_dic.pkl'), 'wb') as f: pickle.dump(match_dic, f)
with open(os.path.join(t_path, 'match_dic.pkl'), 'rb') as f: match_dic = pickle.load(f)

total_user_dic = {}
for user in list(user_fdic.keys()):
    new_ch_list = []
    for channel_day in user_fdic[user]:
        new_ch_day = []
        for channel in channel_day:
            if channel == 'Jap': 
                new_ch_day.append('Jap')
            else:
                if channel in list()
                new_ch_day.append(match_dic[channel][0])
        new_ch_list.append(new_ch_day)
    total_user_dic[user] = new_ch_list        
'''



# unfold
user_ch_lst = [week for user in list(user_fdic.keys()) for week in user_fdic[user]]
print('average of {} days per user'.format(len(user_ch_lst)/len(user_fdic.keys())))

# vectorization (Word2Vec)
MIN_SAMPLE = 0
EMBEDDING_SIZE = 100
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
CH_vec = word2vec.Word2Vec(user_ch_lst, iter=5, min_count=MIN_SAMPLE, sg=1, size=EMBEDDING_SIZE)
CH_embedding = np.zeros((len(CH_vec.wv.vocab), EMBEDDING_SIZE))
for i in range(len(CH_vec.wv.vocab)):
    embedding_vec = CH_vec.wv[CH_vec.wv.index2word[i]]
    if embedding_vec is not None:
        CH_embedding[i] = embedding_vec
print('shape :', CH_embedding.shape, type(CH_embedding))
# save
t_path = r'D:\USERLOG\select_user\total'
np.save(os.path.join(t_path, 'user_iter5_300dim.npy'), CH_embedding)
with open(os.path.join(t_path, 'user_index2word_iter5_300dim.pkl'), 'wb') as f:
    pickle.dump(CH_vec.wv.index2word, f)

# word2vec iter embedding size
for emb in [100,200,300]:
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    CH_vec = word2vec.Word2Vec(user_ch_lst, iter=5, min_count=0, sg=1, size=emb)
    CH_embedding = np.zeros((len(CH_vec.wv.vocab), emb))
    for i in range(len(CH_vec.wv.vocab)):
        embedding_vec = CH_vec.wv[CH_vec.wv.index2word[i]]
        if embedding_vec is not None:
            CH_embedding[i] = embedding_vec
    t_path = r'D:\USERLOG\select_user\total'
    np.save(os.path.join(t_path, 'user_{}dim.npy'.format(emb)), CH_embedding)
    # vocab
    keys = CH_vec.wv.index2word
    CH_vocab = {}
    for i, channel in enumerate(keys):
        CH_vocab[channel] = i + 1
    CH_vocab['UNK'] = 0
    with open(os.path.join(t_path, 'channel_vocab_{}dim.pkl'.format(emb)), 'wb') as f:
        pickle.dump(CH_vocab, f)




'''
# vectorization (Glove)
print('-')

# vectorization (FastText)
# https://www.quora.com/What-is-the-main-difference-between-word2vec-and-fastText
from gensim.models import FastText
MIN_SAMPLE = 1
ITERATION = 5
WINDOW_SIZE = 5
EMBEDDING_SIZE = 300
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
chvec = FastText(sentences=user_ch_lst, size=EMBEDDING_SIZE, window=WINDOW_SIZE, min_count=MIN_SAMPLE, sg=1, iter=ITERATION)
embedding = np.zeros((len(chvec.wv.vocab), EMBEDDING_SIZE))
for i in range(len(chvec.wv.vocab)):
    embedding_vec = chvec.wv[chvec.wv.index2word[i]]
    if embedding_vec is not None:
        embedding[i] = embedding_vec
print(embedding.shape)
np.save(os.path.join(t_path, 'user01_iter5_300dim(FastText).npy'), embedding)
with open(os.path.join(t_path, 'user01_index2word(FastText).pkl'), 'wb') as f:
    pickle.dump(chvec.wv.index2word, f)
'''






######################
# TODO ::: build dataset
# TODO ::: (3 day-level window, next channel be label)
# TODO ::: (sent/doc-level correspond to day/person-level)
# TODO ::: (zero padding) -->  generator
######################
# load
t_path = r'D:\USERLOG\select_user\total'
with open(os.path.join(t_path, 'total_user_dic_10m.pkl'), 'rb') as f:
    user_fdic = pickle.load(f)

def seq_to_table(seq, window_size, stride=1):
    # seq type --> dictionary
    X = []; y = []
    for ind, channel in enumerate(list(seq.values())):
        if ind+1 % 100 == 0:
            print('{}-{}'.format(ind+1, len(seq)))
        for i in range(0, len(channel)-window_size, stride):
            if i == len(channel)-window_size+1:
                break
            subset = channel[i:(i+window_size)]
            X.append(subset[:window_size-1])
            y.append(subset[-1][0])
    return X, y

# # sliding window by user
WINDOW = 4 # 3input, 1target
channel_input, channel_target = seq_to_table(user_fdic, WINDOW)


# target change
def channel_to_category(channel_label, match_path=r"D:\USERLOG\select_user\tv_channel_info.csv"):
    match_data = pd.read_csv(match_path)
    match_data.index = list(match_data.iloc[:,1])
    category_list = []
    for i, channel in enumerate(channel_label):
        channel_ = channel[1:]
        if channel == 'Jap':
            category_list.append('Jap')
        elif int(channel_) in list(match_data.index):
            label = match_data[match_data.index==int(channel_)]['종류'].values[0]
            category_list.append(label)
        else:
            category_list.append('기타')

        if i%1000 == 0: print(i, len(channel_label))
    return category_list


tmp = channel_to_category(channel_target)
print(tmp)
print(set(tmp))
len(set(tmp))
len(set(match_data.종류))



# TODO ::: Sub-y ??????
pass

# TODO ::: build additional feature
# TODO ::: how to modeling with additional feature









'''
### previous version
# channels + japping append
LOGEMB_PATH = 'D:/USERLOG/EMBEDDING'
userset01.index = range(len(userset01))
print(userset01.head(10))
CHlst = []
userset01_hf = int(len(userset01)/2)

for i in range(userset01_hf):
   print(i+1, " processing, total ", userset01_hf)
   if userset01.CH[i] == 'Jap':
       CHlst.append(['Jap'])
   else:
       for k in range(int(betsecond(userset01, i)/3600)+1):
           CHlst.append([userset01.CH[i]])

for i in range(userset01_hf, len(userset01)):
   print(i+1, " precessing, total ", len(userset01))
   if userset01.CH[i] == 'Jap':
       CHlst.append(['Jap'])
   else:
       for k in range(int(betsecond(userset01, i)/3600)+1):
           CHlst.append([userset01.CH[i]])


with open(os.path.join(LOGEMB_PATH + '/user09_list(60minCH_1Jap).pickle'), 'wb') as f:
   pickle.dump(CHlst, f)

del userset01


# vectorization
LOGEMB_PATH = 'D:/USERLOG/EMBEDDING'
MIN_SAMPLE = 0
EMBEDDING_SIZE = 100

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
w2v = word2vec.Word2Vec(CHlst, iter=5, min_count=MIN_SAMPLE, sg=1, size=EMBEDDING_SIZE)
print(w2v.wv.vocab)
embedding = np.zeros((len(w2v.wv.vocab), EMBEDDING_SIZE))
for i in range(len(w2v.wv.vocab)):
   embedding_vec = w2v.wv[w2v.wv.index2word[i]]
   if embedding_vec is not None:
       embedding[i] = embedding_vec
print(embedding.shape, type(embedding))

np.save(os.path.join(LOGEMB_PATH + '/user09_iter5_sg1_100dim'), embedding)

with open(os.path.join(LOGEMB_PATH + '/user09_wv.index2word.pickle'), 'wb') as fi:
   pickle.dump(w2v.wv.index2word, fi)
'''
