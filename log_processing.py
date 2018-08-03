
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
# TODO ::: remove 03/27
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









import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import os
from tqdm import tqdm
import logging
from gensim.models import word2vec

pd.set_option('display.max_columns', 15)


# TODO : remove 3/19, 3/27, 4/4, 4/5
load_path = r"D:\USERLOG\select_user"
save_path = r"D:\USERLOG\select_user2"

dir = os.listdir(load_path)
dir_list = [file for file in dir if "dayselect" in file]
for k, user_set in enumerate(dir_list):
    # load data
    if k != 9:
        userset = pickle.load(open(os.path.join(load_path, "user0{}_dayselect_except27.pkl".format(k+1)), 'rb'))
    else:
        userset = pickle.load(open(os.path.join(load_path, "user{}_dayselect_except27.pkl".format(k+1)), 'rb'))
    # build remove index list
    drop_lst = []
    for i in range(len(userset)):
        print("{} build drop list, {}/{}".format(user_set[:7], i+1, len(userset)))
        if (userset.iloc[i].OutTime.day in [19,27,4,5]) and (userset.iloc[i].InTime.day in [19,27,4,5]):
            drop_lst.append(i)
    # remove day
    userset_rm_day = userset.drop(userset.index[drop_lst])
    # save
    if k!= 9:
        userset_rm_day.to_pickle(os.path.join(save_path, "user0{}_rm_day.pkl".format(k+1)))
    else:
        userset_rm_day.to_pickle(os.path.join(save_path, "user{}_rm_day.pkl".format(k+1)))



# TODO : user select, add day, channel matching, change Jap
save_path = r"D:\USERLOG\select_user2"
set_dir = os.listdir(save_path)
t_path = r'D:\USERLOG\select_user\total'
with open(os.path.join(t_path, 'match_dic.pkl'), 'rb') as f: match_dic = pickle.load(f)
upper_bound = 3600*15*14
lower_bound = 3600*2*14
jap_line = 5*60
for dir_ind, u_set in enumerate(set_dir):
    user_set = pickle.load(open(os.path.join(save_path, u_set), "rb"))
    user_set.index = range(len(user_set))
    # user select
    user_time = user_set[["ID", "bets"]].groupby(["ID"]).sum().reset_index()
    user_time_sel = user_time[(user_time.bets <= upper_bound) & (user_time.bets >= lower_bound)]
    user_time_sel = user_time_sel.drop('bets', 1)
    user_sltd = pd.merge(user_set, user_time_sel, how='right', on='ID')
    # selected set processing..
    day_ = []; day_tf = []
    CH_ = []
    for set_ind in range(len(user_sltd)):
        In_day = user_sltd.iloc[set_ind].InTime.day
        Out_day = user_sltd.iloc[set_ind].OutTime.day
        day_CH = user_sltd.iloc[set_ind].CH
        day_bets = user_sltd.iloc[set_ind].bets
        # add day
        if In_day != Out_day:
            day_.append(In_day)
            day_tf.append(99)
        else:
            day_.append(In_day)
            day_tf.append(0)
        # channel matching
        # change to "Jap" less than 5 minutes
        if In_day >= 19:
            if day_CH == "Jap":
                CH_.append("Jap")
            elif day_bets < jap_line:
                CH_.append("Jap")
            elif day_CH in list(match_dic.keys()):
                CH_.append(match_dic[day_CH][0])
            else:
                CH_.append(day_CH)
        else:
            CH_.append(day_CH)
        # print
        if set_ind % 10000 == 0:
            print("{} processing.. {}/{}".format(u_set[:6], set_ind, len(user_sltd)-1))
            print("Length {},{},{}".format(len(day_), len(day_tf), len(CH_)))
    user_sltd['day'] = day_
    user_sltd['day_tf'] = day_tf
    user_sltd['CH_matching'] = CH_
    user_sltd.to_pickle(os.path.join(save_path, "{}_sltd.pkl".format(u_set[:6])))



# TODO : matching again, channel to category
data_lst = [data for data in os.listdir(save_path) if "sltd." in data]

t_path = r'D:\USERLOG\select_user\total'
with open(os.path.join(t_path, 'match_dic.pkl'), 'rb') as f: match_dic = pickle.load(f)
match_dic_2 = {value[0]:value[1] for key, value in match_dic.items()}

for data_f in data_lst:
    data_path = os.path.join(save_path, data_f)
    user_sltd = pickle.load(open(data_path, "rb"))

    CH_match = []
    category = []
    for i in range(len(user_sltd)):
        ind_day = user_sltd.iloc[i].day
        ind_CH = user_sltd.iloc[i].CH
        if ind_day in range(10,28):
            if ind_CH == "Jap":
                CH_match.append("Jap")
                category.append("Jap")
            elif ind_CH in list(match_dic.keys()):
                CH_match.append(match_dic[ind_CH][0])
                category.append(match_dic[ind_CH][1])
            else:
                CH_match.append(ind_CH)
                category.append('기타')
        else:
            CH_match.append(ind_CH)
            if ind_CH in list(match_dic_2.keys()):
                category.append(match_dic_2[ind_CH])
            else:
                category.append('기타')

        if i % 1000 == 0:
            print("{}/{}/{}".format(data_f, i+1, len(user_sltd)))
        if i % 10000 == 0:
            print(len(CH_match), len(category))

    user_sltd['CH_match'] = CH_match
    user_sltd['Category'] = category

    user_sltd.to_pickle(r"D:\USERLOG\select_user2\user08_sltd_mod.pkl")




# TODO : user select again (viewing time per day)
data_path = r"D:\USERLOG\select_user2"
data_lst = [data for data in os.listdir(data_path) if 'mod' in data]

for data_set in data_lst:
    print("{} in progress ...".format(data_set[:6]))

    sltd_mod = pickle.load(open(os.path.join(data_path, data_set), "rb"))
    sltd_mod = sltd_mod.drop('CH_matching', 1)

    day_viewing = sltd_mod[["ID","bets","day"]].groupby(["ID","day"]).sum().reset_index()
    day_count = day_viewing[["ID","day"]].groupby(["ID"]).count().reset_index()
    #day_count.describe(percentiles=np.arange(0,1,.05))

    droped_subset = day_count[day_count.day >= 7]
    droped_user_lst = pd.DataFrame({'ID':droped_subset.ID})

    sltd = pd.merge(sltd_mod, droped_user_lst, how='right', on='ID')
    sltd.to_pickle(os.path.join(data_path, "{}_sltd_mod2.pkl".format(data_set[:6])))




# TODO : user select again again (continuous viewing time)
data_path = r"D:\USERLOG\select_user2"
data_lst = [data for data in os.listdir(data_path) if 'mod2' in data]

for data_set in data_lst:
    print("{} in progress ...".format(data_set[:6]))

    sltd_mod = pickle.load(open(os.path.join(data_path, data_set), "rb"))
    dif_day_ = sltd_mod[sltd_mod.day_tf == 99]
    dif_day = dif_day_[dif_day_.bets > (3600 * 15)]

    all_user = list(sltd_mod['ID'].unique())
    drop_user = list(dif_day['ID'].unique())
    sel_user = list(set(all_user) - set(drop_user))
    sel_user_df = pd.DataFrame({'ID':sel_user})

    sltd = pd.merge(sltd_mod, sel_user_df, how='right', on='ID')
    sltd.to_pickle(os.path.join(data_path, "{}_sltd_mod3.pkl".format(data_set[:6])))




# TODO : different In/Out day split
def bet_second_2(outtime, intime):
   sec = int(outtime.value) - int(intime.value)
   sec_ = sec / 1000000000
   return int(sec_)

data_path = r"D:\USERLOG\select_user2"
data_lst = [data for data in os.listdir(data_path) if 'mod3' in data]

for data_set in data_lst:
    sltd_mod = pickle.load(open(os.path.join(data_path, data_set), "rb"))

    # different IN/OUT day select.
    dif_day = sltd_mod[sltd_mod.day_tf == 99]
    dif_day.index = range(len(dif_day))

    add_lst = []
    for ind in range(len(dif_day)):
        print("{}/{}/{}".format(data_set[:6], ind+1, len(dif_day)))
        user_id = dif_day.iloc[ind].ID
        in_month = dif_day.iloc[ind].InTime.month
        out_month = dif_day.iloc[ind].OutTime.month
        in_day = dif_day.iloc[ind].InTime.day
        out_day = dif_day.iloc[ind].OutTime.day
        ch = dif_day.iloc[ind].CH
        sess = dif_day.iloc[ind].OnOff_sess
        ch_match = dif_day.iloc[ind].CH_match
        category = dif_day.iloc[ind].Category
        bet_1st = bet_second_2(pd.Timestamp(2017, in_month, in_day, 23, 59, 59), dif_day.iloc[ind].InTime)
        bet_2nd = bet_second_2(dif_day.iloc[ind].OutTime, pd.Timestamp(2017, out_month, out_day, 0, 0, 0))
        # case of same month
        if in_month == out_month:
            add_lst.append([user_id,
                            dif_day.iloc[ind].InTime,
                            pd.Timestamp(2017, in_month, in_day, 23, 59, 59),
                            ch, sess, bet_1st, in_day, 0, ch_match, category])
            add_lst.append([user_id,
                            pd.Timestamp(2017, out_month, out_day, 0,0,0),
                            dif_day.iloc[ind].OutTime,
                            ch, sess, bet_2nd, out_day, 0, ch_match, category])
        # case of different month
        elif in_month != out_month:
            add_lst.append([user_id,
                            dif_day.iloc[ind].InTime,
                            pd.Timestamp(2017, in_month, in_day, 23, 59, 59),
                            ch, sess, bet_1st, in_day, 0, ch_match, category])
            add_lst.append([user_id,
                            pd.Timestamp(2017, out_month, out_day, 0,0,0),
                            dif_day.iloc[ind].OutTime,
                            ch, sess, bet_2nd, out_day, 0, ch_match, category])
        else:
            raise AssertionError()
    # add split day dataframe
    add_df = pd.DataFrame(add_lst, columns=["ID", "InTime", "OutTime", "CH", "OnOff_sess", "bets", "day", "day_tf", "CH_match", "Category"])
    concat_df = pd.concat([sltd_mod, add_df], axis=0)
    # remove different IN/OUT day
    drop99_df = concat_df[concat_df.day_tf == 0]
    # sort
    drop99_df = drop99_df.sort_values(["ID"])
    drop99_df.index = range(len(drop99_df))
    # save
    drop99_df.to_pickle(os.path.join(data_path, "{}_vfin.pkl".format(data_set[:6])))





# TODO : different In/Out hour split
def bet_second_2(outtime, intime):
   sec = int(outtime.value) - int(intime.value)
   sec_ = sec / 1000000000
   return int(sec_)

data_path = r"D:\USERLOG\select_user2"
data_lst = [data for data in os.listdir(data_path) if 'vfin.pkl' in data]

for data_set in data_lst[8:]:
    user_vfin = pickle.load(open(os.path.join(data_path, data_set) ,"rb"))
    user_vfin = user_vfin.sort_values(["ID","InTime"])
    user_col = list(user_vfin.columns)
    vfin_len = len(user_vfin)//4

    user_vfin = user_vfin.iloc[(vfin_len*3):]

    proc_lst = []
    for i in range(len(user_vfin)):
        print("{}/{}/{}".format(data_set[:6], i+1, len(user_vfin)))

        e_row = list(user_vfin.iloc[i])
        year_ = user_vfin.iloc[i].InTime.year
        month_ = user_vfin.iloc[i].InTime.month
        day_ = user_vfin.iloc[i].InTime.day
        in_hour = user_vfin.iloc[i].InTime.hour
        out_hour = user_vfin.iloc[i].OutTime.hour
        in_minute = user_vfin.iloc[i].InTime.minute
        out_minute = user_vfin.iloc[i].OutTime.minute
        in_second = user_vfin.iloc[i].InTime.second
        out_second = user_vfin.iloc[i].OutTime.second

        if in_hour == out_hour:
            proc_lst.append(e_row)
        elif in_hour+1 == out_hour:
            proc_lst.append([e_row[0], e_row[1],
                            pd.Timestamp(year_,month_,day_,in_hour,59,59),
                            e_row[3], e_row[4],
                            bet_second_2(pd.Timestamp(year_,month_,day_,in_hour,59,59), e_row[1]),
                            e_row[6], e_row[7], e_row[8], e_row[9]])
            proc_lst.append([e_row[0],
                             pd.Timestamp(year_,month_,day_,out_hour,00,00),
                             e_row[2], e_row[3], e_row[4],
                             bet_second_2(e_row[2], pd.Timestamp(year_,month_,day_,in_hour+1,00,00)),
                             e_row[6], e_row[7], e_row[8], e_row[9]])
        else:
            proc_lst.append([e_row[0], e_row[1],
                            pd.Timestamp(year_,month_,day_,in_hour,59,59),
                            e_row[3], e_row[4],
                            bet_second_2(pd.Timestamp(year_,month_,day_,in_hour,59,59), e_row[1]),
                            e_row[6], e_row[7], e_row[8], e_row[9]])
            for hr in range(1, (out_hour-in_hour)):
                proc_lst.append([e_row[0],
                                 pd.Timestamp(year_,month_,day_,in_hour+hr,00,00),
                                 pd.Timestamp(year_,month_,day_,in_hour+hr,59,59),
                                 e_row[3], e_row[4], 3599, e_row[6], e_row[7], e_row[8], e_row[9]])
            proc_lst.append([e_row[0],
                             pd.Timestamp(year_,month_,day_,out_hour,00,00),
                             e_row[2], e_row[3], e_row[4],
                             bet_second_2(e_row[2], pd.Timestamp(year_,month_,day_,out_hour,00,00)),
                             e_row[6], e_row[7], e_row[8], e_row[9]])

    user_vfin = pd.DataFrame(proc_lst, columns=user_col)
    del proc_lst
    user_vfin.to_pickle(os.path.join(data_path, "{}_vfin2_v4.pkl".format(data_set[:6])))
    del user_vfin




user_vfin_v1 = pickle.load(open(r"D:\USERLOG\select_user2\user10_vfin2_v1.pkl", "rb"))
user_vfin_v2 = pickle.load(open(r"D:\USERLOG\select_user2\user10_vfin2_v2.pkl", "rb"))
user_vfin_v3 = pickle.load(open(r"D:\USERLOG\select_user2\user10_vfin2_v3.pkl", "rb"))
user_vfin_v4 = pickle.load(open(r"D:\USERLOG\select_user2\user10_vfin2_v4.pkl", "rb"))

user_vfin2 = pd.concat([user_vfin_v1, user_vfin_v2, user_vfin_v3, user_vfin_v4])
print(len(user_vfin2) == len(user_vfin_v1) + len(user_vfin_v2) + len(user_vfin_v3) + len(user_vfin_v4))
user_vfin2.to_pickle(r"D:\USERLOG\select_user2\user10_vfin2.pkl")




# TODO : remove day 3/19, 3/27, 4/4, 4/5 again (not processed to unknown reasons)
data_path = r"D:\USERLOG\select_user2"
data_lst = [data for data in os.listdir(data_path) if 'vfin2.pkl' in data]

for data_set in data_lst:
    sltd_mod = pickle.load(open(os.path.join(data_path, data_set), "rb"))

    rm_list = [4, 5, 19, 27]
    for rm_day in rm_list:
        sltd_mod = sltd_mod[sltd_mod.day != rm_day]
        print("{} delete {} ...".format(data_set[:6], rm_day))
    sltd_mod.to_pickle(r"D:\USERLOG\select_user2\{}_vfin2_.pkl".format(data_set[:6]))






# 20min
import pandas as pd
import os
import numpy as np
import pickle
import time

TSLOT_TIME_STR = '20min'
TSLOT_TIME_NUM = 20
FULL_SLOT_SEC = TSLOT_TIME_NUM * 60
START_DATE = '2017-03-20'
END_DATE = '2017-04-04'
DAYS = (pd.to_datetime(END_DATE) - pd.to_datetime(START_DATE)).days
NUM_TSLOT = 3 * 24 * DAYS

start_dt = pd.to_datetime(START_DATE)
end_dt = pd.to_datetime(END_DATE)
tslot_delta = pd.Timedelta(TSLOT_TIME_STR)
onesec = pd.Timedelta('1s')

data_path = r"D:\USERLOG\select_user2"
data_dir = [data for data in os.listdir(data_path) if 'prof' in data]

prof_ = pickle.load(open(os.path.join(data_path, data_dir[2]), "rb"))
prof_ = prof_[(prof_['OutTime'] >= start_dt) & (prof_["InTime"] < end_dt)]
CH_list = np.unique(prof_['CH'])

total_user_len = len(prof_.groupby('ID'))
user_set = {}
for i, (u, log) in enumerate(prof_.groupby('ID')):
    tframe = pd.DataFrame(index=np.arange(NUM_TSLOT), columns=CH_list)
    tframe = tframe.fillna(0)
    for tin, tout, ch in log[['InTime', 'OutTime', 'CH']].values:
        tin, tout = max(tin, start_dt), min(tout, end_dt)
        start_t = tin.ceil(freq=TSLOT_TIME_STR)
        start_t += ((start_t == tin) * 1) * tslot_delta
        end_t = tout.floor(freq=TSLOT_TIME_STR)
        end_t -= ((end_t == tout) * 1) * tslot_delta
        num_slot = (end_t - start_t) // tslot_delta + 2
        start_slot = (start_t - start_dt) // tslot_delta - 1
        if num_slot > 1:
            tframe.loc[np.arange(start=start_slot, stop=start_slot + num_slot, dtype='int'), ch] += np.concatenate(
                ([(start_t - tin) / onesec], np.ones(num_slot - 2) * FULL_SLOT_SEC if num_slot > 2 else [], [(tout - end_t) / onesec]))
        else:
            tframe.loc[start_slot, ch] += (tout - tin) / onesec

    viewing = []
    non_zeros = list(tframe.sum(1).nonzero()[0])
    for idx in range(len(tframe)):
        if idx in non_zeros:
            if tframe.iloc[idx].max() >= (1200/2):
                viewing.append(tframe.iloc[idx].idxmax())
            else:
                viewing.append("Jap")
        else:
            viewing.append("Off")
    user_set[u] = viewing

    print("{} - {}/{}".format(data_dir[2][:6], i+1, total_user_len))

with open(os.path.join(data_path, 'user{}_dict.pkl'.format(data_dir[2][4:6])), 'wb') as f:
    pickle.dump(user_set, f)





# output {user1: {day1:[ [,,], [,,] ... ],
#                 day2:[ [,,], [,,] ... ] ... },
#         user2: {day1:[ [,,], [,,] ... ],
#                 day2:[ [,,], [,,] ... ] ... },
#         ... } (20 min ver.)
# TODO : channel to vector embedding
# user1 [ , , , , ... ], user2 [ , , , , ... ], user3 [ , , , , ... ]
import pickle
import numpy as np
import pandas as pd
import os
import logging
from gensim.models import word2vec


pd.set_option('display.max_columns', 15)

data_path = r"D:\USERLOG\select_user2"
data_lst = [data for data in os.listdir(data_path) if 'dict.pkl' in data]

# unfold
all_user_name = []
all_user_lst_embed = []
for data_set in data_lst:
    with open(os.path.join(data_path, data_set), 'rb') as f:
        user_dic = pickle.load(f)
    print(data_set)

    for name in list(user_dic.keys()):
        all_user_name.append(name)

    users_lst = []
    for i, user in enumerate(list(user_dic.keys())):
        users_ = []
        for day in list(user_dic[user].keys()):
            for log_3 in user_dic[user][day]:
                for log in log_3:
                    users_.append(log)
        users_lst.append(users_)

    for day_log in users_lst:
        all_user_lst_embed.append(day_log)


all_user_lst = []
for user in all_user_lst_embed:
    user_ = []
    for i in range(len(user)):
        if i % 72 == 0:
            day_ = user[i:i+72]
            user_.append(day_)
    all_user_lst.append(user_)


# vectorization (Word2Vec)
MIN_SAMPLE = 0
EMBEDDING_SIZE = 100
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
CH_vec = word2vec.Word2Vec(all_user_lst_embed, iter=5, min_count=MIN_SAMPLE, sg=1, size=EMBEDDING_SIZE)
CH_embedding = np.zeros((len(CH_vec.wv.vocab), EMBEDDING_SIZE))
for i in range(len(CH_vec.wv.vocab)):
    embedding_vec = CH_vec.wv[CH_vec.wv.index2word[i]]
    if embedding_vec is not None:
        CH_embedding[i] = embedding_vec
print('shape :', CH_embedding.shape, type(CH_embedding))
# save
data_path = r"D:\USERLOG\select_user2\total"
np.save(os.path.join(data_path, 'user_iter5_{}dim.npy'.format(EMBEDDING_SIZE)), CH_embedding)
with open(os.path.join(data_path, 'user_index2word_iter5_{}dim.pkl'.format(EMBEDDING_SIZE)), 'wb') as f:
    pickle.dump(CH_vec.wv.index2word, f)
# vocab
keys = CH_vec.wv.index2word
CH_vocab = {}
for i, channel in enumerate(keys):
    CH_vocab[channel] = i + 1
CH_vocab['UNK'] = 0
with open(os.path.join(data_path, 'channel_vocab_{}dim.pkl'.format(EMBEDDING_SIZE)), 'wb') as f:
    pickle.dump(CH_vocab, f)






# TODO : build dictionary data (key:user_name, value:viewing_log)
data_path = r"D:\USERLOG\select_user2\total"

tmp_user_dic = {}
for key_, value_ in zip(all_user_name, all_user_lst):
    tmp_user_dic[key_] = value_

with open(os.path.join(data_path, 'tmp_user_dic.pkl'), 'wb') as f:
    pickle.dump(tmp_user_dic, f)

