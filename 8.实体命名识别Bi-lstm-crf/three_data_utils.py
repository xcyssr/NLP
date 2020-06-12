import pandas as pd
import pickle
import numpy as np
from tqdm import tqdm
import os
import math
import random

def get_data_with_windows(name='train'):
    with open(f'../data/prepare/dict.pkl','rb')as f:
        map_dict=pickle.load(f)# 读取dict.pkl

    def item2id(data,w2i): # 定义一个转id函数
        # 接受一个data列表,这个列表是所有的字符,再接受一个w2i字典,这个字典是字符到id
        # 转换成每一个字符对应的id
        return [w2i[x] if x in w2i else w2i['UNK'] for x in data]
                # data本身是一个字符构成的列表,全部转换成对应的id
                # 如果字符在正常运行,如果不在则对应'UNK'的id

    results=[] # 保存最终所有的数据的一个列表

#------------------------拿到train或者test文件夹下的所有文件,以列表的方式返回
    root=os.path.join('../data/prepare/',name) # 路径拼接写法
    files=list(os.listdir(root)) # 用于返回指定的文件夹包含的文件或文件夹的名字的列表。
    # print(files)#(检查站)

    for file in tqdm(files): # 进度条
        result=[]
        path=os.path.join(root,file) # 路径拼接
        samples=pd.read_csv(path,sep=',') # 读一个csv文件

        num_samples=len(samples) # 确定每一个csv文件的总长度,也是文章最后一串sep下标的位置

        sep_index=[-1]+samples[samples['word']=='sep'].index.tolist()+[num_samples]
        # pandas：根据条件获取元素所在的位置（索引）
        # 一个句子完了会有一串sep sep sep sep,根据这些sep,拿到了分割的那些行的下标
        #  0-19  21-39 41-59 60-[num_samples]..... sep此时是一个元组,下标占1,

#-----------------拿到所有的句子--------------------------------------
        for i in range(len(sep_index)-1): # 遍历上面的每一个下标
            start=sep_index[i]+1
            end=sep_index[i+1]

#-----------------开始对每一个特征进行处理----------------------------
            data=[] # 最终有6个元素,字符,标签,词性,部首,拼音的一句话
            for feature in samples.columns: # 对每一列进行截取,同时转换成下标
                data.append(item2id(list(samples[feature])[start:end],map_dict[feature][1]))
                    # 取出word,bound,flag,label,pinyin....,转换成对应id
                    # data.append(转成对应id(取出每一列的元素))
            result.append(data)
            # 每一句话放到data里去---result是一个大的列表,其中每个元素是一个句子,每个句子又是一个列表,每个列表包含6个id元素


#----------------数据增强---拼接id--------------------------------------
        two=[] # 两个句子合并的结果
        for i in range(len(result)-1): # 遍历切分好的的文本的每一个句话的下标,10个能拼9个出来,所以要-1
            first=result[i] # 拿到第一个句子
            second=result[i+1] # 拿到第二个句子
            two.append([first[k]+second[k] for k in range(len(first))]) # 两句话的id进行拼接,第一句话对应的6个元素+第二个句子对应的6元素

        three=[] # 三个句子合并的结果
        for i in range(len(result) - 2):  # 遍历上面转换的所有id
            first=result[i]
            second = result[i+1]  # 拿到第一个句子
            third = result[i + 2]  # 拿到第二个句子
            three.append([first[k] + second[k] + third[k] for k in range(len(first))])

        results.extend(result+two+three)# 单个,两个,三个句子全部放进results里
        # extend() 函数用于在列表末尾一次性追加另一个序列中的多个值

    # return results # 所有拼接的句子

    with open(f'../data/prepare/'+name+'.pkl','wb')as f: # 将results写入train.plk
        pickle.dump(results,f)

def get_dict(path): # 供给five调用dict.pkl
    with open(path,'rb')as f:
        dict=pickle.load(f)
    return dict



class BatchManager(object):
    def __init__(self,batch_size,name='train'):
        with open(f'../data/prepare/'+name+'.pkl','rb')as f:
            data=pickle.load(f) # 读train.plk
        self.batch_data=self.sort_and_pad(data,batch_size)
        self.len_data=len(self.batch_data)

    def sort_and_pad(self,data,batch_size):
        num_batch=int(math.ceil(len(data) /batch_size)) # 总共有多少个批次
        sorted_data=sorted(data,key=lambda x:len(x[0])) # 按照句子长度排序
        batch_data=list()
        for i in range(num_batch):
            batch_data.append(self.pad_data(sorted_data[i*int(batch_size):(i+1)*int(batch_size)]))
        return batch_data
    @staticmethod
    def pad_data(data):
        chars=[]
        bounds=[]
        flags=[]
        radicals=[]
        pinyins=[]
        targets=[]
        max_length=max([len(sentence[0])for sentence in data])
        for line in data:
            char,bound,flag,target,radical,pinyin=line
            padding=[0]*(max_length-len(char))
            chars.append(char+padding)
            bounds.append(bound+padding)
            flags.append(flag+padding)
            targets.append(target+padding)
            radicals.append(radical+padding)
            pinyins.append(pinyin+padding)
        return [chars,bounds,flags,radicals,pinyins,targets]

    def iter_batch(self,shuffle=False):
        """
        批次,直接可以用在模型里.
        :param shuffle:
        :return:
        """
        if shuffle:
            random.shuffle(self.batch_data)
            for idx in range(self.len_data):
                yield self.batch_data[idx] # 每次拿一个批次


if __name__ == '__main__':
    # get_data_with_windows('train')
    # train_data=BatchManager(10,'train') # 训练的时候才用,这里是一个示例
    get_data_with_windows('test')