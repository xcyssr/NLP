import os
import pandas as pd
from collections import Counter
    # counter工具用于支持便捷和快速地计数
from tqdm import tqdm
    # 进度条提示
import jieba.posseg as psg
    # jieba词性标注
from result.one_data_clear import split_text
from cnradical import Radical,RunOption# 部首、拼音提取器
import shutil # 创建一个树
from random import shuffle
import pickle

train_dir='ruijin_round1_train2_20181022'
def process_text(idx,split_method=None,split_name='train'): # 用来接收切分好的句子
    """
    读取文本---切割---然后打上标记---并提取词边界、词性、部首、拼音、等文本特征
    :param idx: 文件的名字,不含扩展名
    :param split_method: 切割文本的函数---来自上一个写好的程序
    split_name: 最终保存的文件夹名字
    :return:

    """
    data={}
#------------------------------切分句子----------------------------------------

    if split_method is None: # 如果未将切分函数传给split_method
        with open(f'../datas/{train_dir}/{idx}.txt','r',encoding='utf8')as f:
            # 打开路径的新写法,读取不同的文件
            texts=f.readlines() # 按行,做简单的切割
    else:
        with open(f'../datas/{train_dir}/{idx}.txt', 'r', encoding='utf8')as f:
            texts=f.read() # 读整篇文章
            texts=split_method(texts) # 调用上次的切分函数
    data['word']=texts
    # 从one中提取切分好的句子,然后作为一个值赋给data字典里['word']这个键


#------------------------------依据.ann人工标注,首先给   每一个字都打上标签----------------------------------------
# 一、 .txt,对每一个字标记为'O'
    tag_list=['O'for s in texts for x in s] # s遍历了text里的每一句话,再遍历每一句话里的每一个字---双重循环
    # return tag_list # (检查站: main-1用于查看给所有字标记'O')

    tag =pd.read_csv(f'../datas/{train_dir}/{idx}.ann',header=None,sep='\t') # 读取对应的.ann文件

# 二、 获取.ann人工标注中的类别和边界
    for i in range(tag.shape[0]): # 这里是做一个文件里的第一行,即一行一行读
        tag_item=tag.iloc[i][1].split(' ')
            # 取每一行的第二列,获取的实体类别以及起始位置,eg:Disease 18445 1850
        # print(tag_item) # (检查站)
        cls,start,end=tag_item[0],int(tag_item[1]),int(tag_item[-1]) # 字符串转换成整数
            # 分别抽取出实体类别,起始位置,终止位置,分别保存在cls,start,end里

# 三、 按照人工标注给已经标注为'O'的根据句子长度,标注B,I标签
        tag_list[start]='B-'+cls  # 每一句话中,给起始位置打B+类别名
        for j in range(start+1,end): # 起始位置之后...到最后的结束为止
            tag_list[j]='I-'+cls # 后面的位置打I+类别名
    assert len([x for s in texts for x in s ])==len(tag_list)
        # 保证两个序列长度一致,即需要对切分好的每一句话里面的每一个字的长度==打完标记后的每一个字的长度
        # texts是切分好的句子,s遍历这些切分好的句子,x再遍历每一个句子的每一个字,
        # text_list是整篇文章根据人工标注打的bi标签,对起始位置和结束位置标记好的实体结果
    # return  tag_list # (检查站:可断点查看)*


#-----------------------------提取词性和词边界特征----------------------------------------
    word_bounds=['M' for item in tag_list ] # 定义词边界,全部标记为M,text_list是对起始位置和结束位置标记好的实体
    word_flags=[] # 用来保存每次切好的词,并给下一次切根据长度提供起始位置
    for text in texts:
        for word,flag in psg.cut(text): # 对每一句话进行带词性切词
            if len(word)==1: # 单个字(词)时
                start=len(word_flags) # 确定起始位置,起始位0
                word_bounds[start]='S' # 对单个的词标注修改为S
                word_flags.append(flag) # 把标注好的词添加进word_flags
            else: # 当不是单个词时
                start=len(word_flags) # 确定起始位置
                word_bounds[start]='B' # 每个词的第一个字符标注位B
                word_flags+=[flag]*len(word) # 将这个词的每个字都加上词性标记
                end=len(word_flags)-1 # 确定结束位置
                word_bounds[end]='E' # 将最后一个字标注位E

    #---------统一做,本来放在上面,现在和共用一个循环-------------------------------------------------
    # 需要对text_list同样进行切分,切成texts一样长度的句子,从一整篇文章变成一句话一句话的样子
    tags=[]
    bounds=[]
    flags=[]
    start=0
    end=0
    for s in texts:  # 一句话一句话遍历整个.txt文本的句子
        l=len(s) # 计算每个切分好的句子的长度
        end+=l # 结束位置
        bounds.append(word_bounds[start:end])  # 按照切分好句子的长度对打完标签的txst_list标签进行截取
        flags.append(word_flags[start:end])  # 按照切分好句子的长度对打完标签的txst_list标签进行截取
        tags.append(tag_list[start:end])
        start+=l # 定位下一句话的起始位置

    data['bound']=bounds # 词边界特征
    data['flag']=flags # 词性特征
    data['label']=tags


# -----------------------------提取拼音特征----------------------------------------
    radical=Radical(RunOption.Radical) # 提取偏旁部首
    pinyin=Radical(RunOption.Pinyin) # 提取拼音
    # 通过库提取偏旁特征,对没有偏旁的标注位 UNK
    data['radical']=[[radical.trans_ch(x) if radical.trans_ch(x) is not None else 'UNK' for x in s ]for s in texts] # 列表推导式
    # 提取拼音特征,对没有拼音的标注PAD
    data['pinyin'] = [[pinyin.trans_ch(x) if pinyin.trans_ch(x) is not None else 'UNK' for x in s] for s in texts]  # 列表推导式

    # return texts[1],tags[1],bounds[1],flags[1],data['radical'][1],data['pinyin'][1] # (检查站)
    # return (len(data['label'])) #(检查站)



#----------------------------------合并和存储切分好的文件---------------------------------------
    num_samples=len(texts) # 有多少句话
    num_col=len(data.keys()) # 统计有多少列

    dataset=[]
    for i in range(num_samples):
        records=list(zip(*[list(v[i])for v in data.values()])) # 将6个列表分别拆分成一个字一个列表,*=解压,无*=压缩
        dataset+=records+[['sep']*num_col] # 每存完一个句子需要一行sep进行隔离,sep为元组格式
    dataset=dataset[:-1]# 最后一行sep不要

    dataset=pd.DataFrame(dataset,columns=data.keys())# 转换成数据框
    save_path=f'../data/prepare/{split_name}/{idx}.csv'

    #print(dataset) # (检查站)

    def clean_word(w): # 由于已经标注完成,可以清理换行,空格等符号
        if w=='\n': # 换行
            return 'LB'
        if w in[' ','\t','\u2003']: # 空格,TAB,中文空格
            return 'SPACE'
        if w.isdigit(): # 命名实体识别不关心数字是多少,所以把数字变成一个类
            return 'num'
        return w

    dataset['word']=dataset['word'].apply(clean_word) # 存储之前进行清洗
    dataset.to_csv(save_path,index=False,encoding='utf-8')

#----------------------------分测试集和验证集--------------------
def multi_process(split_method=None,train_ratio=0.8):
    """
    拿80%来做训练
    :param split_method:
    :param train_ratio:
    :return:
    """
    if os.path.exists('../data/prepare/'): # 如果有
        shutil.rmtree('../data/prepare/') # 删除
    if not os.path.exists('../data/prepare/train/'): # 创建目录
        os.makedirs('../data/prepare/train')
        os.makedirs('../data/prepare/test')
    idxs=list(set([file.split('.')[0] for file in os.listdir('../datas/'+train_dir)])) # 获取所有文件的名字
    shuffle(idxs)# 打乱顺序
    index=int(len(idxs)*train_ratio) # 取80%作为训练集
    train_ids=idxs[:index] # 取训练集文件名集合
    test_ids=idxs[index:] # 取测试集训练集合

#------------------------cpu多线程----------------------------------------------
    import multiprocessing as mp
    num_cpus=mp.cpu_count() # 获取cpu的个数
    # 做一个线程池
    pool=mp.Pool(num_cpus)
    results=[]
    for idx in train_ids: # 处理训练集,保存到train_results
        result=pool.apply_async(process_text,args=(idx,split_method,'train')) # 异步并传入参数
        results.append(result)
    for idx in test_ids: # 处理测试集,保存到text_results
        result=pool.apply_async(process_text,args=(idx,split_method,'test')) # 异步并传入参数
        results.append(result)
    pool.close() # 关闭进程池
    pool.join()
    [r.get for r in results] # 因为没有返回值,该句无效
    #return  texts[0],tags[0],bounds[0],flags[0],data['radical'][0],data['pinyin'][0] #(检查站)

#-------------------------------统计字典---------------------------------------------
def mapping(data,threshold=10,is_word=False,sep='sep',is_label=False):
    count=Counter(data) # 自动统计列表里有多种类,每一类有多少个,返回的是字典
    if sep is not None: # 去掉之前为了区分切分位置所添加的sep
        count.pop(sep)
    if is_word: # 判断是不是词,
        count['PAD']=100000001
            # 专门造了一个PAD,因为一个批次的句子送进去句子长度不一致,所以将句子填充成一致的长度,才能送进神经网络
            # ------------
            # ----------pp
            # -------ppppp
            # -----ppppppp
            # 如果不是词,则不用加PAD和UNK
        count['UNK']=100000000
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)
            # 按照降序排列,返回的是列表,排序的目的是为了让尽可能让一个批次的句子尽可能等长,加快运算时间

        data = [x[0] for x in data if x[1] >= threshold]
            # 去掉频率小于10的元素---即未登陆词,因为某些偏旁部首没有在训练集中出现,测试集就没有这个部首,所以要去掉训练集出现频率较低的值
            # 基本在部首和拼音这两个类里会出现这种状况,词性和边界出现的可能性则不大
        id2item=data
        item2id={id2item[i]:i for i in range(len(id2item))}
    elif is_label:
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)  # 按照降序排列,返回的是列表
        data = [x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))}
    else:
        count['PAD'] = 100000001
        data = sorted(count.items(), key=lambda x: x[1], reverse=True)  # 按照降序排列,返回的是列表
        data=[x[0] for x in data]
        id2item = data
        item2id = {id2item[i]: i for i in range(len(id2item))} # 映射
    return id2item,item2id
#------------------------------------映射-------------------------------------
# 统计不同特征做成字典的映射
def get_dict(): # 将data映射成字典
    map_dict={}
    from glob import glob # 从.csv文件里拿到每一类所有数据
    all_w=[] # 字
    all_bound=[] # 边界
    all_flag=[] # 词性
    all_label=[] # 标签
    all_radical=[] # 部首
    all_pinyin=[] # 拼音

    for file in glob('../data/prepare/train/*.csv')+glob('../data/prepare/test/*.csv'):
        df=pd.read_csv(file,sep=',') # # 每读一个.csv文件,
        all_w += df['word'].tolist() # 每拿到一个word添加进入all_w,转换成列表
        all_bound += df['bound'].tolist()# 每拿到一个bound添加进入all_bound,转换成列表
        all_flag += df['flag'].tolist()# 同上
        all_label += df['label'].tolist()#同上
        all_radical += df['radical'].tolist()#同上
        all_pinyin += df['pinyin'].tolist()#同上

    map_dict['word']=mapping(all_w,threshold=20,is_word=True)
    map_dict['bound']=mapping(all_bound)
    map_dict['flag']=mapping(all_flag)
    map_dict['label'] = mapping(all_label, is_label=True)
    map_dict['radical']=mapping(all_radical)
    map_dict['pinyin']=mapping(all_pinyin)

    return map_dict
    # with open(f'../data/prepare/dict.pkl','wb')as f:
    #     pickle.dump(map_dict,f) # 保存到f里面去


if __name__ == '__main__':
    # print(process_text('0',split_method=split_text,split_name='train')) # main-1

    # print(set([file.split('.')[0] for file in os.listdir('../datas/'+train_dir)]))

    # multi_process(split_text) # 切分函数
    # get_dict() # 映射函数
    print(get_dict())
    #
    # with open(f'../data/prepare/dict.pkl','rb')as f: # 读dict.plk查看
    #     data=pickle.load(f)
    # print(data['bound'])