
import tensorflow
import os
import codecs # 读取库
import re
import numpy as np


train_dir='../datas/ruijin_round1_train2_20181022'
def get_entities(dir):# (---main-1)
    """
    :一、获取实体,返回实体种类总数列表---.ann
    :param dir:
    :return:
    """
    entities={} # 用来存储实体,要用{ },不能用[ ]
    files=os.listdir(dir) # 抽取文件夹下所有的目录文件
    files=list(set([file.split('.')[0] for file in files])) # 去扩展名,set去重,list表单
    for file in files:
        path=os.path.join(dir,file+'.ann') # 路径拼接,回+后缀
        with open(path,'r',encoding='utf8')as f:  # 读已经抽取的所有ann文件
            for line in f.readlines(): # 读每一个文件
                #print(line.split('\t')[1].split(' ')[0])
                name=line.split('\t')[1].split(' ')[0]
                if name in entities: # 类别统计
                    entities[name]+=1
                else:
                    entities[name]=1
    return entities


def ischinese(char):
    """
    定义字符是否为中文
    :param char:
    :return:
    """
    if '\u4e00' <= char <= '\u9fff': # 中文范围,应该记住
        return True
    return False


def get_labelencoder(entities):
    """
    二、得到标签和下标的映射(俗称打标签)---.ann
    :param entities:
    :return:
    """
    entities=sorted(entities.items(),key=lambda x: x[1],reverse=True) #
    # entities = 排序(列表返回可遍历元组,参数1,参数2)
    entities=[x[0] for x in entities]
    id2label=[]
    id2label.append('O') # 先给所有打上"O"标签,方便以后操作
    for entity in entities:
        id2label.append('B-'+entity) # 给同类第一个加上B标签
        id2label.append('I-'+entity) # 给同类其余加上I标签
    label2id={id2label[i]:i for i in range(len(id2label))} # 遍历所有的类添加B,I标签
    return id2label,label2id # id2lable=O,label2id=B,I

def split_text(text):
    """
    三、数据预处理---对所有文章进行切分---.txt
    :param text:
    :return:
    """
    split_index=[]
    # 可能性一
    pattern1='。|，|,|;|；|\.|\?'
    for m in re.finditer(pattern1,text):
        idx=m.span()[0] # idx定位为符号的下标
        if text[idx - 1]=='\n': # 如果标点出现在最后，即准备换行时
            continue # 不作为分割点
        if text[idx - 1].isdigit()and text[idx + 1].isdigit(): # 前后是数字
            continue
        if text[idx - 1].isdigit()and text[idx + 1].isspace() and text[idx+2].isdigit(): # 钱数字,后空格,后后数字
            continue
        if text[idx - 1].islower()and text[idx + 1].islower():#前小写字母,后小写字母
            continue
        if text[idx - 1].islower() and text[idx + 1].isdigit():  # 检查参数前是小写字母，参数后是数字
            continue
        if text[idx - 1].isupper() and text[idx + 1].isdigit():# 检查参数前是大写字母，参数后是数字
            continue
        if text[idx - 1].isdigit() and text[idx + 1].islower():# 检查参数前是数字，参数后是小写字母
            continue
        if text[idx - 1].isdigit() and text[idx + 1].isupper():# 检查参数前是数字，参数后是小写字母
            continue
        if text[idx + 1]in set('.。;；,，'): # 连续两个符号时,
            continue
        if text[idx - 1].isspace()and text[idx-2].isspace()and text[idx-3]=='C': # 前一位是空格,前两位是空格,前三位是字符C
            continue
        if text[idx - 1].isspace()and text[idx-2]=='C':
            continue
        if text[idx - 1].isupper()and text[idx+1].isupper(): # 前面是大写,后面是大写
            continue
        if text[idx]=='.'and text[idx+1:idx+4]=='com':
            continue
        split_index.append(idx+1) # 不满足上面的条件,则切开,从标点符号后面开始切

    # 可能性二
    pattern2='\([一二三四五六七八九零十]\)|[一二三四五六七八九零十]、|'
    pattern2+='注:|附录 |表 \d|Tab \d+|\[摘要\]|\[提要\]|表\d[^。，,;]+?\n|图 \d|Fig \d|'
    pattern2+='\[Abstract\]|\[Summary\]|前  言|【摘要】|【关键词】|结    果|讨    论|'
    pattern2+='and |or |with |by |because of |as well as '

    for m in re.finditer(pattern2,text):
        idx=m.span()[0] # 定位初始下标
        if (text[idx:idx+2] in ['or','by'] or text[idx:idx+3]=='and' or text[idx:idx+4]=='with')\
            and (text[idx-1].islower() or text[idx-1].isupper()): # 上一行结尾的\相当于续行
            continue
        split_index.append(idx)

    # 可能性三
    pattern3 ='\n\d\.'# 匹配1.  2.  这些序号
    for m in re.finditer(pattern3, text):
        idx = m.span()[0] # 定位初始下标
        if ischinese(text[idx + 3]):  # 如果后面3个是中文
            split_index.append(idx+1) # 从该符号后面切割

    for m in re.finditer('\n\(\d\)', text):  # 如果是(1) (2)这样的序号
        idx = m.span()[0]
        split_index.append(idx + 1) # 直接切
    split_index = list(sorted(set([0, len(text)] + split_index)))
    # 加入后第一次排序
    # split_index=表(排序(去重(0,最后+之前的)))
    # sorted的返回值是一个列表



# 处理标题的换行问题
    other_index=[]
    for i in range(len(split_index)-1):
        begin=split_index[i] # 起始位置
        end=split_index[i+1] # 终止位置
        if text[begin] in '一二三四五六七八九零十' or \
                (text[begin]=='(' and text[begin+1]in '一二三四五六七八九零十'): # 一... (一)...
            for j in range(begin,end):
                if  text[j] == '\n':
                    other_index.append(j+1) # 从换行符后面切
    # 加进去,再排序
    split_index+=other_index
    split_index = list(sorted(set([0, len(text)] + split_index))) # 加入other切分的句子后第二次排序


# 长句子拆成短句子
    other_index=[]
    for i in range(len(split_index)-1):
        b=split_index[i]
        e=split_index[i+1]
        other_index.append(b)
        if e-b>150: # 至少150个字符
            for j in range(b,e):
                if (j+1-other_index[-1]) > 15: # 保证句子长度在15以上
                    if text[j]=='\n':
                        other_index.append(j+1) # 换行后面拆分
                    if text[j]==' ' and text[j-1].isnumeric() and text[j+1].isnumeric():
                        other_index.append(j+1) # 换行后面拆分
    # 加进来,再排序
    split_index+=other_index
    split_index = list(sorted(set([0, len(text)] + split_index)))  # 截断后第三次排序


# 删除句子之间的空格
    for i in range(1,len(split_index)-1): # 变相合并全部为空格的句子
        idx=split_index[i] # 拿到一个下标
        while idx >split_index[i-1]-1 and text[idx-1].isspace():
            # 若idx下标大于前一个下标,并且前一个下标为空
            idx-=1
            # 10  <---   20 == 1020
        split_index[i]=idx # 空格之后的下标
    split_index = list(sorted(set([0, len(text)] + split_index)))  # 去掉空格后的第四次排序

# 短句子处理---拼接
    temp_idx=[]
    i=0
    while i < len(split_index)-1: # 0 10 20 30 45  (合并)
        b=split_index[i]
        e=split_index[i+1]
        num_ch=0
        num_en=0
        if e-b<15: # 如果句子小于15个字符
            for ch in text[b:e]:
                if ischinese(ch):
                    num_ch+=1 # 统计中文频数
                elif ch.islower() or ch.isupper:
                    num_en+=1 # 统计英文频数
                if num_ch+0.5*num_en>5: # 汉字个数+英文个数*0.5>5时,不管
                    temp_idx.append(b)
                    i+=1
                    break
            if num_ch+0.5*num_en<=5: # 汉字的个数+英文个数*0.5<5时,合并
                temp_idx.append(b)
                i+=2
        else: # 若>15,不合并,将句子添加进去,i向后走一个段
            temp_idx.append(b)
            i+=1
    split_index = list(sorted(set([0, len(text)] + temp_idx))) # 第四次排序
    # 储存切分的结果
    result=[]
    for i in range(len(split_index)-1):
        result.append(text[split_index[i]:split_index[i+1]])
        # 从i到i+1,全部添加到result里

    # 做一个检查
    s=''
    for r in result:
        s+=r
    assert len(s)==len(text)
    return result


    # # #输出最长和最短的句子---(main-5)---[:-1]
    # lens=[split_index[i+1]-split_index[i]for i in range(len(split_index)-1)][:-1]
    # print(max(lens),min(lens))
    # # 输出切分文件查看
    # for i in range(len(split_index)-1):
    #     print(i,'|||||',text[split_index[i]:split_index[i+1]])



if __name__ == '__main__':
#     #读实体数目和长度---(main-1)
#     print(get_entities(train_dir))
#     print(len(get_entities(train_dir)))
#     print()
#
# # #-------------------------------------------
#     #查看打好的标签---(main-2)
#     entities=get_entities(train_dir)
#     label=get_labelencoder(entities)
#     print(label)
#-------------------------------------------

    # 寻找标点
    # pattern = '。|，|,|;|；|\.'
    # with open('E:/学习代码/知识图谱/ruijin_round1_train2_20181022/0.txt','r',encoding='utf8')as f:
    #     text=f.read()
    #     for m in re.finditer(pattern,text):
    #         print(m)
    #         start=m.span()[0]-5
    #         end=m.span()[1]+5
    #         print('****',text[start:end],'*****')
    #         print(text[start+5])
#-------------------------------------------

    # # 寻找数据中切分有误的地方（可重复使用）
    # files=os.listdir(train_dir)
    # files=set([file.split('.')[0]for file in files])
    # # pattern1 = '。|，|,|；|\.|\?'
    # # pattern2 = '\([一二三四五六七八九零十]\)|[一二三四五六七八九零十]、|'  # \( 匹配括号里的
    # # pattern2 += '注:|附录 |表 \d|Tab \d+|\[摘要\]|\[提要\]|表\d[^。，,;]+?\n|图 \d|Fig \d'
    # # pattern2 += '\[Abstract\]|\[Summary\]|\[Summary\]|前  言|【摘要】|【关键词】|结    果|讨    论|'
    # # pattern2 += 'and |or |with |by |because of |as well as '
    # pattern2 = '\n\(\d\)'
    # for file in files:
    #     path = os.path.join(train_dir, file + '.txt')  # 路径拼接,回+后缀
    #     with open(path, 'r', encoding='utf8')as f:  # 读已经抽取的所有ann文件
    #         text=f.read()
    #         for m in re.finditer(pattern2, text):
    #             idx = m.span()[0]  # 符号的下标
    #             #if text[idx - 1] == '\n':
    #             #if text[idx-1].islower()and text[idx+1].islower():
    #             #if text[idx - 1].islower() and text[idx + 1].isdigit():
    #             #if text[idx + 1] in set(',，.。;；'):
    #             #if text[idx - 1].isspace() and text[idx - 2].isspace() and text[idx - 3] == 'C':
    #             #if text[idx] == '.' and text[idx + 1:idx + 4] == 'com':
    #             # ...
    #             #if ischinese(text[idx+2]):
    #             print(file+'|||',text[idx-10:idx+10])

#-------------------------------------------
#测试某个文件的切分结果---(main-5)
    files = os.listdir(train_dir)
    files = list(set([file.split('.')[0] for file in files]))

    path = os.path.join(train_dir,files[1] + '.txt')  # files[可以换随便单个数据]
    with open(path, 'r', encoding='utf8')as f:
        text=f.read()
        print(split_text(text))

#--------------------------------------
    # 查看每一个文件最后一个/n(---main-6)
    # l=[]
    # files = os.listdir(train_dir)
    # files = list(set([file.split('.')[0] for file in files])) # 从第0个文件开始
    #
    # for file in files:
    #     path = os.path.join(train_dir, file + '.txt') # 查看每一个.txt文件
    #     with open(path, 'r', encoding='utf8')as f:
    #         text = f.read()
    #         l.append(split_text(text)[-1])
    # print(l)