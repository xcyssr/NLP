import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib.crf import crf_log_likelihood,viterbi_decode # crf的对数似然

def network(inputs,shapes,num_tags,lstm_dim=100,initializer=tf.truncated_normal_initializer()):
    """
    外部函数,做预测使用
    接收一个批次样本的特征数据,计算出网络的输出值
    :lengths:之前对数据做了PAD填充,现在为了送入模型不计算,需要每句话的实际长度
    :param char:
    :param bound:
    :param flag:
    :param radical:
    :param pinyin:
    :return:
    """
#---------------------特征嵌入:将所有特征的id转换成一个固定长度的向量,然后拼接------------------------------------------------------

    # 词向量的嵌入
    # 下面的代码把一个字的5个特征全部映射成一个31长度的向量
    embedding=[]
    keys=list(shapes.keys())
    for key in shapes.keys():
        with tf.variable_scope(key+'_embedding'):  # 变量空间
            lookup = tf.get_variable(
                name=key+'_embedding',  # 给变量指定一个名字
                shape=shapes[key],  # 指定形状
                initializer=initializer  # 初始化器
            )
            embedding.append(tf.nn.embedding_lookup(lookup,inputs[key]))
                # 把key里的内容全部映射成向量,本来分开写,为简便写成一个循环,实现特征嵌入
    embed=tf.concat(embedding,axis=-1)
        # 因为要合起来成为一个向量送进神经网络里
        # 所以要拼接,在最后一个维度上,shape[None × None × char_dim + bound_dim + flag_dim + radical_dim + pinyin_dim]

    sign=tf.sign(tf.abs(inputs[keys[0]]))
        # 计算出每句话的长度, sign=符号(绝对值(inputs['char'])),为保证传进来数值正确,所以变为keys=list(shapes.keys())---sign=符号(绝对值(inputs[keys][0]))---[None(批次),None(每个句子填充的长度)]
        # tf.sign--- -1 0 1
    lengths=tf.reduce_sum(sign,reduction_indices=1)
        # 为了防止1个字符的句子,所以在第二个维度进行求和,1在这里是第二个维度,这样即便只有1个字符的句子,仍能组成一个列表
        # 统计1有多少个,就能算出实际有多少个字符,也就是求出未填充PAD前的句子长度

    num_time=tf.shape(inputs[keys[0]])[1] # 序列长度


#---------------------循环神经网络,BiLstm-双层双向-------------------------------------------------------
    with tf.variable_scope('BiLstm_layer1'):
        lstm_cell={}
        for name in ['forward','backward',]: # 第一层正反
            with tf.variable_scope(name): # 命名空间
                lstm_cell[name]=rnn.BasicLSTMCell(
                    lstm_dim# 神经元个数
                )
        outputs1,finial_states1=tf.nn.bidirectional_dynamic_rnn(  # 双向动态rnn
            lstm_cell['forward'], # 第一层正向
            lstm_cell['backward'], # 第一层反向
            embed, # 将5类数据映射成向量作为输入
            dtype=tf.float32,
            sequence_length=lengths, # 未填充PAD的句子真实长度,(可给可不给)
        )
#------------------第一层的输出,第二层的输入------------
    outputs1=tf.concat(outputs1,axis=-1) # 拼接,b,L,2 × lstm_dim
#------------------第二层-------------------------------
    with tf.variable_scope('BiLstm_layer2'):
        lstm_cell = {}
        for name in ['forward', 'backward', ]:  # 第一层正反
            with tf.variable_scope(name):  # 命名空间
                lstm_cell[name] = rnn.BasicLSTMCell(
                    lstm_dim  # 神经元个数
                )
        outputs, finial_states1 = tf.nn.bidirectional_dynamic_rnn(  # 双向动态rnn
            lstm_cell['forward'],  # 第一层正向
            lstm_cell['backward'],  # 第一层反向
            outputs1,  # 将映射好的向量作为输入
            dtype=tf.float32,
            sequence_length=lengths,  # 未填充PAD的句子真实长度,(可给可不给)
        )
#-----------------第二层输出---------------------------
    output = tf.concat(outputs, axis=-1)  # batch_size , maxlength , 2 × lstm_dim

#-----------------第一层输出映射-----------------------------
    output=tf.reshape(output,[-1,2*lstm_dim]) # reshap成二维矩阵
    with tf.variable_scope('project_layer1'): # 第一层映射
        w=tf.get_variable(
            name='w',
            shape=[2*lstm_dim,lstm_dim],
            initializer=initializer,
        )
        b=tf.get_variable(
            name='b',
            shape=[lstm_dim],
            initializer=tf.zeros_initializer
        )
        output=tf.nn.relu(tf.matmul(output,w)+b) # relu=激活
#----------------第二层输出映射-----------------------------
    with tf.variable_scope('project_layer2'): # 第一层映射
        w=tf.get_variable(
            name='w',
            shape=[lstm_dim,num_tags],
            initializer=initializer,
        )
        b=tf.get_variable(
            name='b',
            shape=[num_tags],
            initializer=tf.zeros_initializer
        )
        output=tf.matmul(output,w)+b
        output=tf.reshape(output,[-1,num_time,num_tags])

    return output,lengths # [?,?,31]
        # batch_size, max_length,num_tags
        # 网络的输出,序列每一句话的真是长度



class Model(object): # 包含tf里的东西,输入,输出,计算,损失,优化器等
        # 用到的参数值
    def __init__(self,dict,lr=0.0001): # 实例化(接收dict.pkl,用于统计长度)  dict['word]=[我 你 她...],{我=1,你=2,她=3...}

            # 矩阵的行
        self.num_char=len(dict['word'][0]) # 测试第一个列表的长度就知道有多少个字
        self.num_bound = len(dict['bound'][0]) # ...多少标签
        self.num_flag = len(dict['flag'][0]) # ...多少个词性
        self.num_radical = len(dict['radical'][0]) # ...多少个部首
        self.num_pinyin = len(dict['pinyin'][0]) # ...多少个拼音
        self.num_tags = len(dict['label'][0]) # ...多少实体类型(源自人工标注)

            # 矩阵的列
        self.char_dim = 100 # 手动指定字符映射成100长度的向量
        self.bound_dim = 20 # ...指定标签映射成20长度的向量
        self.flag_dim = 50 # ...指定词性映射成50长度的向量
        self.radical_dim = 50 # ...指定部首映射成50长度的向量
        self.pinyin_dim = 50 # ...指定拼音映射成100长度的向量

        self.lstm_dim=100 # lstm的神经元个数
        self.lr=lr # 学习率

        self.map=dict


#---------------流程构造-----------------------------------------------
#---------------定义接受数据的placeholder------------------------------
        self.char_inputs=tf.placeholder(dtype=tf.int32,shape=[None,None],name='char_inputs')
        self.bound_inputs=tf.placeholder(dtype=tf.int32,shape=[None,None],name='bound_inputs')
        self.flag_inputs=tf.placeholder(dtype=tf.int32,shape=[None,None],name='flag_inputs')
        self.radical_inputs=tf.placeholder(dtype=tf.int32,shape=[None,None],name='radical_inputs')
        self.pinyin_inputs=tf.placeholder(dtype=tf.int32,shape=[None,None],name='pinyin_inputs')
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name='targets') # 真实值

        self.global_step=tf.Variable(0,trainable=False) # 关闭训练,不更新求导,只计数

#----------------计算模型输出值-----------------------------------------
        self.logits,self.lengths = self.get_logits(self.char_inputs,
                                                     self.bound_inputs,
                                                     self.flag_inputs,
                                                     self.radical_inputs,
                                                     self.pinyin_inputs
                                                    )   # 接受了网络的输出和真实长度
                                                        # 传给 def get_logits()
#-----------------计算损失-----------------------------------------------
        self.cost = self.loss(self.logits,self.targets,self.lengths) # loss(输出,目标,真实长度)

#-----------------优化器-------------------------------------------------
        # 采用梯度截断,防止梯度爆炸
        with tf.variable_scope('optimizer'):
            opt=tf.train.AdadeltaOptimizer(self.lr) # 学习率
            grad_vars=opt.compute_gradients(self.cost)  # 损失函数的到数值(所有参数的导数)
            clip_grad_vars=[[tf.clip_by_value(g,-5,5),v]for g,v in grad_vars] # 按照值来梯度截断 4*2 +5=13,两层全连接,w b*2 +转移矩阵=18
            self.train_op=opt.apply_gradients(clip_grad_vars,self.global_step)
                # 使用阶段后的梯度,对参数进行更新


#-----------------保存模型---------------------------------------------
        self.saver=tf.train.Saver(tf.global_variables(),max_to_keep=5)
            # 只保留最近的5次模型





    def get_logits(self,char,bound,flag,radical,pinyin):
            # 定义网络,接受一个批次的样本
        """
            一、负责接受一个批次(暂定10个)的样本数据,计算出网络的输出值
        :param char: type of int,id of chars,a tensor of shape 2-D[None(批次),None(句子长度)]
        :param bound: 同上
        :param flag:  同上
        :param radical: 同上
        :param pinyin: 同上
        :return:3-d tensor [batch_size,max_length,num_tags]
        """

        shapes={}
        shapes['char']=[self.num_char,self.char_dim] # 形状['char]=[行,列]
        shapes['bound'] = [self.num_bound, self.bound_dim]
        shapes['flag'] = [self.num_flag, self.flag_dim]
        shapes['radical'] = [self.num_radical, self.radical_dim]
        shapes['pinyin'] = [self.num_pinyin, self.pinyin_dim]

        inputs={}
        inputs['char']=char
        inputs['bound']=bound
        inputs['flag'] = flag
        inputs['radical'] = radical
        inputs['pinyin'] = pinyin

        return network(inputs,shapes,lstm_dim=self.lstm_dim,num_tags=self.num_tags)


#------------------------crf条件随机场------------------------------------------------------
        #二、 用到了crf的似然函数,对crf进行了嫁接,即把crf里用来提取特征的部分,在本来的crf++(用的传统统计算法统计出每一个时刻的分值)算法中,
    def loss(self,output,targets,lengths):
        b=tf.shape(lengths)[0] # 拿到第一个维度的长度
        num_steps=tf.shape(output)[1] # 序列长度
        with tf.variable_scope('crf_loss'):
            small=-1000.0 # 代表概率接近0,log -1000
            start_logits=tf.concat(
                [small*tf.ones(shape=[b,1,self.num_tags]),tf.zeros([b,1,1])],axis=-1
            )
                # 在第一个时刻加一个时刻,前面31个是-1000,只有第32个概率是1

            pad_logits=tf.cast(small*tf.ones([b,num_steps,1]),tf.float32)
                # 在每一个时刻加一个状态

            logits=tf.concat([output,pad_logits],axis=-1)
            logits=tf.concat([start_logits,logits],axis=1)
            targets=tf.concat(
                [tf.cast(self.num_tags*tf.ones([b,1]),tf.int32),targets],axis=-1
            )
            self.trans=tf.get_variable(
                name='trans',
                shape=[self.num_tags+1,self.num_tags+1],
                initializer=tf.truncated_normal_initializer(),
            )
            log_likehood,self.trans=crf_log_likelihood(  # 每一个样本返回一个值
                inputs=logits,
                tag_indices=targets,
                transition_params=self.trans,
                sequence_lengths=lengths,
            )

            return tf.reduce_mean(-log_likehood)  # 所有的平均值


    def run_step(self,sess,batch,istrain=True): #用于自动化提取数据接口
        if istrain:
            feed_dict={self.char_inputs:batch[0],
                       self.bound_inputs:batch[1],
                       self.flag_inputs:batch[2],
                       self.radical_inputs:batch[3],
                       self.pinyin_inputs:batch[4],
                       self.targets:batch[5]
            }
            _, loss = sess.run([self.train_op, self.cost], feed_dict=feed_dict)
            return loss
        else:
            feed_dict = {self.char_inputs: batch[0],
                         self.bound_inputs: batch[1],
                         self.flag_inputs: batch[2],
                         self.radical_inputs: batch[3],
                         self.pinyin_inputs: batch[4],
                         }

            logits,lengths= sess.run([self.logits,self.lengths],feed_dict=feed_dict)
            return logits,lengths

        
    def decode(self,logits,lengths,matrix):
        # 预测解码(每一时刻隐状态的输出值,真实长度,转移矩阵)
        # 维特比算法解码
        paths=[]
        small=-1000.0
        start=np.asarray([[small]*self.num_tags,+[0]]) # 二维数组
        for score,length in zip(logits,lengths):
            score=score[:length] # 只取有效字符的输出
            pad=small*np.ones([length,1])
            logits=np.concatenate([score,pad],axis=-1)
            logits=np.concatenate([start,logits],axis=0)

            path,_=viterbi_decode(logits,matrix)# 维特比解码
            paths.append(path[1:])
        return paths # 解码出的id


    def predict(self,sess,batch):
        results=[]
        # 对一个批次进行预测
        matrix=self.trans.eval() # 转移矩阵
        logits,lengths=self.run_step(sess,batch,istrain=False)
        paths=self.decode(logits,lengths,matrix)
        chars=batch[0] # 拿到字对应的id
        for i in range(len(paths)):
            length=lengths[i]
            string=[self.map['word'][0][index]for index in chars[i][:length]] # 第i句话真是的数据(id),转换成每一个id对应的字
            tags=[self.map['label'][0][index] for index in paths[i]]
            result=[k for k in zip(string,tags)]
            results.append(result)
        return results




            















