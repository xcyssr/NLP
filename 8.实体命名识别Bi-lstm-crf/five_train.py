import tensorflow as tf
from result.three_data_utils import BatchManager,get_dict
from result.four_model import Model
import time

batch_size=20
dict_file='../data/prepare/dict.pkl'


def train():
    #一、数据准备
    train_manager=BatchManager(batch_size=20,name='train')
    test_manager = BatchManager(batch_size=100, name='test')
    #二、读取字典
    mapping_dict=get_dict(dict_file)

    #二、载入模型
    model=Model(mapping_dict)

    init=tf.global_variables_initializer()
    with tf.Session()as sess:
        sess.run(init) # 初始化
        for i in range(10):
            j=1
            for batch in train_manager.iter_batch(shuffle=True):# batch为一个批次的数据,shuffle打乱顺序
            # 封装
            #     _,loss=sess.run([model.train_op,model.cost],feed_dict={model.char_inputs:batch[0],
            #                                                            model.bound_inputs:batch[1],
            #                                                            model.flag_inputs:batch[2],
            #                                                            model.radical_inputs:batch[3],
            #                                                            model.pinyin_inputs:batch[4],
            #                                                            model.targets:batch[5]
            #                                                            })
                start=time.time()
                loss=model.run_step(sess,batch)
                end=time.time()
                if j%10==0:
                    print('epoch:{},step:{},loss:{},elapse:{},estimate:{}'.format(i+1,j,train_manager.len_data,loss,end-start,(end-start)*(train_manager.len_data)))
                j+=1
            for batch in test_manager.iter_batch(shuffle=True):
                print(model.predict(sess,batch))

if __name__ == '__main__':
    train()