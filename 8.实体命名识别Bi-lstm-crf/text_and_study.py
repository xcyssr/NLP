import tensorflow as tf
import numpy as np

matrix=np.array([
    [1,1,1,1,1,1,],
    [2,2,2,2,2,2,],
    [3,3,3,3,3,3,],
    [4,4,4,4,4,4,],
])
# 矩阵
x=np.array([
    [0,2,1,1,2],
    [3,2,0,2,2,]
])
# 下标

result=tf.nn.embedding_lookup(matrix,x)
# 根据下标在相应的矩阵里找

with tf.Session() as sess:
    print(sess.run(result)) # 2 × 5 × 6 的矩阵