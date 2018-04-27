"""
Enumeration class of tensorflow-activation methods

date: 9/18
author: arabian9ts
"""

import enum #枚举
import tensorflow as tf

class Activation(enum.Enum):   # 用class关键字定义枚举，继承类，这个class和定义类的class不同
    identity = tf.identity #### 叠加
    relu = tf.nn.relu
    softmax = tf.nn.softmax

sess = tf.Session()
sess.run(tf.global_variables_initializer())