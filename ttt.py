import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model.vgg16 import *
from util import *
import sys
import _pickle as pickle

#全局变量
dataset_num=10000
batch=64
epoch=10

images=[]
labels=[]

def gen_onehot_list(label=0):

    return [1 if i==label else 0 for i in range(0,10)]  ##

def load_data():
    with open ('dataset/data_batch_1','rb') as  f:
        data=pickle.load(f,encoding='latin-1')
        # data = pickle.load (f, encoding='latin-1')
        slicer=int(dataset_num*0.8)
        train_images=np.array(data['data'][:slicer])/255
        train_labels=np.array(data['labels'][:slicer])
        test_images=np.array(data['data'][slicer:])/255
        test_labels=np.array(data['labels'][slicer:])
        reshaped_train_images=np.array([x.reshape([32,32,3]) for x in train_images])
        reshaped_train_labels=np.array([gen_onehot_list(i) for i in train_labels])
        reshaped_test_images=np.array([x.reshape([32,32,3]) for x in test_images])
        reshaped_test_labels=np.array([gen_onehot_list(i) for i in test_labels])  #int

    return reshaped_train_images,reshaped_train_labels,reshaped_test_images,reshaped_test_labels

def get_next_batch(max_length,length=batch,is_training=True):
    train_images, train_labels, test_images, test_labels = load_data()###
    if is_training:
        indicies=np.random.choice(max_length,length)
        next_batch=train_images[indicies]
        next_labels=train_labels[indicies]
    else:
        indicies=np.random.choice(max_length,length)
        next_batch=test_images[indicies]
        next_labels=test_labels[indicies]
    return np.array(next_batch),np.array(next_labels)

def test():
    # train_images, train_labels, test_images, test_labels = load_data()###
    images,labels=get_next_batch(max_length=len(test_labels),length=100,is_training=False)
    with tf.Session() as sess:
        result=sess.run(predict,feed_dict={input:images})

    correct=0
    total=100
    for i in range(len(labels)):
        pass

with tf.Session()as sess:
    args=sys.argv
    vgg=VGG16()
    w=tf.Variable(tf.truncated_normal([512,10],0.0,1.0)*0.01,name='w_last')
    b=tf.Variable(tf.truncated_normal([10],0.0,1.0)*0.01,name='b_last')

    input=tf.placeholder(shape=[None,32,32,3],dtype=tf.float32)
    fmap=vgg.build(input,is_training=True)
    predict=tf.nn.softmax(tf.add(tf.matmul(fmap,w),b))

    ans=tf.placeholder(shape=None,dtype=tf.float32)
    ans=tf.squeeze(tf.cast(ans,tf.float32)) ###

    loss=tf.reduce_mean(-tf.reduce_sum(ans*tf.log(predict),reduction_indices=[1]))
    optimizer=tf.train.GradientDescentOptimizer(0.05)
    train_step=optimizer.minimize(loss)

    sess.run(tf.global_variables_initializer())

    train_images, train_labels, test_images, test_labels = load_data ()

    if 2==len(args) and 'eval'==args[1]:
        saver=tf.train.Saver()
        saver.restore(sess,'./params.ckpt')
        test()
        sys.exit()

    lossbox=[]
    for e in range(epoch):
        for b in range(int(dataset_num/batch)):
            batch,actuals=get_next_batch(len(train_labels))
            sess.run(train_step,feed_dict={input:batch,ans:actuals})

            if (b+1)%100==0:
                test()

        lossbox.append(sess.run(loss,feed_dict={input:batch,ans:actuals}))

    saver=tf.train.Saver()
    saver.save(sess,'./params.ckpt')

    plt.plot(np.array(range(epoch)),lossbox)
    plt.show()



