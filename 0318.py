# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

#example_six:matploblib

#def hidden_layer(inputs,in_size,out_size,activation_function=None):
#    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
#    biase=tf.Variable(tf.zeros([1,out_size])+0.5)
#    Wx_plus_b=tf.matmul(inputs,Weights)+biase
#    if activation_function is None:
#        outputs=Wx_plus_b
#    else:
#        outputs=activation_function(Wx_plus_b)
#    return outputs
##
#x_data=np.linspace(-1,1,300)[:,np.newaxis]
#noise=np.random.normal(0,0.05,x_data.shape)
#y_data=np.square(x_data)-0.5+noise
#
#xs=tf.placeholder(tf.float32,[None,1])
#ys=tf.placeholder(tf.float32,[None,1])
#
#l1=hidden_layer(xs,1,10,activation_function=tf.nn.relu)
#pred=hidden_layer(l1,10,1,activation_function=None)
#
#loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-pred),
#                    reduction_indices=[1]))
#train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
#init=tf.global_variables_initializer()
#sess=tf.Session()
#sess.run(init)
#
#fig=plt.figure()
#ax=fig.add_subplot(1,1,1)
#ax.scatter(x_data,y_data)
#plt.ion()
#plt.show()
#
#for i in range(1001):
#    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
#    if i%50==0:
##        print('loss=',sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
#        try:
#            ax.lines.remove(lines[0])
#        except Exception:
#            pass
#        
#        pred_value=sess.run(pred,feed_dict={xs:x_data,ys:y_data})
#        lines=ax.plot(x_data,pred_value,'r-',lw=5)
#        plt.pause(0.1)


#example_seven:tensorboad

#def hidden_layer(inputs,in_size,out_size,activation_function=None):
#    with tf.name_scope('layer'):
#        with tf.name_scope('weights'):
#            Weights=tf.Variable(tf.random_normal([in_size,out_size]))
#        with tf.name_scope('biase'):
#            biase=tf.Variable(tf.zeros([1,out_size])+0.5)
#            Wx_plus_b=tf.matmul(inputs,Weights)+biase
#            if activation_function is None:
#                outputs=Wx_plus_b
#            else:
#                outputs=activation_function(Wx_plus_b)
#    return outputs
##
#x_data=np.linspace(-1,1,300)[:,np.newaxis]
#noise=np.random.normal(0,0.05,x_data.shape)
#y_data=np.square(x_data)-0.5+noise
#
#with tf.name_scope('inputs'):
#    xs=tf.placeholder(tf.float32,[None,1],name='x_input')
#    ys=tf.placeholder(tf.float32,[None,1],name='y_input')
#
#l1=hidden_layer(xs,1,10,activation_function=tf.nn.relu)
#pred=hidden_layer(l1,10,1,activation_function=None)
#
#with tf.name_scope('loss'):
#    loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-pred),
#                    reduction_indices=[1]))
#with tf.name_scope('train'):
#    train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)
#
#init=tf.global_variables_initializer()
#sess=tf.Session()
#writer=tf.summary.FileWriter('logs/',sess.graph)
#sess.run(init)

#example_eight:classification

#def hidden_layer(inputs,in_size,out_size,activation_function=None):
#    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
#    biase=tf.Variable(tf.zeros([1,out_size])+0.1)
#    Wx_plus_b=tf.matmul(inputs,Weights)+biase
#    if activation_function is None:
#        outputs=Wx_plus_b
#    else:
#        outputs=activation_function(Wx_plus_b)
#    return outputs
#
#def compute_accuracy(v_xs,v_ys):
#    global pred
#    y_pre=sess.run(pred,feed_dict={xs:v_xs})
#    correct_pred=tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
#    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
#    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
#    return result    
#
#xs=tf.placeholder(tf.float32,[None,784]) #28x28
#ys=tf.placeholder(tf.float32,[None,10])
#
#pred=hidden_layer(xs,784,10,activation_function=tf.nn.softmax)
#
#cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pred),
#                                            reduction_indices=[1]))
#train_step=tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)
#
#sess=tf.Session()
#sess.run(tf.global_variables_initializer())
#
#for i in range(1000):
#    batch_xs,batch_ys=mnist.train.next_batch(100)
#    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
#    if i%50==0:
#        print(compute_accuracy(
#                               mnist.test.images,mnist.test.labels))

#My_code_classification
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biase=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biase
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global pred
    y_pred=sess.run(pred,feed_dict={xs:v_xs})
    correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result

xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])

pred=add_layer(xs,784,10,activation_function=tf.nn.softmax)
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pred),
                                            reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.8).minimize(cross_entropy)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print('accuracy=',compute_accuracy(
                               mnist.test.images,mnist.test.labels))
#example_nine:overfitting


#example_ten:convolutional_nerual_network
def add_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
    biase=tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b=tf.matmul(inputs,Weights)+biase
    if activation_function is None:
        outputs=Wx_plus_b
    else:
        outputs=activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs,v_ys):
    global pred
    y_pred=sess.run(pred,feed_dict={xs:v_xs})
    correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(v_ys,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    result=sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys})
    return result
    
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)

def biase_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    #stride[1,x_movement,y_movement,1]
    #Must have stride[0]=stride[3]=1
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')    

xs=tf.placeholder(tf.float32,[None,784])
ys=tf.placeholder(tf.float32,[None,10])

pred=add_layer(xs,784,10,activation_function=tf.nn.softmax)
cross_entropy=tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pred),
                                            reduction_indices=[1]))
train_step=tf.train.GradientDescentOptimizer(0.8).minimize(cross_entropy)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys})
    if i%50==0:
        print('accuracy=',compute_accuracy(
                               mnist.test.images,mnist.test.labels))
