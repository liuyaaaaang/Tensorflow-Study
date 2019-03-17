import tensorflow as tf
import numpy as np 

#example_one:Linear Regression

#x_Data=np.random.rand(100).astype(np.float32)
#y_Data=x_Data*0.1+0.3
#
#
#Weights=tf.Variable(tf.random_uniform([1],-1.0,1.0))
#biases=tf.Variable(tf.zeros([1]))
#
#y=Weights*x_Data+biases
#
#loss=tf.reduce_mean(tf.square(y-y_Data))
#optimizer=tf.train.GradientDescentOptimizer(0.5)
#train=optimizer.minimize(loss)
#
#init=tf.initialize_all_variables()
#
#sess=tf.Session()
#sess.run(init)
#
#for step in range(201):
#    sess.run(train)
#    if step%20==0:
#        print(step,sess.run(Weights),sess.run(biases))


#My_code_one

#x1=np.random.rand(100).astype(np.float32)
#x2=np.random.rand(100).astype(np.float32)
#y_true=(x1*2.0)+(x2*3.0)+5.0
#
#Weight1=tf.Variable(tf.random_uniform([1],-2.0,2.0))
#Weight2=tf.Variable(tf.random_uniform([1],-2.0,2.0))
#biases=tf.zeros([1])
#y_pred=Weight1*(x1)+Weight2*(x2)+biases
#
#loss=tf.reduce_mean(tf.square(y_true-y_pred))
#optimizer=tf.train.GradientDescentOptimizer(0.8)
#train=optimizer.minimize(loss)
#
#init=tf.initialize_all_variables()
#sess=tf.Session()
#sess.run(init)
#
#for step in range(1001):
#    sess.run(train)
#    if step%50==0:
#        print(step,sess.run(Weight1),sess.run(Weight2),sess.run(biases))



#example_two:matrix_matul

#matrix1=tf.constant([[2],
#                     [2]])
#matrix2=tf.constant([[3,3]])
#product=tf.matmul(matrix1,matrix2)
#
##sess=tf.Session()
##result=sess.run(product)
##print(result)
##sess.close()
#
#with tf.Session() as sess:
#    reslut2=sess.run(product)
#    print(reslut2)

#My_code_two

#mat1=tf.constant([[2,3],
#                  [4,5]])
#mat2=tf.constant([[2,2],
#                  [2,2]])
#product=tf.matmul(mat1,mat2)
#
##sess=tf.Session()
##result=sess.run(product)
##print(result)
##sess.close()
#
#with tf.Session() as sess:
#    result=sess.run(product)
#    print(result)




#example_three:usage of variable

#state=tf.Variable(0,name='counter')
##print(state.name)
#one=tf.constant(1)
#
#new_value=tf.add(state,one)
#update=tf.assign(state,new_value)
#
#init=tf.global_variables_initializer()
#sess=tf.Session()
#sess.run(init)
#
#for step in range(3):
#    sess.run(update)
#    print(sess.run(state))

#My_code_three

#number1=tf.Variable(3,name='counter')
#number2=tf.constant(2)
#
#new_number=tf.multiply(number1,number2)
#update=tf.assign(number1,new_number)
#
#init=tf.global_variables_initializer()
#sess=tf.Session()
#sess.run(init)
#
#for step in range(5):
#    sess.run(update)
#    print(sess.run(number1))

#example_four:placeholder

#input1=tf.placeholder(tf.float32)
#input2=tf.placeholder(tf.float32)
#
#output=tf.multiply(input1,input2)
#
#init=tf.global_variables_initializer()
#sess=tf.Session()
#sess.run(init)
#
#print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
 

#My_code_four

#input1=tf.placeholder(tf.float32)
#input2=tf.placeholder(tf.float32)
#
#
#output=tf.add(input1,input2)
#
#init=tf.global_variables_initializer()
#sess=tf.Session()
#sess.run(init)
#
#print(sess.run(output,feed_dict={input1:[7.0],input2:[9.0]}))   


#example_five:def_hidden_layer,use_neural_network


#def add_layer(inputs,in_size,out_size,activation_function=None):
#    Weights=tf.Variable(tf.random_normal([in_size,out_size]))
#    biases=tf.Variable(tf.zeros([1,out_size])+0.1)
#    Wx_plus_b=tf.matmul(inputs,Weights)+biases
#    if activation_function is None:
#        outputs=Wx_plus_b
#    else:
#        outputs=activation_function(Wx_plus_b)
#    return outputs
#
#x_data=np.linspace(-1,1,300)[:,np.newaxis]
#noise=np.random.normal(0,0.05,x_data.shape)
#y_data=np.square(x_data)-0.5+noise
#
#xs=tf.placeholder(tf.float32,[None,1])
#ys=tf.placeholder(tf.float32,[None,1])
#
#layer1=add_layer(xs,1,10,activation_function=tf.nn.relu)
#pred=add_layer(layer1,10,1,activation_function=None)
#
#loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-pred),
#                   reduction_indices=[1]))
#train_step=tf.train.GradientDescentOptimizer(0.05).minimize(loss)
#
#init=tf.global_variables_initializer()
#
#with tf.Session() as sess:
#    sess.run(init)
#    for i in range(1001):
#        sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
#        if i%50==0:
#            print('loss=',sess.run(loss,feed_dict={xs:x_data,ys:y_data}))

#def hidden_layer(inputs,in_size,out_size,activation_function=None):
#   Weights=tf.Variable(tf.random_normal([in_size,out_size]))
#   biase=tf.Variable(tf.zeros([1,out_size])+0.1)
#   Wx_plus_b=tf.matmul(inputs,Weights)+biase
#   if activation_function is None:
#       outputs=Wx_plus_b
#   else:
#       outputs=activation_function(Wx_plus_b)
#   return outputs
#   
#x_data=np.linspace(-1,1,300)[:,np.newaxis]
#noise=np.random.normal(0,0.5,x_data.shape)
#y_data=np.square(x_data)-0.5+noise
#
#xs=tf.placeholder(tf.float32,[None,1])
#ys=tf.placeholder(tf.float32,[None,1])
#
#layer1=hidden_layer(xs,1,10,activation_function=tf.nn.relu)
#pred=hidden_layer(layer1,10,1,activation_function=None)
#
#loss=tf.reduce_mean(tf.reduce_sum(tf.square(ys-pred),
#                                  reduction_indices=[1]))
#train_step=tf.train.GradientDescentOptimizer(0.05).minimize(loss)
#
#init=tf.global_variables_initializer()
#sess=tf.Session()
#sess.run(init)
#
#for i in range(1001):
#    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
#    if i%50==0:
#        print('loss=',sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
