import tensorflow as tf

import numpy as  np

n_nodes_hl1 = 512
n_nodes_hl2 = 32

n_classes = 1
batch_size = 32

x = tf.placeholder('float', [None, 4096])
y = tf.placeholder('float')
extra = tf.placeholder('float')

train_x = []
train_y = []
test_x = []
test_y = []
a_width = []
n_width = []

def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([4096, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    #data = tf.layers.dropout(data, 0.6)
    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)
    l1 = tf.layers.dropout(l1, 0.6)
    
    
    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2  = tf.layers.dropout(l2, 0.6)
    
    
    output = tf.matmul(l2,output_layer['weights']) + output_layer['biases']
    output = tf.nn.sigmoid(output)		
    

    return output , hidden_1_layer['weights'] , hidden_2_layer['weights'] , output_layer['weights']

def train_neural_network(x):
    prediction , weights_1 , weights_2, weights_3 = neural_network_model(x)
    cost = 0.5 * (tf.nn.l2_loss(weights_1) + tf.nn.l2_loss(weights_2) + tf.nn.l2_loss(weights_3)) + extra
    optimizer = tf.train.AdagradOptimizer(0.001).minimize(cost)
    
    hm_epochs = 1
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
	for epoch in range(hm_epochs):
            epoch_loss = 0
	    i=0
            while i < len(train_x):
		start = i
		end =i+batch_size
		batch_x =np.array(train_x[start:end])
		batch_y =np.array(train_y[start:end])
		
		anomaly_score = sess.run(prediction,feed_dict={x: batch_x})
                normal_score,W1= sess.run([prediction,weights_1],feed_dict={x: batch_y})
                temp_ano = [float(inter[0]) for inter in anomaly_score.tolist()]
                temp_norm = [float(inter[0]) for inter in normal_score.tolist()]
                
                L = max(0.0,(1.0 - max(temp_ano) + max(temp_norm)))
                add = 0.0
                for index in range(len(temp_ano) - 1):
                	add += (temp_ano[index] - temp_ano[index+1]) ** 2
                 
                final_cost = L + (add*1.0 + sum(temp_ano)) * 0.00008 
                o,_ = sess.run([optimizer,prediction], feed_dict={extra : final_cost,x: batch_x})
		i =i + batch_size
            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)
        i = 0
        count = 0
        while i < len(test_x):
        	ret = []
        	for vec in test_x[i:i+32]:
        		temp = float(sess.run(prediction,feed_dict={x: np.array(vec).reshape(1,-1)})[0][0])
        		ret.append(temp)
        	print sum(ret) * a_width[count]
        	i = i + 32
        	count += 1
        print "--------------------------------------------"
        i = 0
        count = 0
        while i < len(test_y):
        	ret = []
        	for vec in test_y[i:i+32]:
        		temp = float(sess.run(prediction,feed_dict={x: np.array(vec).reshape(1,-1)})[0][0])
        		ret.append(temp)
        	print sum(ret) * n_width[count]
        	i = i + 32
        	count += 1		

        

def train_NN(x1,y1,x2,y2,w_a,w_n):
	global train_x,train_y,test_x,test_y,a_width,n_width
	train_x = x1
	train_y = y1
	test_x = x2
	test_y = y2
	a_width = w_a
	n_width = w_n
	train_neural_network(x)
