import tensorflow as tf
#import iemocap_batch
#import os
from random import shuffle
import numpy as np

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.3
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 20000
MOVING_AVERAGE_DECAY = 0.99
INPUT_NODE = 750
OUTPUT_NODE = 5
LAYER_NODE = 256
MODEL_SAVE_PATH="IEMOCAP_model/"
MODEL_NAME="iemocap_model"
train_list = 'train_list'
test_list =  'test_list'
    
LABEL_PATH = '/home/lemn/experiment/code/feature/labels/'

IEMOCAP_FEATURE_PATH = '/home/lemn/experiment/code/feature/iemocap/'
UTTERANCE_FEATURE_PATH = '/home/lemn/experiment/code/feature/iemocap_utterance_feature/'
EMOTIONS = {'exc':0,'fru':1,'hap':2,'neu':3,'sad':4}

def make_emotion(emotion):
    ret = [0 for i in range(len(EMOTIONS))]
    ret[EMOTIONS[emotion]]=1
    return ret

def get_x_y(x,y):
    #print('x:'+x)
    #print('y:'+y)
    x = [[float(a) for a in line.split()] for line in x]
    y = [[int(a) for a in line.split()] for line in y]
    return[x,y]

def get_features_num(labels_filenames):
    n = 0
    for filename in labels_filenames:
        f = open(filename)
        n+=len(f.readlines())
        f.close()
    return(n)

def dnn(list_file):
    #building dataset.
    f = open(list_file)
    lines = f.readlines()
    shuffle(lines)
    f.close()
    names = [line.strip().split(' ')[0] for line in lines]
    emotions = [line.strip().split(' ')[1] for line in lines]
    filenames = [''.join([IEMOCAP_FEATURE_PATH,line.strip().split(' ')[0],'.fea']) for line in lines]
    #labels = [make_emotion(emotion) for emotion in emotions]
    
    frames_filenames = filenames
    labels_filenames = [''.join([LABEL_PATH,name,'.lbl']) for name in names]
    total_feature_nums = get_features_num(labels_filenames)
    print('batch times of a total dataset riding : %d' % (total_feature_nums/BATCH_SIZE))
    
    frame_dataSet = tf.contrib.data.TextLineDataset(frames_filenames)
    #frame_dataSet = frame_dataSet.map(_parse_frame_function)
    
    label_datSet = tf.contrib.data.TextLineDataset(labels_filenames)
    #label_datSet = label_datSet.map(_parse_label_function)
    
    dataset = tf.contrib.data.Dataset.zip((frame_dataSet,label_datSet))
    
    dataset = dataset.shuffle(buffer_size=10000)
    
    batched_dataset = dataset.batch(batch_size=BATCH_SIZE)
    iterator = batched_dataset.make_one_shot_iterator()
    next_element = iterator.get_next()

    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None,OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE,LAYER_NODE],stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1,shape=[LAYER_NODE]))

    weights2 = tf.Variable(tf.truncated_normal([LAYER_NODE,LAYER_NODE],stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1,shape=[LAYER_NODE]))

    weights3 = tf.Variable(tf.truncated_normal([LAYER_NODE,LAYER_NODE],stddev=0.1))
    biases3= tf.Variable(tf.constant(0.1,shape=[LAYER_NODE]))

    weights4 = tf.Variable(tf.truncated_normal([LAYER_NODE,OUTPUT_NODE],stddev=0.1))
    biases4 = tf.Variable(tf.constant(0.1,shape=[OUTPUT_NODE]))

    layer1 = tf.nn.relu(tf.matmul(x, weights1) + biases1)
    layer2 = tf.nn.relu(tf.matmul(layer1, weights2) + biases2)
    layer3 = tf.nn.relu(tf.matmul(layer2,weights3) + biases3)
    layer4 = tf.nn.relu(tf.matmul(layer3,weights4) + biases4)

    y = tf.nn.softmax(layer4)

    global_step = tf.Variable(0, trainable=False)


    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = tf.argmax(y_, 1))
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels = layer4,logits=y_)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + regularizer(weights1)+regularizer(weights2)+regularizer(weights3)+regularizer(weights4)
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        total_feature_nums / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
    #with tf.control_dependencies([train_step]):
        train_op = tf.no_op(name='train')

    correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

    #saver = tf.train.Saver()
    with tf.Session() as sess:
        ##initializing
        tf.global_variables_initializer().run()
        #train
        for i in range(TRAINING_STEPS):
            #print('i:%s' %i) 
            try:
                [xs, ys] = sess.run(next_element)
            except tf.errors.OutOfRangeError:
                iterator = batched_dataset.make_one_shot_iterator()
                next_element = iterator.get_next()
                [xs, ys] = sess.run(next_element)
            [xs,ys] = get_x_y(xs,ys)
            _,loss_value, step,l_rate,validate_acc= sess.run([train_op,loss, global_step,learning_rate,accuracy], feed_dict={x: xs, y_: ys})
            if (i+1) % 10 == 0:                                                               
                #validate_acc = sess.run(accuracy,feed_dict = {x:xs,y_:ys})                                                                                                                                                                                                                                                                                                                                                                                                            
                print("After %d training step(s), learning_rate=%g,loss on training batch is %g ,accuracy=%g." 
                      % (step,l_rate, loss_value,validate_acc))                                                                                                                                                                                                                                      
                #saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        
        ##test

        for ff in ['evaluate_list','test_list']:
            f=open(ff)
            print(ff)
            test_files = f.readlines()
            f.close()
            for test_file in test_files:
                [name,emotion] = test_file.strip().split(' ')
#                print(name)
#                print(emotion)
                emotion = make_emotion(emotion)
                f1=open(''.join([UTTERANCE_FEATURE_PATH,name]),'w')
                f2 = open(''.join([IEMOCAP_FEATURE_PATH,name,'.fea']))
                xt = []
                lines = f2.readlines()
                f2.close()
                ys = [emotion for i in range(len(lines))]
                xt = [[float(x) for x in line.strip().split(' ')] for line in lines]
                y_eval = sess.run(y,feed_dict={x:xt,y_:ys})
                for a in y_eval:
                    f1.write(' '.join([str(b) for b in a])+'\n')
                f1.close()


def main(argv=None):
    list_file = 'train_list'
    print('BATCH_SIZE=%d' % BATCH_SIZE)
    #print('LEARNING_RATE_BASE:%f'%LEARNING_RATE_BASE)
    print('TRAINING_STEPS=%d' % TRAINING_STEPS)
    print('list_file=%s' % list_file)
    dnn(list_file)

if __name__ == '__main__':
    tf.app.run()

