import tensorflow as tf
import numpy as np

# CHECK : Constants
omega = 1.

#FILE_PATH ='temporal/'
UTTERANCE_FEATURE_PATH = '/home/lemn/experiment/code/feature/iemocap_utterance_feature/'
TRAINING_STEPS = 100000
batch_size = 100
input_len = 20
hidden_num = 120
output_len = 5

EMOTIONS = {'exc':0,'fru':1,'hap':2,'neu':3,'sad':4}
def make_emotion(emotion):
    ret = [0 for i in range(len(EMOTIONS))]
    ret[EMOTIONS[emotion]]=1
    return ret
    
def get_utterance_feature(file_list,ratio = 0.2):
    f = open(file_list)
    readlines = f.readlines()
    f.close()
    xMat =[]
    yMat = []
    for line0 in readlines:
        [name,emotion] = line0.strip().split(' ')
        filename = ''.join([UTTERANCE_FEATURE_PATH,name])
        y = make_emotion(emotion)
        yMat.append(y)
        f = open(filename)
        segment_probs=[]
        lines= f.readlines()
        f.close()
        for line in lines:
            a = [float(x) for x in line.strip().split(' ')]
            segment_probs.append(a)
        segment_probs = np.array(segment_probs)
        (n,m) = np.shape(segment_probs)
        ret = []
        for i in range(m):
            a = segment_probs[:,i]
            #print(a)
            ret.append(np.max(a))
            ret.append(np.min(a))
            ret.append(np.mean(a))
            ret.append(np.sum(a>ratio)/n)
        xMat.append(ret)
    return [xMat,yMat]



class ELM(object):
  def __init__(self, sess, batch_size, input_len, hidden_num, output_len):
    '''
    Args:
      sess : TensorFlow session.
      batch_size : The batch size (N)
      input_len : The length of input. (L)
      hidden_num : The number of hidden node. (K)
      output_len : The length of output. (O)
    '''
    
    self._sess = sess 
    self._batch_size = batch_size
    self._input_len = input_len
    self._hidden_num = hidden_num
    self._output_len = output_len 

    # for train
    self._x0 = tf.placeholder(tf.float32, [self._batch_size, self._input_len])
    self._t0 = tf.placeholder(tf.float32, [self._batch_size, self._output_len])

    # for test
    self._x1 = tf.placeholder(tf.float32, [None, self._input_len])
    self._t1 = tf.placeholder(tf.float32, [None, self._output_len])

    self._W = tf.Variable(
      tf.random_normal([self._input_len, self._hidden_num]),
      trainable=False, dtype=tf.float32)
    self._b = tf.Variable(
      tf.random_normal([self._hidden_num]),
      trainable=False, dtype=tf.float32)
    self._beta = tf.Variable(
      tf.zeros([self._hidden_num, self._output_len]),
      trainable=False, dtype=tf.float32)
    self._var_list = [self._W, self._b, self._beta]

    self.H0 = tf.sigmoid(tf.matmul(self._x0, self._W) + self._b) # N x L
    self.H0_T = tf.transpose(self.H0)

    self.H1 = tf.sigmoid(tf.matmul(self._x1, self._W) + self._b) # N x L
    self.H1_T = tf.transpose(self.H1)

    # beta analytic solution : self._beta_s (K x O)
    if self._input_len < self._hidden_num: # L < K
      identity = tf.constant(np.identity(self._hidden_num), dtype=tf.float32)
      self._beta_s = tf.matmul(tf.matmul(tf.matrix_inverse(
        tf.matmul(self.H0_T, self.H0) + identity/omega), 
        self.H0_T), self._t0)
      # _beta_s = (H_T*H + I/om)^(-1)*H_T*T
    else:
      identity = tf.constant(np.identity(self._batch_size), dtype=tf.float32)
      self._beta_s = tf.matmul(tf.matmul(self.H0_T, tf.matrix_inverse(
        tf.matmul(self.H0, self.H0_T)+identity/omega)), self._t0)
      # _beta_s = H_T*(H*H_T + I/om)^(-1)*T
    
    self._assign_beta = self._beta.assign(self._beta_s)
    self._fx0 = tf.matmul(self.H0, self._beta)
    self._fx1 = tf.matmul(self.H1, self._beta)

    self._cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self._fx0, labels=self._t0))
    #self._train_step = tf.train.AdamOptimizer(0.001).minimize(self._cost)
    self._init = False
    self._train = False

    # for the mnist test
    self._correct_prediction = tf.equal(tf.argmax(self._fx1,1), tf.argmax(self._t1,1))
    self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))

  def train(self, x, t):
    '''
    Args :
      x : input array (N x L)
      t : label array (N x O)
    '''

    if not self._init : self.init()
    for i in range(TRAINING_STEPS):
        batch_start = i*self._batch_size
#        if batch_start >= len(x):
#            batch_start = batch_start %len(x)
        batch_end = batch_start + self._batch_size
        _x = []
        _t = []
        for j in range(batch_start,batch_end):
          k = j
          if j>=len(x):
            k = j%len(x)
          _x.append(x[k])
          _t.append(t[k])
        _,loss = self._sess.run([self._assign_beta,self._cost], {self._x0:_x, self._t0:_t})
        print('After %d training steps,loss comes %f' %(i,loss))

    self._train = True

  def init(self):
    self._sess.run(tf.initialize_variables(self._var_list))
    self._init = True

  def test(self, x, t=None):
    if not self._train : exit("Not feed-forward trained")
    if t is not None :
        _accuracy,_y = self._sess.run([self._accuracy,self._t1],{self._x1:x, self._t1:t})
        print("Accuracy: %f" % (_accuracy))
        print(_y)
    else :
      return self._sess.run(self._fx1, {self._x1:x})


if __name__ == '__main__':
    with tf.Session() as sess:
        elm = ELM(sess,batch_size,input_len,hidden_num,output_len)
        [x,y] = get_utterance_feature('evaluate_list')
        #print(x)
        #print(y)
        elm.train(x,y)
        [x1,y1] = get_utterance_feature('test_list')
        elm.test(x1,y1)
        