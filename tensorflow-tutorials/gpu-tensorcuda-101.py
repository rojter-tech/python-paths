#%%
#from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
print("GPU Available: ", tf.test.is_gpu_available())


#%%
import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess)
#%%
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))


#%%