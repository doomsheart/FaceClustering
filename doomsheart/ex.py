import tensorflow as tf

sess_ = tf.Session()
input = [[-1, -1, -1],
         [2, 2, 2],
         [3, 3, 3],
         [4, 4, 4],
         [5, 5, 5],
         [6, 6, 6]
]

a = tf.Variable(tf.slice(input, [0,0], [2, -1]))
sess_ = tf.Session()
init_op_ = tf.global_variables_initializer()
sess_.run(init_op_)
result = (sess_.run(a))
expanded_vectors_ = tf.expand_dims(input, 1)
re = sess_.run(expanded_vectors_)
print((re))
t = [2]
print(t)
print(sess_.run(tf.expand_dims(t, 0)))
# for a in expanded_vectors_:
#     print(a)
# tf.slice(input, [1, 0, 0], [1, 2, 3]) ==> [[[3, 3, 3],
#                                             [4, 4, 4]]]
# tf.slice(input, [1, 0, 0], [2, 1, 3]) ==> [[[3, 3, 3]],
                                           # [[5, 5, 5]]]