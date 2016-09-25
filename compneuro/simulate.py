import tensorflow as tf

from compneuro import lif


def fill_feed_dict(model, current, dt):
    return {current: [1.5]*model.size,
            dt: [model.dt]*model.size}


def main(_):
    model = lif.LIFNeuronModel()
    time = 50  # s
    n_steps = int(time / model.dt)

    with tf.Graph().as_default(), tf.Session() as sess:
        current = tf.placeholder(tf.float32, shape=[model.size], name='current')
        dt = tf.placeholder(tf.float32, shape=[model.size], name='dt')

        step_op = lif.step(model, current, dt)

        summary_op = tf.merge_all_summaries()
        init_op = tf.initialize_all_variables()
        summary_writer = tf.train.SummaryWriter('data', sess.graph)

        sess.run(init_op)
        for t in range(n_steps):
            feed_dict = fill_feed_dict(model, current, dt)
            sess.run(step_op, feed_dict=feed_dict)

            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, t)
            summary_writer.flush()


if __name__ == '__main__':
    tf.app.run()
