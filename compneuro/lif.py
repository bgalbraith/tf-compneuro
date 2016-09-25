"""
Leaky Integrate-and-Fire Neuron model
"""
import tensorflow as tf


class LIFNeuronModel(object):
    size = 1
    dt = 0.125  # ms
    v_rest = 0.0  # V
    v_spike = 1.0  # V
    r_m = 1.0  # kOhm
    c_m = 10.0  # uF
    tau_ref = 4.0  # ms


def step(model, current, dt):
    with tf.name_scope('neurons'):
        v_rest = tf.constant([model.v_rest] * model.size,
                             name='v_rest')
        r_m = tf.constant([model.r_m] * model.size,
                          name='r_m')
        c_m = tf.constant([model.c_m] * model.size,
                          name='c_m')
        tau_m = tf.mul(r_m, c_m, name='tau_m')
        tau_ref = tf.constant([model.tau_ref] * model.size,
                              name='tau_ref')
        v_spike = tf.constant([model.v_spike] * model.size,
                              name='v_spike')

        v_m = tf.Variable(tf.zeros(model.size), trainable=True, name='v_m')
        tf.scalar_summary(['v_m']*model.size, v_m)

        t_rest = tf.Variable(tf.zeros(model.size), trainable=True,
                             name='t_rest')

        v_m_ = (tf.neg(v_m) + current * r_m) / tau_m * dt

        # neurons = tf.dynamic_partition(t_rest, tf.greater(t_rest, 0.0), 2)
        # neurons[0]
        def resting_op():
            return tf.tuple((t_rest.assign_sub(dt),
                             v_m.assign(v_rest)))

        def spiking_op():
            return tf.tuple((t_rest.assign_add(tau_ref),
                             v_m.assign_add(v_spike)))

        def responding_op():
            return tf.tuple((t_rest,
                             v_m.assign_add(v_m_)))

        _step = tf.case((
            (tf.reshape(tf.greater(t_rest, 0), []), resting_op),
            (tf.reshape(tf.greater(v_m, v_spike), []), spiking_op)),
            responding_op)
    return _step
