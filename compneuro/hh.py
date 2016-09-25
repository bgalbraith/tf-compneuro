"""
Hodgkin-Huxley Neuron model
"""
import tensorflow as tf


class HHNeuronModel(object):
    size = 1
    dt = 0.025  # ms
    v_rest = 0.0  # mV
    c_m = 1.0  # uF/cm2
    gbar_na = 120.0  # mS/cm2
    gbar_k = 36.0  # mS/cm2
    gbar_l = 0.3  # mS/cm2
    e_na = 115.0  # mV
    e_k = -12.0  # mV
    e_l = 10.613  # mV


def step(model, current, dt):
    with tf.name_scope('neurons'):
        v_rest = tf.constant([model.v_rest] * model.size,
                             name='v_rest')
        c_m = tf.constant([model.c_m] * model.size,
                          name='c_m')
        gbar_na = tf.constant([model.gbar_na] * model.size,
                          name='gbar_na')
        gbar_k = tf.constant([model.gbar_k] * model.size,
                              name='gbar_k')
        gbar_l = tf.constant([model.gbar_l] * model.size,
                              name='gbar_l')
        e_na = tf.constant([model.e_na] * model.size,
                      name='e_na')
        e_k = tf.constant([model.e_k] * model.size,
                              name='e_k')
        e_l = tf.constant([model.e_l] * model.size,
                              name='e_l')
        v_m = tf.Variable(v_rest, name='v_m')
        tf.scalar_summary(['v_m']*model.size, v_m)

    # Ion channel dynamics
        alpha_n = 0.01*(-v_m + 10.0)/(tf.exp((-v_m + 10.0)/10.0) - 1)
        beta_n = 0.125*tf.exp(-v_m/80)
        n_inf = alpha_n / (alpha_n + beta_n)

        alpha_m = 0.1*(-v_m + 25)/(tf.exp((-v_m + 25)/10) - 1)
        beta_m = 4*tf.exp(-v_m/18)
        m_inf = alpha_m /(alpha_m + beta_m)

        alpha_h = 0.07*tf.exp(-v_m/20)
        beta_h = 1/(tf.exp((-v_m + 30)/10) + 1)
        h_inf = alpha_h / (alpha_h + beta_h)

        m = tf.Variable([0.0529324852572], name='m')
        h = tf.Variable([0.596120753508], name='h')
        n = tf.Variable([0.317676914061], name='n')

        g_na = tf.mul(tf.mul(gbar_na, tf.pow(m, 3.0)), h, name='g_na')
        g_k = tf.mul(gbar_k, tf.pow(n, 4.0), name='g_k')
        g_l = gbar_l

        m_ = (alpha_m*(1 - m) - beta_m*m) * dt
        h_ = (alpha_h*(1 - h) - beta_h*h) * dt
        n_ = (alpha_n*(1 - n) - beta_n*n) * dt

        v_m_ = (current - g_na*(v_m - e_na) - g_k*(v_m - e_k) - g_l*(v_m - e_l)) / c_m * dt

        _step = tf.group(
            m.assign_add(m_),
            h.assign_add(h_),
            n.assign_add(n_),
            v_m.assign_add(v_m_)
        )

    return _step
