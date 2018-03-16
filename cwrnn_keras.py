from keras.layers import SimpleRNN
import numpy as np
import keras.backend as K


class ClockworkRNNCell(layer):
	"""Cell class for ClockworkRNN.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
		period_spec: it is the special constant of clockwork, 
			which present the clock period of each netron group;
			it is used to calculatr the utm_mask and t % Ti;
    """
    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 period_spec = [1],
                 **kwargs):
        super(ClockworkRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.state_size = self.units
        self.period_spec = period_spec

        self.num_group = len(self.period_spec)
        self.group_size = self.units // self.num_group

    def build(self, input_shape):

        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        
        # According to paper: http://arxiv.org/abs/1402.3511
        # the Clockwork network connection is upper Triangle matrix by group;
        # that means all of the units of clockwork cell will be divide into several group,
        # each group size is the same, and the connection rule is:
        # 1. If group j period > group i period, then 
        # 	all of units of group j should connect with group i;
        # 2. regarding group internal connection, all of the units should be fully connected;
        # so, below mask is the implement of above connection feature. 
        # the detail please refer the figure 1 in paper.(http://arxiv.org/abs/1402.3511) 
        mask = np.zeros((self.units, self.units), K.floatx())

        # calculate the upper Triangle matrix mask for clockwork; and
        # generate period for each unit(netron) 
        for group_index, group_period in enumerate(self.period_spec):
            mask[group_index * self.group_size:(group_index + 1) * self.group_size, group_index*self.group_size:] = 1

        # utm_mask: Upper Triangle Matrix Mask, it is the special mask of clockwork,
        # it presents the netron connection of Clockwork network;
        # Note, this mask is based on simpleRNN, because simpleRNN is fully interconnected,
        # but Clockwork is Upper Triangle base on simpleRNN fully interconnected matrix;
        self.utm_mask = K.variable(mask, name='clockwork_mask')

        self.built = True

    def call(self, inputs, states, training=None):
    	# The states[0] stores the previous output date, 
    	# and the shape is (batch_size, self.units);
        prev_output = states[0]
        # the states[1] stores the current timestep;
        timestep = states[1]
        
        # base on timestep, got the valid column 
        valid_column = states[2][timestep]

        # Compute (W_I*x_t + b_I)
        # Please refer equation (1) of the paper 
        # The inputs shape is (batch_size, input_dim);
        # According to the paper, only need to calc the valid group,
        # that means only need calc the valid_column in front of weight matrix;
        # the invalid group column(upper group column in matrix) won't attend compute; 
        h = K.dot(inputs, self.kernel[:,:valid_column])
        if self.bias is not None:
            h = K.bias_add(h, self.bias[:valid_column])

    	# Compute (W_H*y_{t-1} + b_H)
    	# please refer equation (1) of the paper
    	# Note: the y_{t-1} means the previous output (state)
    	# Note: the self.recurrent_kernel * self.utm_mask maybe move to build function;
    	# similar to above compute, we only need compute the valid_column part.
        output = h + K.dot(prev_output, (self.recurrent_kernel * self.utm_mask)[:,:valid_column])
        if self.activation is not None:
            output = self.activation(output)

        # concatenate the valid group output and invalid group pre_output
        output = K.concatenate([output, prev_output[:,valid_column:]], axis = 1)

        # return output, [states, timestep]
        return output, [output, timestep + 1]

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'period_spec': self.period_spec}
        base_config = super(ClockworkRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class ClockworkRNN(RNN):
    """Clock-work RNN which work based on clock group.

    # Arguments
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use
            (see [activations](../activations.md)).
            Default: hyperbolic tangent (`tanh`).
            If you pass `None`, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix,
            used for the linear transformation of the inputs
            (see [initializers](../initializers.md)).
        recurrent_initializer: Initializer for the `recurrent_kernel`
            weights matrix,
            used for the linear transformation of the recurrent state
            (see [initializers](../initializers.md)).
        bias_initializer: Initializer for the bias vector
            (see [initializers](../initializers.md)).
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix
            (see [regularizer](../regularizers.md)).
        bias_regularizer: Regularizer function applied to the bias vector
            (see [regularizer](../regularizers.md)).
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation").
            (see [regularizer](../regularizers.md)).
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix
            (see [constraints](../constraints.md)).
        recurrent_constraint: Constraint function applied to
            the `recurrent_kernel` weights matrix
            (see [constraints](../constraints.md)).
        bias_constraint: Constraint function applied to the bias vector
            (see [constraints](../constraints.md)).
        return_sequences: Boolean. Whether to return the last output.
            in the output sequence, or the full sequence.
        return_state: Boolean. Whether to return the last state
            in addition to the output.
        go_backwards: Boolean (default False).
            If True, process the input sequence backwards and return the
            reversed sequence.
        stateful: Boolean (default False). If True, the last state
            for each sample at index i in a batch will be used as initial
            state for the sample of index i in the following batch.
        unroll: Boolean (default False).
            If True, the network will be unrolled,
            else a symbolic loop will be used.
            Unrolling can speed-up a RNN,
            although it tends to be more memory-intensive.
            Unrolling is only suitable for short sequences.
        period_spec: the period value list of each (CWRNN) Clcok group. 
    """

    @interfaces.legacy_recurrent_support
    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 period_spec = [1],
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            warnings.warn('The `implementation` argument '
                          'in `ClockworkRNN` has been deprecated. '
                          'Please remove it from your layer call.')

        cell = ClockworkRNNCell(units,
                             activation=activation,
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             recurrent_initializer=recurrent_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=kernel_regularizer,
                             recurrent_regularizer=recurrent_regularizer,
                             bias_regularizer=bias_regularizer,
                             kernel_constraint=kernel_constraint,
                             recurrent_constraint=recurrent_constraint,
                             bias_constraint=bias_constraint,
                             period_spec = period_spec)
        super(ClockworkRNN, self).__init__(cell,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        unroll=unroll,
                                        **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None):

    	initial_states = _get_initial_states(inputs)
    	constants = _get_constants(inputs)

        return super(ClockworkRNN, self).call(inputs,
                                           mask=mask,
                                           training=training,
                                           initial_state=initial_states,
                                           constants=constants)

    def _get_initial_states(self, inputs):
    	"""calculate the initial_states of CWRNN
		
		The states of standard CWRNN should be:
		1. standard_rnn_state, which shape is (samples, state_size)
		2. timestep, which is a int variable means step count of currently call. 
		
		this function return cernn_states[standard_rnn_state, timestep]

    	"""

        # build an all-zero tensor of shape (samples, output_dim)
        initial_state = K.zeros_like(inputs)  # (samples, timesteps, input_dim)
        initial_state = K.sum(initial_state, axis=(1, 2))  # (samples,)
        initial_state = K.expand_dims(initial_state)  # (samples, 1)

        standard_rnn_state = K.tile(initial_state, [1, self.cell.state_size])
        timestep = K.variable(0, dtype=int16)

        cwrnn_states = [standard_rnn_state, timestep]

        return cwrnn_states

    def _get_constants(self, inputs):

    	# number of total steps
    	input_shape = K.int_shape(inputs)
    	timesteps = input_shape[1]

        # timestep % period(ts % Ti) calculation needn't run each time in call(RNN Step),
        # we can calculate it in build;
        # How to use valid_column_group ?
        # self.valid_column_group[ts] store which max column of weight matrix will attend computing.
        valid_column_group = []
        for ts in range(timesteps):
        	for group_index, group_period in enumerate(self.cell.period_spec):
        		if ts % group_period == 0:
        			valid_column_group[ts] = (group_index + 1) * self.group_size

		valid_column_group_cons = K.constant(valid_column_group, dtype=int16)

		constants = [valid_column_group_cons]

    	return constants

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'period_spec': self.cell.period_spec}
        base_config = super(SimpleRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)
