# LSTM for training the drummer.
#
# The training procedure is based on Andrej Karpathy's min-char-rnn script from
# https://gist.github.com/karpathy/d4dee566867f8291f086
#
# Training repeats for the specified number of steps. The model is saved to disk
# periodically in the checkpoints directory. Press Ctrl+C to stop training.
#
# To see the TensorBoard statistics while training, run:
#   tensorboard --logdir=logs --reload_interval=30
#
# NOTE: You should manually remove the logs directory before every training run,
# or TensorBoard will get confused.

import os
import sys
import numpy as np
import tensorflow as tf
import pickle
import struct

checkpoint_dir = "checkpoints"
summaries_dir = "logs"

# NOTE: the model described in the blog post uses 200 unroll_steps. However, the
# training data in the repo is too small for that, so we use 21 steps instead.

hidden_size = 200          # number of neurons in hidden layer
unroll_steps = 21          # number of steps to unroll the RNN for
training_steps = 100000    # make this a big number!

################################################################################

def usage():
    script_name = sys.argv[0]
    print("Usage:")
    print("  %s train                      train a new model" % script_name)
    print("  %s train <checkpoint_file>    resume training" % script_name)
    print("  %s sample <checkpoint_file>   sample from saved model" % script_name)
    print("  %s export <checkpoint_file>   save the weights" % script_name)
    print("  %s random                     drum like a monkey" % script_name)
    sys.exit(1)

mode = None
if len(sys.argv) >= 2:
    if sys.argv[1] == "train":
        mode = "train"
        if len(sys.argv) >= 3:
            model_file = sys.argv[2]
            print("Resuming training from model %s" % model_file)
        else:
            model_file = None
            print("Training new model")
        print("Saving model to %s" % checkpoint_dir)       
    elif sys.argv[1] == "sample":
        if len(sys.argv) >= 3:
            mode = "sample"
            model_file = sys.argv[2]
            print("Sampling from model %s" % model_file)
    elif sys.argv[1] == "export":
        mode = "export"
        model_file = sys.argv[2]
        print("Exporting from model %s" % model_file)
    elif sys.argv[1] == "random":
        mode = "random"

if mode is None:
    usage()

################################################################################

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

class RNN:
    def __init__(self, note_vector_size, tick_vector_size, hidden_size, unroll_steps):
        """Creates a new RNN object.

        Parameters
        ----------
        note_vector_size: int
            number of elements in each (one-hot encoded) input note
        tick_vector_size: int
            number of elements in each (one-hot encoded) input duration
        hidden_size: int
            size of hidden layer of neurons
        unroll_steps: int
            number of steps to unroll the RNN for            
        """
        self.note_vector_size = note_vector_size
        self.tick_vector_size = tick_vector_size
        self.vector_size = self.note_vector_size + self.tick_vector_size
        self.hidden_size = hidden_size
        self.unroll_steps = unroll_steps
        self.build_graph()

    def build_graph(self):
        print("Creating graph...")

        with tf.name_scope("hyperparameters"):
            self.learning_rate = tf.placeholder(tf.float32, name="learning-rate")

        # The dimensions of the input tensor x and the target tensor y are 
        # (unroll_steps, vector_size) but we leave the first dimension as None,
        # so that in sample() we can pass in a single value at a time.
        with tf.name_scope("inputs"):
            self.x = tf.placeholder(tf.float32, [None, self.vector_size], name="x-input")

            # Because we train to predict the next element, y contains almost the
            # same elements as x but shifted one step in time: y[t] = x[t-1].
            self.y = tf.placeholder(tf.float32, [None, self.vector_size], name="y-input")

            # Input for the initial memory state of the LSTM. This is the last memory 
            # state of the previous time rnn.train() was called.
            self.h = tf.placeholder(tf.float32, [1, self.hidden_size], name="h-prev")
            self.c = tf.placeholder(tf.float32, [1, self.hidden_size], name="c-prev")

        # Model parameters for a single LSTM layer. This is what the network will learn.
        # The "layer" really consists of a single LSTM cell but since we unroll the network
        # in time, we will have unroll_steps cells in each layer. These all share the same 
        # weights but have their own internal state vectors.
        with tf.name_scope("lstm-cell"):
            # This matrix combines the weights for x, h, and the bias.
            # Notice that normally we'd initialize the bias values with 0 but
            # here they get the same initializations as the rest of the weights.
            self.Wx = weight_variable([self.vector_size + self.hidden_size + 1, self.hidden_size*4])

        # Parameters of hidden (h) to output (y). This is also what the network will learn.
        with tf.name_scope("lstm-to-output"):
            # This matrix combines the weights and the bias.
            self.Wy = weight_variable([self.hidden_size + 1, self.vector_size])

        # The logic for the LSTM cell. We unroll the network into unroll_steps steps, 
        # each with its own cell. The cell stores hidden state ("h") but also cell state
        # ("c"). Since we "unroll" the LSTM, we need to keep track of unroll_steps of 
        # these h and c state vectors (each vector contains hidden_size elements). 
        hs = [self.h]
        cs = [self.c]
        ys_note = []
        ys_tick = []
        for t in range(self.unroll_steps):
            # Create an input vector of size [x + h + 1]. The 1 is for the bias.
            h_flat = tf.reshape(hs[t], [self.hidden_size])
            combined = tf.concat([self.x[t], h_flat, tf.ones(1)], axis=0)

            # Turn the vector into a matrix with shape (1, size) so we can matmul() 
            # it. After the computation, hs[t] will have the shape (1, hidden_size). 
            # We keep it in that shape because we need to matmul() to compute the 
            # output ys[t] too.
            reshaped = tf.reshape(combined, [1, self.vector_size + self.hidden_size + 1])
           
            # Compute the new hidden state and cell state, which depends on the "current" 
            # input x[t] and the previous hidden state, h[t - 1] and c[t - 1]. 
            cell = tf.matmul(reshaped, self.Wx)
            cell = tf.reshape(cell, [4, self.hidden_size])
            cell_c = tf.sigmoid(cell[0]) * cs[t] + tf.sigmoid(cell[1]) * tf.tanh(cell[3])
            cell_h = tf.sigmoid(cell[2]) * tf.tanh(cell_c)

            # Slightly confusing: we write hs[t] and cs[t] here and not 't - 1' because 
            # hs[0] and cs[0] are the "old" h and c coming in from the chunk that was 
            # trained previously. And so hs[t] is really h[t - 1], likewise for c/cs.
            # Formulas used from https://en.wikipedia.org/wiki/Long_short-term_memory

            # Store the new hidden and cell state, which we need to compute the
            # output for this time step ys[t].
            hs.append(cell_h)
            cs.append(cell_c)

            # Add 1 for the bias.
            combined = tf.concat([cell_h, tf.ones((1, 1))], axis=1)
            y_pred = tf.matmul(combined, self.Wy)

            # Each ys[t] is the predicted element for step t in the RNN, a matrix of shape 
            # (1, vector_size). We reshape it so that ys will be (unroll_steps, vector_size) 
            # and so we can more easily compare it to self.y, which also has that shape.
            y_pred = tf.reshape(y_pred, [self.vector_size])

            # The output of the network is the unnormalized log probabilities for what the 
            # next element in the sequence is predicted to be. We convert this to actual 
            # probabilities (softmax). We compute the softmax separately over the note and 
            # tick parts of the output vector, so that we get two probability distributions.
            # We don't recombine these parts into a new vector because it's more convenient
            # to have them separate.

            # Predict the next note.            
            y_pred_note = tf.nn.softmax(y_pred[:self.note_vector_size])
            ys_note.append(y_pred_note)

            # Predict the next duration.
            y_pred_tick = tf.nn.softmax(y_pred[self.note_vector_size:])
            ys_tick.append(y_pred_tick)

        # We don't need to remember any of the intermediate steps, only the first 
        # one (for sampling) and the last one (for training the next batch).
        self.y_pred_note = ys_note[0]
        self.y_pred_tick = ys_tick[0]
        self.first_h = hs[1]           # since hs[0] is the old one
        self.last_h = hs[-1]
        self.first_c = cs[1]           # since cs[0] is the old one
        self.last_c = cs[-1]

        # The following operations are only used during training, not for inference.

        # Need to split up the expected output into note and duration. This isn't
        # strictly needed for the loss calculation but it is for accuracy, since
        # that needs to do an argmax() on each of these separate parts.
        y_note = self.y[:, :self.note_vector_size]
        y_tick = self.y[:, self.note_vector_size:]

        with tf.name_scope("loss-function"):
            # Softmax, so use cross entropy loss.
            # Because we have two probability distributions (one for notes, one
            # for ticks), the loss is the sum of their individual losses.
            self.loss = (tf.reduce_mean(-tf.reduce_sum(y_note * tf.log(ys_note), reduction_indices=[1]))
                      +  tf.reduce_mean(-tf.reduce_sum(y_tick * tf.log(ys_tick), reduction_indices=[1])))

        with tf.name_scope("train"):
            optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

            # Apply gradient clipping.
            grads_and_vars = optimizer.compute_gradients(self.loss)
            clipped = [(tf.clip_by_value(grad, -5.0, 5.0), var) for grad, var in grads_and_vars]
            self.train_op = optimizer.apply_gradients(clipped)

        # The accuracy op computes the % correct predictions. This is only the accuracy
        # across a single unrolled chunk of data, not across the entire dataset!
        with tf.name_scope("accuracy"):
            # Combine notes and ticks into a new tensor that looks like this:
            # [[note1,tick1], [note2,tick2], ..., [note_n, tick_n]]
            y_stacked = tf.stack([tf.argmax(y_note, 1), tf.argmax(y_tick, 1)], axis=1)
            ys_stacked = tf.stack([tf.argmax(ys_note, 1), tf.argmax(ys_tick, 1)], axis=1)
            
            # Then compare the predictions with the truth. We only count success
            # if both the note and the tick are correct.
            correct_prediction = tf.to_float(tf.reduce_all(tf.equal(y_stacked, ys_stacked), axis=1))
            self.accuracy = tf.reduce_mean(correct_prediction)

        self.init = tf.global_variables_initializer()

    def prepare_for_training(self, sess):
        """Call this before training starts."""
        sess.run(self.init)

        # Compute the loss at iteration 0. This is the "ideal" loss when the weights 
        # are all 0. Because we initialize the weights with small random numbers, the 
        # true initial loss will be slightly different.
        initial_loss = -np.log(1.0/self.note_vector_size) + -np.log(1.0/self.tick_vector_size)

        print("Expected initial loss:", initial_loss)

    def train(self, sess, x, y, h, c, learning_rate):
        """Runs the RNN unroll_steps steps forward and backward. 
        
        Parameters
        ----------
        sess: tf.Session
            the TensorFlow session
        x: ndarray of shape (unroll_steps, vector_size)
            the one-hot encoded inputs for the entire chunk
        y: ndarray of shape (unroll_steps, vector_size)
            the one-hot encoded targets for the entire chunk
        h, c: ndarray of shape (hidden_size, 1)
            the starting memory state            
        learning_rate: float
            the learning rate of the optimizer

        Returns
        -------
        The loss after training, the new memory state
        """
        feed = {self.x: x, self.y: y, self.h: h, self.c: c, self.learning_rate: learning_rate}
        ops = [self.train_op, self.loss, self.last_h, self.last_c]
        _, loss_value, h, c = sess.run(ops, feed_dict=feed)
        return loss_value, h, c

    def sample(self, sess, h, c, seed_ix_note, seed_ix_tick, n):
        """Samples a sequence from the model.

        This performs the forward pass n number of times and adds every predicted output
        to an array. We use this to make the network generate output based on what it has
        learned so far.

        Parameters
        ----------
        sess: tf.Session
            the TensorFlow session        
        h, c: ndarray of shape (hidden_size, 1)
            the starting memory state
        seed_ix_note/tick: int
            seed indices for the first time step
        n: int 
            the number of elements to generate

        Returns
        -------
        A list of (note, tick) indices.
        """
        x = np.zeros((1, self.vector_size))
        ixes = []
        for t in range(n):
            # One-hot encode the input values. Recall that x actually contains two
            # separate vectors that we must both one-hot encode.
            x[0, seed_ix_note] = 1
            x[0, self.note_vector_size + seed_ix_tick] = 1

            # Do the forward pass. Note that we don't need the entire "unrolled" 
            # RNN now. We only feed in a single example and we compute a single
            # output. (Can't do more than one at a time because the next input
            # depends on the current output.)
            feed = {self.x: x, self.h: h, self.c: c}
            ops = [self.y_pred_note, self.y_pred_tick, self.first_h, self.first_c]
            predicted_note, predicted_tick, h, c = sess.run(ops, feed_dict=feed)
            
            # Randomly sample from the output probability distributions.
            ix_note = np.random.choice(range(self.note_vector_size), p=predicted_note.ravel())
            ix_tick = np.random.choice(range(self.tick_vector_size), p=predicted_tick.ravel())
            ixes.append((ix_note, ix_tick))

            # Use the output as the next input.
            x[0, seed_ix_note] = 0
            x[0, self.note_vector_size + seed_ix_tick] = 0
            seed_ix_note = ix_note
            seed_ix_tick = ix_tick
        return ixes

################################################################################

class Data:
    def __init__(self, filename):
        print("Loading data...")

        self.ix_to_note = pickle.load(open("ix_to_note.p", "rb"))
        self.ix_to_tick = pickle.load(open("ix_to_tick.p", "rb"))

        self.unique_notes = len(self.ix_to_note)
        self.unique_ticks = len(self.ix_to_tick)

        self.note_to_ix = { n:i for i,n in enumerate(self.ix_to_note) }
        self.tick_to_ix = { t:i for i,t in enumerate(self.ix_to_tick) }

        self.X = np.load(filename)
        self.data_size = self.X.shape[0]

        self.reset()

    def reset(self):
        self.p = 0

    def next_batch(self, unroll_steps):
        """Grabs the next chunk of elements."""

        # Reached the end? Then go back to start of data.
        new_epoch = False
        if self.p + unroll_steps + 1 >= self.data_size: 
            new_epoch = True
            self.p = 0

        x, y = self.get_range(self.p, unroll_steps)

        # Move data pointer ahead.
        self.p += unroll_steps

        return x, y, new_epoch

    def get_range(self, start, length):
        x = self.X[start   : start+length  ]
        y = self.X[start+1 : start+length+1]
        return x, y

    def to_text(self, ixes):
        return ",".join(str(self.ix_to_note[ix_note]) + ":" + \
                        str(self.ix_to_tick[ix_tick]) for ix_note, ix_tick in ixes)

################################################################################

def write_32bit(f, value):
    f.write(struct.pack(">I", value))

def write_16bit(f, value):
    f.write(struct.pack(">H", value & 0xffff))

def write_byte(f, value):
    f.write(struct.pack("B", value & 0xff))

def write_var_length(f, value):
    count = 0
    buf = value & 0x7f

    value >>= 7
    while value != 0:
        buf <<= 8
        buf |= (value & 0x7f) | 0x80
        value >>= 7

    while True:
        write_byte(f, buf)
        count += 1
        if buf & 0x80:
            buf >>= 8
        else:
            return count

def write_midi_file(filename, notes_and_ticks):
    print("Saving MIDI file '%s'" % filename)
    with open(filename, "wb") as f:
        f.write(bytes([0x4D, 0x54, 0x68, 0x64]))    # MThd
        write_32bit(f, 6)
        write_16bit(f, 0)                           # format 0
        write_16bit(f, 1)                           # one track
        write_16bit(f, 480)                         # ticks per beat
        f.write(bytes([0x4D, 0x54, 0x72, 0x6b]))    # MTrk

        # Remember this position to write chunk length afterwards.
        length_offset = f.tell()
        write_32bit(f, 0)
        byte_count = 0

        for note, ticks in notes_and_ticks:
            # Write delta time for this event. Subtract 1 tick
            # from the previous NOTE_OFF event.
            delta = max(0, ticks - 1)
            byte_count += write_var_length(f, delta)

            # Write a NOTE_ON event for the new note.
            write_byte(f, 0x9A)      # channel 10
            write_byte(f, note)      # MIDI note number
            write_byte(f, 0x64)      # velocity
            byte_count += 3

            # Write delta time of 1 tick.
            byte_count += write_var_length(f, 1)

            # Write a NOTE_OFF event for the note.
            write_byte(f, 0x8A)      # channel 10
            write_byte(f, note)      # MIDI note number
            write_byte(f, 0x64)      # velocity
            byte_count += 3

        # Write the end-of-track marker.
        byte_count += write_var_length(f, 0)
        write_byte(f, 0xff)
        write_byte(f, 0x2f)
        write_byte(f, 0x00)
        byte_count += 3

        # Fill in the byte_count in the chunk length header.
        f.seek(length_offset)
        write_32bit(f, byte_count)

################################################################################

def train(rnn, data, steps):
    print("Training RNN...")

    tf.gfile.MakeDirs(checkpoint_dir)

    with tf.Session() as sess:
        # For writing training checkpoints and reading them back in.
        saver = tf.train.Saver()

        rnn.prepare_for_training(sess)

        h = np.zeros((1, rnn.hidden_size))
        c = np.zeros((1, rnn.hidden_size))

        # Continue training from a previously saved checkpoint.
        if model_file is not None:
            saver.restore(sess, model_file)

        # Compute initial loss over the first batch, so we have a starting point
        # for smoothing the loss. (Since the loss varies a lot between chunks.)
        x, y, _ = data.next_batch(rnn.unroll_steps)
        feed = {rnn.x: x, rnn.y: y, rnn.h: h, rnn.c: c}
        smooth_loss = sess.run(rnn.loss, feed_dict=feed)
        print("Initial loss: %f" % smooth_loss)

        # Register summary objects for TensorBoard.
        tf.summary.scalar("cross-entropy-loss", rnn.loss)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(summaries_dir, sess.graph)

        epoch = 1

        # Note: I found it useful to train for a while until the accuracy did not
        # improve, then stop, lower the learning rate, and run the script again
        # to resume training from the last checkpoint. You need to change these
        # variables when you do that.
        start_n = 0
        lr = 1e-2

        for n in range(start_n, steps + 1):
            # Get the next chunk of data.
            x, y, new_epoch = data.next_batch(rnn.unroll_steps)

            if new_epoch:
                # Reset the RNN's memory on every new epoch.
                h = np.zeros((1, rnn.hidden_size))
                c = np.zeros((1, rnn.hidden_size))
                epoch += 1

            # Train the RNN.
            loss_value, h, c = rnn.train(sess, x, y, h, c, learning_rate=lr)
            smooth_loss = smooth_loss * 0.999 + loss_value * 0.001

            # Update summaries for TensorBoard.
            if n % 100 == 0:
                feed = {rnn.x: x, rnn.y: y, rnn.h: h, rnn.c: c}
                summary = sess.run(summary_op, feed_dict=feed)
                summary_writer.add_summary(summary, n)
                summary_writer.flush()

            # Print progress.
            if n % 100 == 0:
                print("step %d, epoch: %d, loss: %f (smoothed %f), lr: %g" % \
                        (n, epoch, loss_value, smooth_loss, lr))

            # Sample from the model now and then to see how well it works.
            if n % 1000 == 0:
                seed_ix_note = np.argmax(x[0, :data.unique_notes])
                seed_ix_tick = np.argmax(x[0, data.unique_notes:])
                sampled = rnn.sample(sess, h, c, seed_ix_note, seed_ix_tick, 400)
                print("----\n%s\n----" % data.to_text(sampled))

            # Compute accuracy across the entire dataset.
            if n % 1000 == 0:
                # Run the accuracy op multiple times (once for each possible chunk
                # of data) and average the results.
                num_chunks = data.data_size // rnn.unroll_steps
                print("Computing accuracy over %d chunks... " % num_chunks, end="")
                scores = np.zeros(num_chunks)
                for b in range(num_chunks):
                    x, y = data.get_range(b*unroll_steps, unroll_steps)
                    feed = {rnn.x: x, rnn.y: y, rnn.h: h, rnn.c: c}
                    scores[b] = sess.run(rnn.accuracy, feed_dict=feed)
                print("score: %f" % scores.mean())

            # Save the model.
            if n % 500 == 0:
                checkpoint_file = os.path.join(checkpoint_dir, "model-%d" % n)
                saver.save(sess, checkpoint_file)            
                print("*** SAVED MODEL '%s' ***" % checkpoint_file)

        summary_writer.close()

################################################################################

def sample(rnn, data):
    print("Sampling...")

    with tf.Session() as sess:
        # Load the saved model back into the session. (This automatically loads
        # the weights back into rnn.Wx and rnn.Wy, since these point to the same
        # tensor objects that are in the currently active graph.)
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

        # Start with an empty memory. Note that the output will be somewhat 
        # different every time, since sample() does random sampling on the 
        # output vector.
        #h = np.zeros((1, rnn.hidden_size))
        #c = np.zeros((1, rnn.hidden_size))

        # Or start with a random memory for more varied results. 
        h = np.random.randn(1, rnn.hidden_size) * 0.5
        c = np.random.randn(1, rnn.hidden_size) * 0.5
        
        # Or with uniform random memory.
        #h = np.random.random((1, rnn.hidden_size)) * 0.5
        #c = np.random.random((1, rnn.hidden_size)) * 0.5

        first_ix_note = data.note_to_ix[36]
        first_ix_tick = 0
        sampled = rnn.sample(sess, h, c, first_ix_note, first_ix_tick, 1000)
        print("----\n%s\n----" % data.to_text(sampled))

        notes = []
        for ix_note, ix_tick in sampled:
            notes.append((data.ix_to_note[ix_note], data.ix_to_tick[ix_tick]))

        write_midi_file("generated.mid", notes)

################################################################################

def export_weights(rnn):
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_file)

        print("Wx shape:", rnn.Wx.shape)
        print("Wy shape:", rnn.Wy.shape)

        rnn.Wx.eval().tofile("Wx.bin")
        rnn.Wy.eval().tofile("Wy.bin")

################################################################################

def random_notes(data):
    notes = []
    for i in range(200):
        note_ix = np.random.randint(data.unique_notes)
        tick_ix = np.random.randint(data.unique_ticks)
        notes.append((data.ix_to_note[note_ix], data.ix_to_tick[tick_ix]))

    write_midi_file("random.mid", notes)

################################################################################

data = Data("X.npy")
rnn = RNN(data.unique_notes, data.unique_ticks, hidden_size, unroll_steps)

if mode == "train":
    train(rnn, data, steps=training_steps)
elif mode == "sample":
    sample(rnn, data)
elif mode == "export":
    export_weights(rnn)
elif mode == "random":
    random_notes(data)
