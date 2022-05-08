import wandb
import tensorflow as tf
from make_dataset import PrepareData
from encoders import  GRU_enc 
from decoders import  GRU_dec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.font_manager import FontProperties
import os
import numpy as np
import random
import seaborn as sns

# prepare the dataset and tokenizers for training validation and testing
a = PrepareData()
train_path = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv'
test_path = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv'
val_path = 'dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv'
ip_tensor, op_tensor, ip_tokenizer, op_tokenizer = a.return_data(train_path)
len_ip_max, len_op_max  = ip_tensor.shape[1], op_tensor.shape[1]

# we found the best model was GRU so finetuning was done on GRU for the attention model
type_ = 'GRU'

def train(train_for_one):
    global embed_dim
    global rnn_units
    global batch_size
    # if we want to train a single model train_for_one = 1
    if(train_for_one):
        rnn_units = 128
        embed_dim = 512
        drpout = 0.2
        epochs = 1
        batch_size = 128
    else:
        # if we want to sweep over all hyperparameter configurations, use wandb
        wandb.init()
        rnn_units = int(wandb.config.units)
        embed_dim = int(wandb.config.embed_dim)
        drpout = float(wandb.config.dropout)
        epochs = int(wandb.config.epochs)
        batch_size = int(wandb.config.batch_size)
        
    bs = len(ip_tensor)
    steps = int(len(ip_tensor)/batch_size)
    global op_size_vc
    global ip_size_vc
    ip_size_vc = len(ip_tokenizer.word_index) + 1
    op_size_vc = len(op_tokenizer.word_index) + 1

    global run_
    run_ = 'type_' + type_ + '_units_' + str(rnn_units) + '_embed_dim_' + str(embed_dim) + '_bs_' + str(batch_size) + '_dropout_' + str(drpout) + '_epochs_' + str(epochs)
    if(train_for_one != 1):
        wandb.run.name = run_
    train_dataset = tf.data.Dataset.from_tensor_slices((ip_tensor, op_tensor)).shuffle(bs)
    # we create batches of data of batch_size and only the current randomly chosen data is retained for this iteration 
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    # each element in iterable train_dataset is taken one after the other using next
    ip_batch, op_batch = next(iter(train_dataset))
    global enc
    # the encoder is created from the GRU encoder class
    enc = GRU_enc(ip_size_vc, embed_dim, rnn_units, batch_size, drpout)
    hidden_st = enc.init_hidden()
    op_st, hidden_st = enc(ip_batch, hidden_st)
    global dec
    # the decoder is made from the GRU decoder class
    dec = GRU_dec(op_size_vc, embed_dim, rnn_units, batch_size, drpout)
    dec_op, _, _ = dec(tf.random.uniform((batch_size, 1)), hidden_st, op_st)
    global opt
    # the Adam optimizer is initialized
    opt = tf.keras.optimizers.Adam()
    global loss_cat_ent
    # the cross entropy loss object is initialized
    loss_cat_ent = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    losses = [0] * epochs

    for x in range(epochs):
        loss = 0
        enc_h = enc.init_hidden()
        for (batch, (ip, op)) in enumerate(train_dataset.take(steps)):
            b_loss = train_step(ip, op, enc_h, enc, dec)
            loss += b_loss
        if batch % 100 == 0:
            print(f'Epoch {x+1} Batch {batch} Loss {b_loss.numpy():.4f}')
        print(f'Epoch {x+1} Loss {loss/steps:.4f}')
        losses[x] = loss.numpy()/steps
        if(train_for_one != 1):
            wandb.log({'train_loss' : loss.numpy()/steps})
    
    # the test and validation accuracy are calculated for each configuration of the model

    # test_acc = validation(test_path, run_)
    val_acc = validation(val_path, type_)
    if(train_for_one != 1):
        wandb.log({'val_acc': val_acc})
    # print("Test Accuracy: ",test_acc)
    print("Validation Accuracy: ", val_acc)
    
    # the random_test_words function can be uncommented to generate the connectivity plot and heatmaps for 
    # the output from one configuration of the model
    # better to use for train(1) --> train for one is true (trains on the best model only, does not sweep)

    random_test_words(10)

# this function is used to calculate the validation and test accuracy
def validation(path_to_data, sub_folder_name):
    folder_path = 'predictions_attention/' + str(sub_folder_name) + '/'
    if(os.path.exists( os.getcwd() + '/' + folder_path)):
        p = 1
    else:
        os.mkdir(folder_path)
    # the correct and wrong predictions are written in two separate files
    correct_op = open(folder_path + "correct.txt", "w" , encoding='UTF-8')
    wrong_op = open(folder_path + "wrong.txt", "w" , encoding='UTF-8')
    c = 0
    a = PrepareData()
    op_w, ip_w = a.preprocess(path_to_data)
    for x in range(len(ip_w)):
        # run the inference model and get the predictions
        pred_word, ip_word,  _, _ = run_inference(ip_w[x], type_)
        rec = ip_word.strip() + ' ' + op_w[x].strip() + ' ' + pred_word[:-1].strip() + '\n'
        if(op_w[x][1:] == pred_word):
            c += 1
            correct_op.write(rec)
        else:
            wrong_op.write(rec)
    correct_op.close()
    wrong_op.close()
    #return accuracy
    return c / len(ip_w)

# this function uses teacher forcing to train the model
def train_step(ip, op, enc_h, enc, dec):
    c = 0
    with tf.GradientTape() as tape:
        enc_op, enc_h = enc(ip, enc_h)
        dec_h = enc_h
        dec_ip = tf.expand_dims([op_tokenizer.word_index['\t']] * batch_size, 1)
        # pass the target as the next input (teacher forcing)
        for x in range(1, op.shape[1]):
            # pass the output to the decoder
            pred, dec_h, _ = dec(dec_ip, dec_h, enc_op)
            c += masked_loss(op[:, x], pred)
            dec_ip = tf.expand_dims(op[:, x], 1)
        #batch loss is computed
        b_loss = (c / int(op.shape[1]))
        # the gradient for each trainable parameters in both encoder and decoder each 
        # applied to the parameters
        var_ = enc.trainable_variables + dec.trainable_variables
        grad = tape.gradient(c, var_)
        # apply the gradients that have been calculated
        opt.apply_gradients(zip(grad, var_))
        return b_loss

# this function calculates loss for sequences
def masked_loss(op, pred):
    # all the seqences are turned to same length by padding zeros
    m_pos = tf.math.logical_not(tf.math.equal(op, 0))
    l_val = loss_cat_ent(op, pred)
    # the positions for where the sequence is 0 is computed
    # and its loss contribution is made zero
    m_pos = tf.cast(m_pos, dtype=l_val.dtype)
    l_val *= m_pos
    return tf.reduce_mean(l_val)

# this method does not use teacher forcing
def run_inference(ip_word, type_):
    attention = np.zeros((len_op_max, len_ip_max))
    a = PrepareData()
    ip_word = a.add_chars(ip_word)
    # tokenizing and making sequence of the given word
    ip_s = [ip_tokenizer.word_index[i] for i in ip_word]
    ip_s = tf.keras.preprocessing.sequence.pad_sequences([ip_s], maxlen=len_ip_max, padding='post')
    ip_s = tf.convert_to_tensor(ip_s)
    pred_word = ''
    h = [tf.zeros((1, rnn_units))]
    enc_op, enc_h = enc(ip_s, h)
    dec_h = enc_h
    dec_ip = tf.expand_dims([op_tokenizer.word_index['\t']], 0)
    all_att_wts = []
    # the input at every time step is the previous prediction, encode output and hidden state 
    for x in range(len_op_max):
        pred, dec_h, att_wt = dec(dec_ip, dec_h, enc_op)
        att_wt = tf.reshape(att_wt, (-1, ))
        attention[x] = att_wt.numpy()
        # all the attention weights are stored for creating heatmaps
        all_att_wts.append(att_wt.numpy()[0:len(ip_word)])
        pred_index = tf.argmax(pred[0]).numpy()
        pred_word += op_tokenizer.index_word[pred_index]
        # stop predicting when the ouput token is '\n'
        if op_tokenizer.index_word[pred_index] == '\n':
            return pred_word, ip_word, attention, all_att_wts
        dec_ip = tf.expand_dims([pred_index], 0)
    return pred_word, ip_word,  attention, all_att_wts

# generate specified number of words for heatmap generation
def random_test_words(n_words):
    op_ws, ip_ws = a.preprocess(test_path)
    for x in range(n_words):
        id_ = random.randint(0, len(ip_ws))
        ip_w = ip_ws[id_]
        sub_folder_name = "predictions_attention/" +  str(run_) + '/'
        file_ = sub_folder_name + ip_w[1:-1] + '.png'
        gen_plot(ip_w[1:-1], file_, sub_folder_name)

# generate the heatmap for each word
def gen_plot(ip_w, file_, sub_folder_name):
    # get the attention weights for the input word
    pred_w, ip_w, attention, all_att_weights = run_inference(ip_w, type_)
    print('Input: ', ip_w)
    print('Prediction: ', pred_w)
    attention = attention[:len(pred_w), :len(ip_w)]
    # pass the attention weights to make the heatmap plots
    make_plot(attention, ip_w, pred_w, file_, sub_folder_name)
    txt = 'Input: {} <br> Prediction: {}'.format(ip_w, pred_w) + '<br>'
    # pass the same attention weights to visualize the 
    connectivity(ip_w, pred_w, all_att_weights, txt)

# make the heatmap plot and save it as image
def make_plot(attention, ip_w, pred_w, file_, sub_folder_name):

    folder_path = sub_folder_name
    if(os.path.exists( os.getcwd() + '/' + folder_path)):
        p = 1
    else:
        os.mkdir(folder_path)
    fig = plt.figure(figsize=(3, 3))
    # a hindi font is used
    font_ = FontProperties(fname = "fonts/Nirmala.ttf")
    ax = fig.add_subplot(1, 1, 1)
    # the color map is set to cividis
    sns.heatmap(attention, cbar=False)
    ax.matshow(attention, cmap='cividis') # cividis
    fontdict = {'fontsize': 12}
    ax.set_xticklabels([''] + list(ip_w), fontdict=fontdict, rotation=0)
    ax.set_yticklabels([''] + list(pred_w), fontdict=fontdict, rotation=0, fontproperties=font_)
    plt.savefig(file_)

# methods to generate colorful backgrounds for the text
def cstr(s, color='black'):
	if s == ' ':
		return "<text style=color:#000;padding-left:10px;background-color:{}> </text>".format(color, s)
	else:
		return "<text style=color:#000;background-color:{}>{} </text>".format(color, s)

def get_clr(value):
	colors = ['#85c2e1', '#89c4e2', '#95cae5', '#99cce6', '#a1d0e8',
		'#b2d9ec', '#baddee', '#c2e1f0', '#eff7fb', '#f9e8e8',
		'#f9e8e8', '#f9d4d4', '#f9bdbd', '#f8a8a8', '#f68f8f',
		'#f47676', '#f45f5f', '#f34343', '#f33b3b', '#f42e2e']
	value = int((value * 100) / 5)
	return colors[value]

# the text and its corresponding background color is appended together as text for the html file
def print_color(t):
    txt_ = ''.join([cstr(ti, color=ci) for ti,ci in t])
    return txt_

# generates the html file that contains the visualization for the randomly chosen words' attention 
def connectivity(ip_w, op_w, att_w, txt_):
    all_text = ''
    all_text += txt_
    for x in range(len(op_w)):
        all_text += "Output Character : " + str(op_w[x]) + '<br>'
        colors = []
        for y in range(len(att_w[x])):
            txt = (ip_w[y], get_clr(att_w[x][y]))
            colors.append(txt)
        all_text += print_color(colors) + '<br>'
    # save the html file as connectivity.html
    f_name = 'predictions_attention/connectivity.html'
    f = open(f_name,"a")
    f.write(all_text)
    f.close()

train(1)
            

