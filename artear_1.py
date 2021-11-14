from datetime import datetime

from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Reshape, Conv2D, Conv2DTranspose, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import RMSprop, Adam
from ssqueezepy import ssq_stft, issq_stft
import random
import librosa
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import os
import sounddevice as sd
import soundfile as sf

K.clear_session()

latent_dim = 32
spec_channels = 1
spec_shape = (128, 256)

gen_input = Input(shape=(latent_dim,))

x = Dense(128 * 256, activation='LeakyReLU')(gen_input)
x = Reshape((16, 16, 128))(x)

x = Conv2D(256, 5, padding='same', activation='LeakyReLU')(x)
x = Conv2DTranspose(256, 4, strides=8, padding='same', activation='LeakyReLU')(x)

x = Conv2D(128, 5, padding='same', activation='LeakyReLU')(x)
x = Conv2D(2, 5, padding='same', activation='LeakyReLU')(x)

x = Reshape((128, 256, 1))(x)
x = Conv2D(spec_channels, 7, activation='tanh', padding='same')(x)

gen = Model(gen_input, x)
gen.trainable = True
gen.summary()

dis_input = Input(shape=spec_shape + (spec_channels,))

x = Conv2D(64, 3, activation='LeakyReLU')(dis_input)
x = Conv2D(64, 4, strides=2, activation='LeakyReLU')(x)
x = Conv2D(64, 4, strides=2, activation='LeakyReLU')(x)
x = Conv2D(64, 4, strides=2, activation='LeakyReLU')(x)

x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(1, activation='sigmoid')(x)

dis = Model(dis_input, x)
dis.compile(optimizer=Adam(), loss='binary_crossentropy')
dis.trainable = True
dis.summary()

gan_input = Input(shape=(latent_dim,))
gan_output = dis(gen(gan_input))

gan = Model(gan_input, gan_output)
gan.compile(optimizer=Adam(), loss='binary_crossentropy')


def pool2d(A, kernel_size=(3, 3), stride=(2, 100), padding=2, pool_mode='max'):
    A = np.pad(A, padding, mode='constant')

    output_shape = ((A.shape[0] - kernel_size[0]) // stride[0] + 1,
                    (A.shape[1] - kernel_size[1]) // stride[1] + 1)
    A_w = as_strided(A, shape=output_shape + kernel_size,
                     strides=(stride[0] * A.strides[0],
                              stride[1] * A.strides[1]) + A.strides)
    A_w = A_w.reshape(-1, *kernel_size)

    if pool_mode == 'max':
        return A_w.max(axis=(1, 2)).reshape(output_shape)
    elif pool_mode == 'avg':
        return A_w.mean(axis=(1, 2)).reshape(output_shape)


def get_files(directory, ext):
    path_list = []
    for paths in [[os.path.join(dirpath, name).replace('\\', '/') for name in filenames if name.endswith('.' + ext)] for
                  dirpath, dirnames, filenames in os.walk(directory)]:
        path_list.extend(paths)
    return path_list


def resample(input_signal, src_fs, tar_fs):
    audio_len = len(input_signal)
    audio_time_max = 1.0*(audio_len-1) / src_fs
    src_time = 1.0 * np.linspace(0, audio_len, audio_len) / src_fs
    tar_time = 1.0 * np.linspace(0, np.int(audio_time_max*tar_fs), np.int(audio_time_max*tar_fs)) / tar_fs
    output_signal = np.interp(tar_time, src_time, input_signal).astype(input_signal.dtype)
    return output_signal


def get_real_data(wav_files, num, cutoff=(128, 256)):
    lst = []
    sr_, ratio = 0, 0.
    for _ in range(num):
        dat, sr_ = librosa.load(random.choice(wav_files))
        # dat = resample(dat, 44100, 512)
        dat, _, *_ = ssq_stft(dat)
        prev_len = dat.shape[1]
        dat = pool2d(dat)
        ratio = dat.shape[1] / prev_len
        dat = dat[:cutoff[0], :cutoff[1]]
        dat = np.pad(dat, (0, cutoff[1] - dat.shape[1] if dat.shape[1] < cutoff[1] else 0), 'constant'
                     , constant_values=(0., 0.)).transpose((1, 0))
        dat = np.pad(dat, (0, cutoff[0] - dat.shape[1] if dat.shape[1] < cutoff[0] else 0), 'constant'
                     , constant_values=(0., 0.)).transpose((1, 0))
        # plot_data(dat)
        lst.append(dat)
        # print((len(lst),) + lst[0].shape + (1,))
    return np.concatenate(lst).reshape((len(lst),) + lst[0].shape + (1,)), sr_ * ratio


def plot_data(dat, save_to=''):
    plt.imshow(np.abs(dat), aspect='auto', cmap='jet')
    if save_to:
        plt.savefig(save_to)
    plt.show()


def to_wav(dat, sr_, save_to='', play=False):
    print('采样率：', sr_, ' -> 44100')
    wav_dat = issq_stft(dat)
    if save_to:
        sf.write(save_to, resample(wav_dat[:, 0], sr_, 44100), 44100)
    if play:
        sd.play(resample(wav_dat[:, 0], sr_, 44100), 44100)
    return wav_dat


sr = 220.5
n_iter = 10000
batch_size = 10
wav_dir = r'C:\dev_spa\artear_1\437_wavs'
save_root_dir = r'C:\dev_spa\artear_2\data_gen'
save_dir = save_root_dir + '\\' + str(datetime.now()).split('.')[0].replace(':', '-').replace(' ', '_')
os.mkdir(save_dir)
os.mkdir(save_dir + '\\img')
weights_file = save_root_dir + '\\artear_gan.h5'

if os.path.exists(weights_file):
    gan.load_weights(weights_file)

wav_files = get_files(wav_dir, 'wav')


def train_dis():
    global dis, gen
    loss = 0.
    sr = 0.
    dis.trainable = True
    gen.trainable = False
    for _ in range(1):
        rand_lat_vecs = np.random.normal(size=(batch_size, latent_dim))
        generated_imgs = gen.predict(rand_lat_vecs)
        real_imgs, sr = get_real_data(wav_files, batch_size)

        combined_imgs = np.concatenate([generated_imgs, real_imgs])
        labels = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        labels += 0.05 * np.random.random(labels.shape)
        loss = dis.train_on_batch(combined_imgs, labels)
        # print('dis_loss:', dis_loss)

    rand_lat_vecs = np.random.normal(size=(batch_size, latent_dim))
    pred = gen.predict(rand_lat_vecs)[0]
    # to_wav(pred, sr, play=True)
    plot_data(pred)
    tell = dis.predict(pred)
    print(tell)
    return loss, sr


def train_adv():
    global gan, gen
    
    loss = 0.
    dis.trainable = False
    gen.trainable = True
    for _ in range(2):
        rand_lat_vecs = np.random.normal(size=(batch_size, latent_dim))
        misleading_targets = np.zeros((batch_size, 1))
        loss = gan.train_on_batch(rand_lat_vecs, misleading_targets)
        # print('adv_loss:', adv_loss)
    rand_lat_vecs = np.random.normal(size=(batch_size, latent_dim))
    pred = gen.predict(rand_lat_vecs)[0]
    to_wav(pred, sr, play=True)
    plot_data(pred)
    return loss


dis_loss, adv_loss = 0., 0.


for step in range(n_iter):
    print('[' + str(step + 1) + '/' + str(n_iter) + '] 正在训练…')

    dis_loss, sr = train_dis()
    adv_loss = train_adv()

    '''inp = input()
    if not inp or inp.startswith('z'):
        dis_loss, sr = train_dis()
    elif inp == 'q':
        break
    else:
        adv_loss = train_adv()'''

    gan.save_weights(weights_file)

    '''if step % 5 == 0:
        rand_lat_vecs = np.random.normal(size=(batch_size, latent_dim))
        prediction = gen.predict(rand_lat_vecs)[0]
        to_wav(prediction, sr, save_to=save_dir + '\\' + str(int(step / 10)) + '.wav', play=True)
        plot_data(prediction, save_to=save_dir + '\\img\\' + str(int(step / 10)) + '.png')'''

    print('dis_loss:', dis_loss, 'adv_loss:', adv_loss)
