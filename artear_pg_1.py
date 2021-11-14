import gc
import os
import random
import time

import dawdreamer as daw
import numpy as np
from datetime import datetime

from keras import Input
from keras.layers import Dense, Conv2D, Flatten, Dropout, LSTM, Reshape, SeparableConv2D, Conv1D, Conv1DTranspose
from keras.models import Model
from keras import backend as K
from keras.optimizer_v2 import adam
from keras.utils.np_utils import to_categorical

from ssqueezepy import ssq_stft, issq_stft, ssq_cwt, issq_cwt
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

from scipy.signal import resample
import librosa

K.clear_session()

sample_rate = 44100
buffer_size = 256
cutoff_time = 2.

to_render = True

plugin_paths = {
    'Sylenth1': 'C:/VstPlugins/64bit/Sylenth1.dll',
    'Massive': 'C:/VstPlugins/64bit/Massive.dll',
}

# fxp_dir = r'C:\dev_spa\artear_1\res\LEAD'
fxp_dir = r'G:\编曲资源'
mid_dir = r'G:\编曲资源'
wav_gen_dir = r'C:\dev_spa\artear_pg_1\wav_rendered'


def get_files(directory, ext, with_dot=True):
    path_list = []
    for paths in [[os.path.join(dirpath, name).replace('\\', '/') for name in filenames if
                   name.endswith(('.' if with_dot else '') + ext)] for
                  dirpath, dirnames, filenames in os.walk(directory)]:
        path_list.extend(paths)
    return path_list


def plot_spec(*specs, title=(), save_to=''):
    mx = max(len(specs), specs[0].shape[0])
    for i in range(len(specs)):
        for j in range(specs[i].shape[0]):
            plt.subplot2grid((mx, mx), (j, i), rowspan=(mx if specs[i].shape[0] == 1 else 1))
            plt.imshow(np.abs(specs[i][j, :, :, 0]), aspect='auto', cmap='jet')
            if j == 0:
                plt.title(title[i])

    if save_to:
        plt.savefig(save_to)
    plt.show()


'''def spec_to_audio(dat, spec_sr, dst_sr, save_to='', play=False, blocking=True):
    dat_ = []
    for i in range(dat.shape[0]):
        print('采样率：', spec_sr, '->', dst_sr)
        da = issq_cwt(dat[i, :, :, :]).reshape((-1))
        # print('da:', da.shape, '...:', dat[i, :, :, :].shape)
        da = resample(da, int(da.shape[0] * dst_sr / spec_sr))
        if save_to:
            sf.write(save_to, da, dst_sr)
        if play:
            print('playing:', da.shape, da)
            sd.play(da, dst_sr, blocking=blocking)
        dat_.append(da)
    return np.concatenate(dat_).reshape((len(dat_),) + dat_[0].shape)'''


def play(data, sr=sample_rate, blocking=False, save_to=''):
    print('正在播放：' + str(sr) + ' Hz')
    for i in range(data.shape[0]):
        if data.ndim == 3:
            if data.shape[1] == 1:
                dat = data[i, ...].reshape(-1)
            else:
                dat = librosa.to_mono(data[i, ...])
            sd.play(dat, sr, blocking=blocking)
            if save_to:
                sf.write(save_to, dat, sr)
        else:
            sd.play(data[i, ...], sr, blocking=blocking)
            if save_to:
                sf.write(save_to, data[i, ...], sr)
    print('播放已结束')


print('正在获取路径下所有预设和MIDI文件……')
fxp_paths = get_files(fxp_dir, 'nmsv')
mid_paths = get_files(mid_dir, 'mid')

engine = daw.RenderEngine(sample_rate, buffer_size)

instr = engine.make_plugin_processor('instr_1', plugin_paths['Massive'])
engine.load_graph([(instr, [])])


def gauss(x, u, sigma):
    return K.exp(- K.square(x - u) / (2 * K.square(sigma))) / (K.sqrt(2 * np.pi) * sigma)


def generator(input_shape, params_shape):
    layer_in = Input(shape=input_shape)
    x = SeparableConv2D(64, 2, strides=(2, 2), activation='LeakyReLU')(layer_in)
    x = SeparableConv2D(64, 2, strides=(2, 2), activation='LeakyReLU')(x)
    x = SeparableConv2D(64, 2, strides=(2, 2), activation='LeakyReLU')(x)
    x = SeparableConv2D(64, 2, strides=(2, 2), activation='LeakyReLU')(x)

    x = Dense(4, activation='LeakyReLU')(x)
    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(params_shape[-2] * params_shape[-1])(x)
    x = Reshape(params_shape[-2:])(x)
    '''x = Reshape((-1, params_shape[-1]))(x)
    x = Conv2D(params_shape[-2] * params_shape[-1], 1)(x)
    x = Reshape(params_shape[-2:])(x)
    x = Conv1DTranspose(16, 2, padding='same', strides=2, activation='LeakyReLU')(x)'''

    # 99x = LSTM(n_params)(x)
    x = Dense(params_shape[-1], activation='softmax')(x)

    modl = Model(layer_in, x)

    modl.compile(optimizer=adam.Adam(), loss='categorical_crossentropy', metrics='acc')
    return modl


def discriminator(params_shape):
    layer_in = Input(shape=params_shape)
    x = Dense(16, activation='LeakyReLU')(layer_in)
    x = Dense(16, activation='LeakyReLU')(x)
    x = Flatten()(x)
    x = Dense(8, activation='LeakyReLU')(x)
    x = Dense(1, activation='sigmoid')(x)

    modl = Model(layer_in, x)
    modl.compile(optimizer=adam.Adam(), loss='binary_crossentropy', metrics='acc')
    return modl


def stereo_to_mono(dat):
    if dat.ndim > 1:
        dat = (dat[0, ...] + dat[1, ...]) / 2.
    return dat


def audio_to_spec(data, src_sr, cutoff_t=2., factor=2e-3):
    data_ = []
    data = data[:, :, :int(src_sr * cutoff_t)]

    for i in range(data.shape[0]):
        dat = librosa.to_mono(data[i, ...])
        dat = resample(dat, int(dat.shape[-1] * factor))
        dats = []
        dats += ssq_stft(dat)[:2]
        dats += ssq_cwt(dat)[:2]
        # print(dat.shape)
        for j in range(len(dats)):
            dats[j] = dats[j].reshape(dats[j].shape + (1,))
        data_.append(np.concatenate(dats, axis=-1))

    return np.concatenate(data_).reshape((len(data_),) + data_[0].shape), src_sr * factor


def render_from_params(params, mid):
    global instr, engine
    audio_ = []
    for i in range(params.shape[0]):
        for index, param in enumerate(params[i, ...]):
            instr.set_parameter(index, 1 / (1 + np.exp(-np.argmax(param).astype(np.float32) / 8.)))
        instr.load_midi(mid)
        engine.render(cutoff_time + 1.)
        audio = engine.get_audio()
        audio_.append(audio.reshape((1,) + audio.shape))
    return np.vstack(audio_)


def get_data(num=1):
    global instr, engine
    data, parameters, audios, mids, spec_sr = [], [], [], [], 0.

    for _ in range(num):
        fxp = random.choice(fxp_paths)
        # print('正在加载并渲染：' + fxp)

        instr.load_preset(fxp)
        params = []
        for i in range(instr.get_plugin_parameter_size()):
            params.append(instr.get_parameter(i))
        params = to_categorical((np.array(params) * 8.).astype(np.int), 9)
        parameters.append(params)

        mid = random.choice(mid_paths)
        instr.load_midi(mid)
        mids.append(mid)

        engine.render(cutoff_time + 1.)
        audio = engine.get_audio()
        dat, spec_sr = audio_to_spec(audio.reshape((1,) + audio.shape), sample_rate, cutoff_t=cutoff_time)
        data.append(dat)
        audios.append(audio)

    return np.concatenate(data).reshape(((-1,) + data[0].shape[1:])), \
           np.concatenate(parameters).reshape(((-1,) + parameters[0].shape)), audios, mids, spec_sr


K.clear_session()

n_iter = 10000
batch_size = 10
training = 'gen'

save_root_dir = r'C:\dev_spa\artear_pg_1\data_gen'
save_dir = save_root_dir + '\\' + str(datetime.now()).split('.')[0].replace(':', '-').replace(' ', '_')
os.mkdir(save_dir)
os.mkdir(save_dir + '\\img')
os.mkdir(save_dir + '\\wav_gen')
gan_weights_file = save_root_dir + '\\artear_pg_1_gan.h5'
test_log_dir = save_dir + '\\test_logs'
os.mkdir(test_log_dir)

for step in range(1, n_iter + 1):
    print('[' + str(step) + '/' + str(n_iter) + '] 正在加载数据……')

    specs, params, _, _, spec_sr = get_data(batch_size)
    print('[' + str(step) + '/' + str(n_iter) + '] 已加载训练数据', specs.shape, params.shape, '，正在创建模型……', )

    dis = discriminator(params.shape[1:])
    gan_in = Input(shape=specs.shape[1:])
    gen = generator(specs.shape[1:], params.shape[1:])
    print(gen.summary())
    print(dis.summary())
    gan_out = dis(gen(gan_in))
    gan = Model(gan_in, gan_out)
    gan.compile(optimizer=adam.Adam(), loss='binary_crossentropy', metrics='acc')

    if step == 1:
        gan.summary()

    if os.path.exists(gan_weights_file):
        gan.load_weights(gan_weights_file)


    def collect_data_for_dis(pred, params):
        rand_ins = [(params[i, ...], 0, specs[i, ...]) for i in range(params.shape[0])] + [(pred[i, ...], 1, None) for i
                                                                                           in range(pred.shape[0])]
        rand_ins = random.sample(rand_ins, specs.shape[0])

        in_data = np.concatenate([params for params, _, spec in rand_ins]).reshape(params.shape)
        out_data = np.array([[label] for _, label, _ in rand_ins])
        return in_data, out_data


    print('[' + str(step) + '/' + str(n_iter) + '] 正在训练……')

    loss, metrics = [0.] * 3, [0.] * 3

    if training == 'dis':
        gen.trainable = False
        dis.trainable = True

        pred = gen.predict(specs)
        loss[1], metrics[1] = dis.train_on_batch(*collect_data_for_dis(pred, params))

        loss[2], metrics[2] = gan.train_on_batch(specs, np.zeros((specs.shape[0], 1)))
    else:
        gen.trainable = True
        dis.trainable = False

        print(specs.shape, params.shape)
        loss[0], metrics[0] = gen.train_on_batch(specs, params)

    print('[' + str(step) + '/' + str(n_iter) + '] gen_l:', loss[0], 'gen_a:', metrics[0]
          , 'dis_l', loss[1], 'dis_a', metrics[1]
          , 'gan_l', loss[1], 'gan_a', metrics[1])

    gan.save_weights(gan_weights_file)

    if step % 1 == 0:
        src_spec, src_params, audios, mids, src_spec_sr = get_data()
        print('[' + str(step) + '/' + str(n_iter) + '] 测试数据：', src_spec.shape, src_params.shape,
              os.path.basename(mids[0]), src_spec_sr)
        print('aaa:', gen.predict(src_spec).shape)
        pred_audio = render_from_params(gen.predict(src_spec), mids[0])
        pred_spec, _ = audio_to_spec(pred_audio, sample_rate, cutoff_t=cutoff_time)

        plot_spec(src_spec, pred_spec, title=('original', 'predicted'),
                  save_to=save_dir + '\\img\\step_' + str(step) + '.png')

        play(audios[0].reshape((1,) + audios[0].shape), sr=sample_rate, blocking=True,
             save_to=save_dir + '\\wav_gen\\step_' + str(step) + '_orig.wav')
        time.sleep(0.5)
        play(pred_audio, sr=sample_rate, blocking=True, save_to=save_dir + '\\wav_gen\\step_' + str(step) + '_pred.wav')

    K.clear_session()
    del gen, dis, gan
    gc.collect()

print('完成。')
