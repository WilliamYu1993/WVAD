import os, sys, pdb, tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="0" #Your GPU number, default = 0
from keras.callbacks import ModelCheckpoint_iter
from keras.models import Sequential, Model
from keras.utils.np_utils import to_categorical as tg
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.35
session = tf.Session(config=config)
KTF.set_session(session)
import time
import numpy as np
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
random.seed(999)

from data_loader import get_filepaths
from feature_extraction import read_fvad_frame, read_wav
from model import FCN_V8, FCN_V9, SEDEC, FCN_SE
from loss import categorical_hinge

Num_traindata= 5400
batch_size=1
iteration=30
vad_iteration=2
se_iteration=1

def creatdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def vad_data_generator(noisy_list, clean_path, vad_path, shuffle="True"):
    #print(noisy_list)
    #random.shuffle(noisy_list)
    index=0
    while True:
        noisy = read_wav(noisy_list[index])
        #clean = read_wav(clean_path+noisy_list[index].split('/')[-1])
        #vad = read_fvad_frame(vad_path+noisy_list[index].split('/')[-5]+'/'+noisy_list[index].split('/')[-1][:-4], cat=True, form='.txt')
        vad = read_fvad_frame(vad_path+noisy_list[index].split('/')[-1][:-4], cat=True, form='.npy')
        #print(noisy_list[index], vad_path+noisy_list[index].split('/')[-1][:-4])
        #print(noisy_list[index])
        length = ((len(noisy)-512)/256)+1
        if length > len(vad):
           noisy=noisy[0:512+(len(vad)-1)*256]
        if length < len(vad):
           vad=vad[0:length-len(vad)]
        noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))
        #vad=vad+1
        #vad=np.expand_dims(vad, -1)
        vad=np.reshape(vad,(1,np.shape(vad)[0],2))
        #print(np.shape(vad), np.shape(noisy))
        index += 1
        if index == len(noisy_list):
            index = 0
            if shuffle == "True":
                random.shuffle(noisy_list)
    
        yield noisy, vad

def se_data_generator(noisy_list, clean_path, vad_path, shuffle="True"):
    #print(noisy_list)
    #random.shuffle(noisy_list)
    index=0
    while True:
        noisy = read_wav(noisy_list[index])
        clean = read_wav(clean_path+noisy_list[index].split('/')[-1])
        #vad = read_fvad_frame(vad_path+noisy_list[index].split('/')[-5]+'/'+noisy_list[index].split('/')[-1][:-4], cat=True, form='.txt')
        vad = read_fvad_frame(vad_path+noisy_list[index].split('/')[-1][:-4], cat=True, form='.txt')
        #print(noisy_list[index], vad_path+noisy_list[index].split('/')[-1][:-4])
        #print(noisy_list[index])
        length = ((len(noisy)-512)/256)+1
        if length > len(vad):
           noisy=noisy[0:512+(len(vad)-1)*256]
        if length < len(vad):
           vad=vad[0:length-len(vad)]
        noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))

        length = ((len(clean)-512)/256)+1
        if length > len(vad):
           clean=clean[0:512+(len(vad)-1)*256]
        if length < len(vad):
           vad=vad[0:length-len(vad)]
        clean=np.reshape(clean,(1,np.shape(clean)[0],1))
        #vad=vad+1
        #vad=np.expand_dims(vad, -1)
        #vad=np.reshape(vad,(1,np.shape(vad)[0],2))
        #print(np.shape(vad), np.shape(noisy))
        index += 1
        if index == len(noisy_list):
            index = 0
            if shuffle == "True":
                random.shuffle(noisy_list)
    
        yield noisy, clean

def se_TIMIT_data_generator(noisy_list, clean_path, vad_path, shuffle="True"):

    index=0
    while True:
        noisy = read_wav(noisy_list[index])
        clean = read_wav(clean_path+noisy_list[index].split('/')[-1])
    
        length = ((len(noisy)-512)/256)+1
        noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))

        length = ((len(clean)-512)/256)+1
        clean=np.reshape(clean,(1,np.shape(clean)[0],1))

        index += 1
        if index == len(noisy_list):
            index = 0
            if shuffle == "True":
                random.shuffle(noisy_list)
    
        yield noisy, clean

def vad_valid_generator(noisy_list, clean_path, vad_path, shuffle="True"):

    index=0
    while True:
        noisy = read_wav(noisy_list[index])
        #clean = read_wav(clean_path+noisy_list[index].split('/')[-1])
        #vad = read_fvad_frame(vad_path+noisy_list[index].split('/')[-5]+'/'+noisy_list[index].split('/')[-1][:-4], cat=True, form='.txt')
        vad = read_fvad_frame(vad_path+noisy_list[index].split('/')[-1][:-4], cat=True, form='.npy')
        length = ((len(noisy)-512)/256)+1
        if length > len(vad):
           noisy=noisy[0:512+(len(vad)-1)*256]
        if length < len(vad):
           vad=vad[0:length-len(vad)]
        noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))
        #vad=vad+1
        #vad=np.expand_dims(vad, -1)
        vad=np.reshape(vad,(1,np.shape(vad)[0],2))
        index += 1
        if index == len(noisy_list):
            index = 0
            if shuffle == "True":
                random.shuffle(noisy_list)

        yield noisy, vad

def se_valid_generator(noisy_list, clean_path, vad_path, shuffle="True"):

    index=0
    while True:
        noisy = read_wav(noisy_list[index])
        clean = read_wav(clean_path+noisy_list[index].split('/')[-1])
        #vad = read_fvad_frame(vad_path+noisy_list[index].split('/')[-5]+'/'+noisy_list[index].split('/')[-1][:-4], cat=True, form='.txt')
        vad = read_fvad_frame(vad_path+noisy_list[index].split('/')[-1][:-4], cat=True, form='.txt')
        length = ((len(noisy)-320)/160)+1
        if length > len(vad):
           noisy=noisy[0:320+(len(vad)-1)*160]
        if length < len(vad):
           vad=vad[0:length-len(vad)]
        noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))

        length = ((len(clean)-320)/160)+1
        if length > len(vad):
           clean=clean[0:320+(len(vad)-1)*160]
        if length < len(vad):
           vad=vad[0:length-len(vad)]
        clean=np.reshape(clean,(1,np.shape(clean)[0],1))
        #vad=vad+1
        #vad=np.expand_dims(vad, -1)
        #vad=np.reshape(vad,(1,np.shape(vad)[0],2))
        index += 1
        if index == len(noisy_list):
            index = 0
            if shuffle == "True":
                random.shuffle(noisy_list)

        yield noisy, clean

def se_TIMIT_valid_generator(noisy_list, clean_path, vad_path, shuffle="True"):

    index=0
    while True:
        noisy = read_wav(noisy_list[index])
        clean = read_wav(clean_path+noisy_list[index].split('/')[-1])

        length = ((len(noisy)-512)/256)+1
        noisy=np.reshape(noisy,(1,np.shape(noisy)[0],1))

        length = ((len(clean)-512)/256)+1
        clean=np.reshape(clean,(1,np.shape(clean)[0],1))

        index += 1
        if index == len(noisy_list):
            index = 0
            if shuffle == "True":
                random.shuffle(noisy_list)

        yield noisy, clean

data_n = sys.argv[1]

if data_n == 'TIMIT_SE':

    ##############################################################3
    noisy_list = get_filepaths("/mnt/md2/user_chengyu/Corpus/TIMIT_SE/Train/Noisy", '.wav')
    clean_path = "/mnt/md2/user_chengyu/Corpus/TIMIT_SE/Train/Clean/"
    vad_path = "/mnt/md2/user_khhung/vad/vadfile/"
    random.shuffle(noisy_list)
    idx = int(len(noisy_list)*0.95)
    Train_list = noisy_list[0:idx]
    Num_traindata = len(Train_list)
    Valid_list = noisy_list[idx:]

    steps_per_epoch = (Num_traindata)//batch_size

    Test_lists = get_filepaths("/mnt/md2/user_chengyu/Corpus/TIMIT_SE/Test/Noisy", '.wav')
    Num_testdata=len(Valid_list)
    ####Aurora2####
    noisy_list_a2 = get_filepaths("/mnt/md2/user_khhung/vad/Aurora2/wavfile/train/noisy", '.wav')
    clean_path_a2 = "/mnt/md2/user_khhung/vad/Aurora2/wavfile/train/clean/"
    vad_path_a2 = "/mnt/md2/user_khhung/vad/Aurora2/fvadfile/train/"
    random.shuffle(noisy_list_a2)
    idx = int(len(noisy_list_a2)*0.95)
    Train_list_a2 = noisy_list_a2[0:idx]
    Num_traindata_a2 = len(Train_list_a2)
    Valid_list_a2 = noisy_list_a2[idx:]

    steps_per_epoch_a2 = (Num_traindata_a2)//batch_size
    Num_testdata_a2=len(Valid_list_a2)

    ##############################################################3

elif data_n == 'Aurora2':

    ##############################################################3
    noisy_list_a2 = get_filepaths("/mnt/md2/user_khhung/vad/Aurora2/wavfile/train/noisy", '.wav')
    clean_path_a2 = "/mnt/md2/user_khhung/vad/Aurora2/wavfile/train/clean/"
    vad_path_a2 = "/mnt/md2/user_khhung/vad/Aurora2/fvadfile/train/"
    random.shuffle(noisy_list_a2)
    idx = int(len(noisy_list_a2)*0.95)
    Train_list_a2 = noisy_list_a2[0:idx]
    Num_traindata_a2 = len(Train_list_a2)
    Valid_list_a2 = noisy_list_a2[idx:]

    steps_per_epoch_a2 = (Num_traindata_a2)//batch_size

    Test_lists_a = get_filepaths("/mnt/md2/user_khhung/vad/Aurora2/wavfile/testa/noisy", '.wav')
    Test_lists_b = get_filepaths("/mnt/md2/user_khhung/vad/Aurora2/wavfile/testb/noisy", '.wav')
    Test_lists_c = get_filepaths("/mnt/md2/user_khhung/vad/Aurora2/wavfile/testc/noisy", '.wav')
    Test_lists = []
    Test_lists.extend(Test_lists_a)
    Test_lists.extend(Test_lists_b)
    Test_lists.extend(Test_lists_c)
    #pdb.set_trace()
    Num_testdata_a2=len(Valid_list_a2)

    ##############################################################3

elif data_n == 'TMHINT':

    ##############################################################3
    noisy_list_a2 = get_filepaths("/mnt/md2/user_chengyu/Corpus/TMHINT/Training/Noisy", '.wav')
    clean_path_a2 = "/mnt/md2/user_chengyu/Corpus/TMHINT/Training/Clean/"
    vad_path_a2 = "/mnt/md2/user_chengyu/Corpus/TMHINT/vadfile/TMHINT/Training/Clean/"
    random.shuffle(noisy_list_a2)
    noisy_list_a2 = noisy_list_a2[0:30000]
    idx = int(len(noisy_list_a2)*0.95)
    Train_list_a2 = noisy_list_a2[0:idx]
    Num_traindata_a2 = len(Train_list_a2)
    Valid_list_a2 = noisy_list_a2[idx:]

    steps_per_epoch_a2 = (Num_traindata_a2)//batch_size
    #pdb.set_trace()
    Num_testdata_a2=len(Valid_list_a2)

    ##############################################################3

start_time = time.time()
name_vad = sys.argv[2]
print('vad model buiding...')
model_vad=locals()[name_vad]()
model_vad.summary()

name_se = sys.argv[3]
print('se model buiding...')
se=locals()[name_se]()
se.summary()

save_vad = sys.argv[4]
save_se = sys.argv[5]

model_se = Sequential()
enc = Model(model_vad.input, model_vad.layers[5].output)
model_se.add(enc)
model_se.add(se)

vad_train_loss = []
vad_valid_loss = []
se_train_loss = []
se_valid_loss = []
vvl=0
svl=0

for epoch in range(iteration):
    
    print('epoch:', epoch)

    if epoch>0:
        vvl=min(vad_valid_loss)
    else:
        vvl=np.Inf
    
    model_vad.compile(loss='categorical_crossentropy', optimizer='adam')
    with open('./models/'+save_vad+'.json','w') as f: #save the model
        f.write(model_vad.to_json())
    checkpointer = ModelCheckpoint_iter(filepath='./models/'+save_vad+'.hdf5', verbose=1, save_best_only=True, mode='min', val=vvl)
    print('VAD training...')

    g1 = vad_data_generator(Train_list_a2, clean_path_a2r, vad_path_a2, shuffle = "True")
    g2 = vad_valid_generator(Valid_list_a2, clean_path_a2, vad_path_a2, shuffle = "False")

    vad_hist=model_vad.fit_generator(g1,
                         samples_per_epoch=Num_traindata_a2,
                         nb_epoch=vad_iteration,
                         verbose=1,
                         validation_data=g2,
                         nb_val_samples=Num_testdata_a2,
                         max_q_size=20,
                         nb_worker=3,
                         pickle_safe=True,
                         callbacks=[checkpointer]
                         )
    vad_valid_loss.extend(vad_hist.history['val_loss'])
    vad_train_loss.extend(vad_hist.history['loss'])
    
    if epoch>0:
        svl=min(se_valid_loss)
    else:
        svl=np.Inf   
    
    model_se.compile(loss='mse', optimizer='adam')
    with open('./models/'+save_se+'.json','w') as f: #save the model
        f.write(model_se.to_json())
    checkpointer = ModelCheckpoint_iter(filepath='./models/'+save_se+'.hdf5', verbose=1, save_best_only=True, mode='min', val=svl)
    print('SE training...')

    if data_n == 'TIMIT_SE':
        g3 = se_TIMIT_data_generator(Train_list, clean_path, vad_path, shuffle = "True")
        g4 = se_TIMIT_valid_generator(Valid_list, clean_path, vad_path, shuffle = "False")
    
    elif data_n == 'TMHINT':
        g3 = se_TIMIT_data_generator(Train_list, clean_path, vad_path, shuffle = "True")
        g4 = se_TIMIT_valid_generator(Valid_list, clean_path, vad_path, shuffle = "False")

    elif data_n == 'Aurora2':
        g3 = se_data_generator(Train_list, clean_path, vad_path, shuffle = "True")
        g4 = se_valid_generator(Valid_list, clean_path, vad_path, shuffle = "False")

    se_hist=model_se.fit_generator(g3,
                         samples_per_epoch=Num_traindata,
                         nb_epoch=se_iteration,
                         verbose=1,
                         validation_data=g4,
                         nb_val_samples=Num_testdata,
                         max_q_size=20,
                         nb_worker=3,
                         pickle_safe=True,
                         callbacks=[checkpointer]
                         )
    se_valid_loss.extend(se_hist.history['val_loss'])
    se_train_loss.extend(se_hist.history['loss'])
    
        
    
        
        
end_time = time.time()

# plotting the vad learning curve
vad_TrainERR=vad_train_loss
vad_ValidERR=vad_valid_loss
print('@%f, Minimun error:%f, at iteration: %i' % (vad_valid_loss[(iteration*vad_iteration)-1], np.min(np.asarray(vad_ValidERR)),np.argmin(np.asarray(vad_ValidERR))+1))
print('drawing the training process...')
plt.figure(2)
plt.plot(range(1,(iteration*vad_iteration)+1),vad_TrainERR,'b',label='TrainERR')
plt.plot(range(1,(iteration*vad_iteration)+1),vad_ValidERR,'r',label='ValidERR')
plt.xlim([1,(iteration*vad_iteration)])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error')
plt.grid(True)
plt.show()
plt.savefig(save_vad+'.png', dpi=150)
plt.close()

# plotting the vad learning curve
se_TrainERR=se_train_loss
se_ValidERR=se_valid_loss
print('@%f, Minimun error:%f, at iteration: %i' % (se_valid_loss[(iteration*se_iteration)-1], np.min(np.asarray(se_ValidERR)),np.argmin(np.asarray(se_ValidERR))+1))
print('drawing the training process...')
plt.figure(2)
plt.plot(range(1,(iteration*se_iteration)+1),se_TrainERR,'b',label='TrainERR')
plt.plot(range(1,(iteration*se_iteration)+1),se_ValidERR,'r',label='ValidERR')
plt.xlim([1,(iteration*se_iteration)])
plt.legend()
plt.xlabel('epoch')
plt.ylabel('error')
plt.grid(True)
plt.show()
plt.savefig(save_se+'.png', dpi=150)
plt.close()

print ('The code for this file ran for %.2fm' % ((end_time - start_time) / 60.))









