import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from conv import *
import os
from maxpool import MaxPool2
from fc import *
from softmax import Softmax
from glob import glob
import numpy as np
from PIL import Image
from random import shuffle


def forward2x2(image, label, file, neurons_layer_no):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            s = ""
            s += '{0:08b}'.format(int(image[i][j]))[:4]
            image[i][j] = int(s, 2)
            
    out = conv1.forward(image)                  #((image / 255) - 0.5)    # 256x256 -> 128x128
    out = pool1.forward(out)                    # 128x128 -> 64x64
    a, b, c = out.shape
    out = out.reshape(c, a, b)
    tmp_out = np.zeros((conv1.num_filters * conv2.num_filters, int(a/2), int(b/2)))
    index = 0
    index_cnt = conv2.num_filters
    for fi_map in range(c):                     # 64x64 => 32x32
        tmp_out[index:index + index_cnt][:][:] = conv2.forward(out[fi_map]).reshape((index_cnt, int(a/2), int(b/2)))
        index += index_cnt
        out = tmp_out.reshape((int(a/2), int(b/2), conv1.num_filters * conv2.num_filters))

    out = pool2.forward(out)                    # 32x32 -> 16x16
    out = out.reshape(1, 256)
    out = softmax.forward(out)                  # 256 -> 1x2
  
    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

def forward2x2_quantized_approx(image, label, file_name, neurons_layer_no):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            s = ""
            s += '{0:08b}'.format(int(image[i][j]))[:4]
            image[i][j] = int(s, 2)

    out, A = conv1.forward_approx(image)                      # 256x256 -> 128x128
    out1 = out.reshape(1, 16384)
    #mina = max(map(max, abs(np.array(A))))
    #maxa = max(map(max, np.array(A)))
    #B.append(mina)
    #C.append(maxa)


    # set the path to the output folder on your desktop
    output_folder = os.path.join(os.path.expanduser('~'), 'Desktop', 'mask_approx')

    # create the folder if it doesn't already exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # set the full file path for 'mask_out_conv1.txt' in the output folder
    output_file = os.path.join(output_folder, 'conv1_aprx.txt')
    with open(output_file, 'a') as f:
        for item in out1:
            s = ''
            for i in item:
                s += str(i)+' '
            f.write("%s\n" % s)
    out = pool1.forward(out)                        # 128x128 -> 64x64
    out1 = out.reshape(1, 4096)
    output_file = os.path.join(output_folder, 'pool1_aprx.txt')
    with open(output_file, 'a') as f:
        for item in out1:
            s = ''
            for i in item:
                s += str(i)+' '
            f.write("%s\n" % s)
    a, b, c = out.shape
    out = out.reshape(c, a, b)
    tmp_out = np.zeros((conv1.num_filters * conv2.num_filters, int(a/2), int(b/2)))
    index = 0
    index_cnt = conv2.num_filters
    for fi_map in range(c):
        tmp_out[index:index + index_cnt][:][:] = conv2.forward(out[fi_map]).reshape((index_cnt, int(a/2), int(b/2)))
        index += index_cnt
        out = tmp_out.reshape((int(a/2), int(b/2), conv1.num_filters * conv2.num_filters))
    a, b, c = out.shape                             # 64x64 -> 32x32
    out1 = out.reshape(1, 1024)
    output_file = os.path.join(output_folder, 'conv2_aprx.txt')
    with open(output_file, 'a') as f:
        for item in out1:
            s = ''
            for i in item:
                s += str(i)+' '
            f.write("%s\n" % s)
    out = pool2.forward(out)                        # 32x32 -> 16x16
    out1 = out.reshape(1, 256)
    output_file = os.path.join(output_folder, 'pool2_aprx.txt')
    with open(output_file, 'a') as f:
        for item in out1:
            s = ''
            for i in item:
                s += str(i)+' '
            f.write("%s\n" % s)
    #out = out.reshape(1, 4)                       # 16x16 -> 256
    out = out.reshape(1, 256)
    output_file = os.path.join(output_folder, 'FC_aprx.txt')
    with open(output_file, 'a') as f:
        for item in out:
            s = ''
            for i in item:
                s += str(i)+' '
            f.write("%s\n" % s)
    out = softmax.forward_quantized(out, file_name) # 64    -> 1x2
    acc = 1 if np.argmax(out) == label else 0
    #print(out)
    #if acc == 0:
    #    print("res: " + file[-20:])
    #    print(out)
    #    print("\n")
    return acc, A


#### Use this function for fault injection **** z in the conv class for fault_forward is the number of iterations for injecting faults ********* there is another parameter in the conv forward_fault for the simultanious number of faults to be injected
def forward2x2_quantized(image, label, file_name, z):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            s = ""
            s += '{0:08b}'.format(int(image[i][j]))[:4]
            image[i][j] = int(s, 2)

    out = conv1.forward_fault(image)                      # 256x256 -> 128x128
    out1 = out.reshape(1, 16384)

    '''with open('conv2_one_sixth_approx_output_conv1.txt', 'a') as f:
        for item in out1:
            s = ''
            for i in item:
                s += str(i)+' '
            f.write("%s\n" % s)'''
    out = pool1.forward(out)                        # 128x128 -> 64x64
    out1 = out.reshape(1, 4096)
    '''with open('conv2_one_sixth_approx_output_pool1.txt', 'a') as f:
        for item in out1:
            s = ''
            for i in item:
                s += str(i)+' '
            f.write("%s\n" % s)'''
    a, b, c = out.shape
    out = out.reshape(c, a, b)
    tmp_out = np.zeros((conv1.num_filters * conv2.num_filters, int(a/2), int(b/2)))
    index = 0 
    index_cnt = conv2.num_filters
    for fi_map in range(c):
        tmp_out[index:index + index_cnt][:][:] = conv2.forward(out[fi_map]).reshape((index_cnt, int(a/2), int(b/2)))
        index += index_cnt
        out = tmp_out.reshape((int(a/2), int(b/2), conv1.num_filters * conv2.num_filters))
    a, b, c = out.shape                             # 64x64 -> 32x32
    out1 = out.reshape(1, 1024)
    '''with open('conv2_one_eighth_approx_output_conv2.txt', 'a') as f:
        for item in out1:
            s = ''
            for i in item:
                s += str(i)+' '
            f.write("%s\n" % s)'''
    out = pool2.forward(out)                        # 32x32 -> 16x16
    out1 = out.reshape(1, 256)
    '''with open('conv2_one_eighth_approx_output_pool2.txt', 'a') as f:
        for item in out1:
            s = ''
            for i in item:
                s += str(i)+' '
            f.write("%s\n" % s)'''
    out = out.reshape(1, 256)       # 16x16 -> 256
    out = softmax.forward_quantized(out, file_name) # 64    -> 1x2
    out1 = out.reshape(1, 2)
    '''with open('conv2_one_eighth_approx_output_FC.txt', 'a') as f:
        for item in out1:
            s = ''
            for i in item:
                s += str(i)+' '
            f.write("%s\n" % s)'''
    acc = 1 if np.argmax(out) == label else 0
    #print(out)
    #if acc == 0:
    #    print("res: " + file[-20:])
    #    print(out)
    #    print("\n")
    return acc

def fault_generation(in_val):
    in_val_bin = bin(int(in_val)).replace('0b','') #''.join('{:0>8b}'.format(c) for c in struct.pack('!f', in_val))
    a = 0
    if in_val_bin[0] == '-' :
        in_val_bin = in_val_bin[1:]
        a = 1
    x1 = in_val_bin[::-1] #this reverses an array
    while len(x1) < 6:
      x1 += '0'
    in_val_bin1 = x1[::-1]
    fault_loc = random.randint(0, 5)
    new_val_bin = in_val_bin1[:5-fault_loc]
    if in_val_bin1[5 - fault_loc] == '0':
        new_val_bin += '1'
    else:
        new_val_bin += '0'
    new_val_bin += in_val_bin1[5-fault_loc+1:]
    if a == 1:
        new_val = -int(new_val_bin, 2)
    else:
        new_val = int(new_val_bin, 2)
    return new_val, fault_loc

def fault_injection_forward(image, label, file_name, neurons_layer_no):
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            s = ""
            s += '{0:08b}'.format(int(image[i][j]))[:4]
            image[i][j] = int(s, 2)
    out = conv1.forward(image)                      # 256x256 -> 128x128
    out = pool1.forward(out)                        # 128x128 -> 64x64
    a, b, c = out.shape
    out = out.reshape(c, a, b)
    tmp_out = np.zeros((conv1.num_filters * conv2.num_filters, int(a/2), int(b/2)))
    index = 0
    index_cnt = conv2.num_filters
    for fi_map in range(c):
        tmp_out[index:index + index_cnt][:][:] = conv2.forward(out[fi_map]).reshape((index_cnt, int(a/2), int(b/2)))
        index += index_cnt
        out = tmp_out.reshape((int(a/2), int(b/2), conv1.num_filters * conv2.num_filters))
    a, b, c = out.shape                             # 64x64 -> 32x32

    out = pool2.forward(out)                        # 32x32 -> 16x16
    #out = out.reshape(1, 4)                       # 16x16 -> 256
    out = out.reshape(1, 256)
    #out2 = softmax.forward_quantized(out, file_name) # 64    -> 1x2
    #golden_detected_label = np.argmax(out2)
    #acc_golden = 1 if golden_detected_label == label else 0
    for i in range(0):
        rand_neuron = random.randint(0, neurons_layer_no - 1)     #number of neurons
        neur_val = out[0][rand_neuron]
        faulty_val, fault_loc = fault_generation(neur_val)
        while faulty_val == 'nan':
            faulty_val, fault_loc = fault_generation(neur_val)
        out[0][rand_neuron] = faulty_val
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            act = out[i][j]
            if act > 150:
                act = 150
            elif act < -8:
                act = -8
            OldRange = 158
            NewRange = 15
            OldMin = -8
            NewMin = 0 #-8
            NewValue = (((act - OldMin) * NewRange) / OldRange) + NewMin
            out[i][j] = round(NewValue)
    out1 = softmax.fault_forward(out, file_name)
    faulty_detected_label = np.argmax(out1)
    acc = 1 if faulty_detected_label == label else 0
    #fault_dif = faulty_val - neur_val
    return acc, faulty_detected_label #, fault_dif, rand_neuron, fault_loc



#network description 
filter_no1 = 1
filter_no2 = 1
funnel_out = 16 * 16
class_no = 2
conv1 = Conv2x2(filter_no1, 1)
pool1 = MaxPool2()
conv2 = Conv2x2(filter_no2, 2)
pool2 = MaxPool2()
softmax = Softmax(funnel_out, class_no)
softmax.read_weights(r"./weights_q_last_91.1.txt")
softmax.read_biases(r"./biases_q_last_93.3.txt")

print("weights and biases are loaded")

test_files = glob(r"./human_animal/test/*")
#shuffle(test_files)
test_no = len(test_files)
B = [0 for l in range(test_no)]
C = [0 for l in range(test_no)]
#test_no = 40
"""
print('----- Evaluation 1 -----')
print('--- FULL PRECISION ---')
loss = 0
num_correct = 0
for file in test_files[0:test_no]:
    label = -1
    if file.find("cat") != -1 or file.find("dog") != -1:
        label = 0
    else:
        label = 1
    img = np.array(Image.open(file))
    _, l, acc = forward2x2(img, label)
    loss += l
    num_correct += acc
    
print('Test Loss:', loss / test_no)
print('Full Precision Test Accuracy:', (num_correct / test_no) * 100)
"""

#print('----- Evaluation 2 -----')
#print('--- QUANTIZED ---')

#softmax.weights_quantization()
#softmax.biases_quantization()
#softmax.save_weights(r"C:\Users\Admin\Dropbox\My shared files\cnn\cnn-catdog\weights_q.txt")
#softmax.save_biases(r"C:\Users\Admin\Dropbox\My shared files\cnn\cnn-catdog\biases_q.txt")
#print("Quantization done")

loss_q = 0
num_correct_q = 0
file_counter = 0
human_mis = 0
animal_mis = 0
pic_no = 1
neurons_layer_no = 256
for file in test_files[0:test_no]:
    label = -1
    if file.find("cat.") != -1 or file.find("dog.") != -1:
        label = 0
    else:
        label = 1
    #label = 0
    img = np.array(Image.open(file))
    #print("name = " + file[-6:])
    #print("label: " + str(label))
    acc_q, A = forward2x2_quantized_approx(img, label, file, neurons_layer_no)
    z = 0
    if acc_q == 0 and label == 1:
        human_mis += 1
        #image = mpimg.imread(file)
        #plt.imshow(image)
        #plt.show()

    elif acc_q == 0 and label == 0:
        animal_mis += 1
    file_counter += 1
    num_correct_q += acc_q
    pic_no += 1
#print('A_min=', min(map(min, np.array(B))))
#print('A_max=', max(map(max, np.array(C))))
print('Quantized without softmax Test Accuracy:', (num_correct_q / test_no) * 100)
print("human misclassed no: " + str(human_mis))
print("animal misclassed no: " + str(animal_mis))
print('pic number='+ str(pic_no))

'''loss_q = 0
num_correct_q = 0
file_counter = 0
human_mis = 0
animal_mis = 0
pic_no = 1
for file in test_files[0:test_no]:
    label = -1
    if file.find("cat.") != -1 or file.find("dog.") != -1:
        label = 0
    else:
        label = 1
    #label = 0
    img = np.array(Image.open(file))
    #print("name = " + file[-6:])
    #print("label: " + str(label))
    acc_q = forward2x2_quantized_approx(img, label, file)
    if acc_q == 0 and label == 1:
        human_mis += 1
        #image = mpimg.imread(file)
        #plt.imshow(image)
        #plt.show()

    elif acc_q == 0 and label == 0:
        animal_mis += 1
    file_counter += 1
    num_correct_q += acc_q
    pic_no += 1

print('Approximated Quantized without softmax Test Accuracy:', (num_correct_q / test_no) * 100)
print("human misclassed no: " + str(human_mis))
print("animal misclassed no: " + str(animal_mis))
print('pic number='+ str(pic_no))'''

'''loss_q = 0
num_correct_q = 0
file_counter = 0
human_mis = 0
animal_mis = 0
pic_no = 1
for file in test_files[0:test_no]:
    label = -1
    if file.find("cat.") != -1 or file.find("dog.") != -1:
        label = 0
    else:
        label = 1
    #label = 0
    img = np.array(Image.open(file))
    #print("name = " + file[-6:])
    #print("label: " + str(label))
    acc_q, faulty_detected_label, fault_dif, rand_neuron, fault_loc = fault_injection_forward(img, label, file, 256)
    if acc_q == 0 and label == 1:
        human_mis += 1
        #image = mpimg.imread(file)
        #plt.imshow(image)
        #plt.show()

    elif acc_q == 0 and label == 0:
        animal_mis += 1
    file_counter += 1
    num_correct_q += acc_q
    pic_no += 1

print('fault injection Quantized without softmax Test Accuracy:', (num_correct_q / test_no) * 100)
print("human misclassed no: " + str(human_mis))
print("animal misclassed no: " + str(animal_mis))
print('pic number='+ str(pic_no))'''
