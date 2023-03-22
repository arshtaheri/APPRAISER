from random import shuffle
from glob import glob
import numpy as np
from PIL import Image
from conv import *
from maxpool import MaxPool2
from fc import *
from softmax import Softmax
import re

def forward2x2(image, label):
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



def forward2x2_quantized(image, label, file_name):
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
    out = out.reshape(1, 256)                       # 16x16 -> 256
    #print(out)
    #print(softmax.input_quantization(out))
    out = softmax.forward_quantized(out, file_name) # 64    -> 1x2
    acc = 1 if np.argmax(out) == label else 0

    return acc


def train2x2(im, label, eta):
    #Forward
    out, loss, acc = forward2x2(im, label)
    
    # Calculate initial gradient
    gradient = np.zeros(2)
    gradient[label] = -1 / out[label]
    
    # Backprop
    gradient = softmax.backprop(gradient, eta)

    return loss, acc

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

train_files = glob(r"/home/mahdi/workspace/courses/approximation-instead of fault injection/fault-injection-MLP/quantized/human_animal/train/*")
shuffle(train_files)

#test_files = glob(r"C:\Users\mojtaba\Dropbox\My shared files\cnn\mixed-dataset\test\*")
#shuffle(test_files)

#train and test parameters
train_imgs = len(train_files)
#train_imgs = 200
step_no = train_imgs // 10
epochs = 3
#test_no = len(test_files)
#test_no = 40
learning_rate = 0.05



for epoch in range(epochs):
    print('----- Epoch %d -----' % (epoch + 1))
    loss = 0
    num_correct = 0
    i = 0
    for file in train_files[0 : train_imgs]:
        label = -1
        if file.find("cat") != -1 or file.find("dog") != -1:
            label = 0
        else:
            label = 1

        img = np.array(Image.open(file))
        if i % step_no == step_no - 1:
            print(
                '[Step %d] Past %d steps: Average Loss %.3f | Accuracy: %.2f' %
                (i + 1, step_no, loss / step_no, num_correct / step_no))
            loss = 0
            num_correct = 0
        l, acc = train2x2(img, label, learning_rate)
        loss += l
        num_correct += acc
        i += 1

        
softmax.save_weights(r"/home/mahdi/workspace/courses/approximation-instead of fault injection/fault-injection-MLP/quantized/newly_trained_network/weights.txt")
softmax.save_biases(r"/home/mahdi/workspace/courses/approximation-instead of fault injection/fault-injection-MLP/quantized/newly_trained_network/biases.txt")

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


print('----- Evaluation 2 -----')
print('--- QUANTIZED ---')
"""
softmax.weights_quantization()
softmax.biases_quantization()
softmax.save_weights(r"/home/mahdi/workspace/courses/approximation-instead of fault injection/fault-injection-MLP/quantized/newly_trained_network/weights_q.txt")
softmax.save_biases(r"/home/mahdi/workspace/courses/approximation-instead of fault injection/fault-injection-MLP/quantized/newly_trained_network/biases_q.txt")
print("Quantization done")

"""
loss_q = 0
num_correct_q = 0
file_counter = 0
for file in test_files[0:test_no]:
    label = -1
    if file.find("cat") != -1 or file.find("dog") != -1:
        label = 0
    else:
        label = 1
    img = np.array(Image.open(file))
    acc_q = forward2x2_quantized(img, label, file[-12:-4])
    file_counter += 1
    num_correct_q += acc_q
    """
#print('Quantized without softmax Test Accuracy:', (num_correct_q / test_no) * 100) 
