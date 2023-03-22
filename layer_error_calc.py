import numpy as np
from ast import literal_eval
from numpy import savetxt

from sklearn.metrics import mean_squared_error

def checkpoint(val):
    print(val)
    exit()

class LayerErrorCalc:

    '''def read_file_contents(self, file_path):

        file = open(file_path, mode='r', encoding='utf-8')
        lines = ''.join(file.readlines())
        lines = [i.strip()[1:] for i in lines.split('\n')]
        #lines = [[literal_eval(i.strip()) for i in line.split()] for line in lines][:-1]
        return np.array(lines)'''
    def subtract(self):
        #array size for conv1: 16384, pool1: 4096,  conv2: 1024, pool2: 256, fc: 2
        array1 = []
        array_1 = [[0 for l in range(4096)] for m in range(450)]
        array2 = []
        array_2 = [[0 for l in range(4096)] for m in range(450)]
        z = 0
        p = 0
        sub_appr_exact = [[0 for l in range(4096)] for m in range(450)]
        MSE_con_out = []
        with open("exact_output_pool1.txt", "r") as f1:
            with open("pool1_aprx.txt", "r") as f2:
               for line in f1:
                    stripped_line = line.replace(".0", "")
                    stripped_lin = stripped_line.replace(".", "")
                    b = stripped_lin.replace("]", "")
                    c = b.replace("[", "")
                    a = c.split()
                    for i in range(len(a)):
                        #print(i, len(a), z)
                        array_1[z][i] = int(a[i])
                        #print(a[-1])
                    z += 1
                    f = np.array(array_1)
                    b = f.astype(np.int64)
                    array1 = np.array(array_1)
               for line in f2:
                    stripped_line = line.replace(".0", "")
                    stripped_lin = stripped_line.replace(".", "")
                    b = stripped_lin.replace("]", "")
                    c = b.replace("[", "")
                    a = c.split()
                    #m = stripped_line.strip()

                    for i in range(len(a)):
                        print(i, len(a), z)
                        array_2[p][i] = int(a[i])
                    p += 1
                    f = np.array(array_2)
                    b = f.astype(np.int64)
                    array2 = np.array(array_2)
               for i in range(len(array1)):
                   for j in range(len(array1[0][:])):
                       sub_appr_exact[i][j] = (array1[i][j] - array2[i][j])
               average_error = []
               for i in range(len(sub_appr_exact[0][:])):
                   sum = 0
                   for j in range(len(sub_appr_exact)):
                        sum += sub_appr_exact[j][i]
                   average = sum / 450
                   average_error.append(average)
               print('MSE neuron1_Approx_out', average_error)
               #all_units_approx
               with open('pool1_average_error_exact_1_aprx.txt', 'a') as f:
                    for item in average_error:
                        f.write("%s\n" % item)
               ##########These lines are for CNV
               a = np.array(array2)
               b = np.array(array2)
        return np.array(sub_appr_exact), average_error  #, MSE_neuron2
    def Pool1_bit_error_calculation(self):
        array1 = []
        array_1 = [[0 for l in range(4096)] for m in range(450)]
        array2 = []
        array_2 = [[0 for l in range(4096)] for m in range(450)]
        z = 0
        p = 0
        with open("exact_output_pool1.txt", "r") as f1:
            with open("pool1_aprx.txt", "r") as f2:
               for line in f1:
                    stripped_line = line.replace(".0", "")
                    a = stripped_line.split()
                    #m = stripped_line.strip()
                    #b = a.replace("]", "")
                    #c = b.replace("[", "")
                    for i in range(len(a)):
                        array_1[z][i] = int(a[i])
                        #print(a[-1])
                    z += 1
                    #f = np.array(array_1)
                    #b = f.astype(np.int64)
                    array1 = np.array(array_1)
               for line in f2:
                    stripped_line = line.replace(".0", "")
                    a = stripped_line.split()
                    #m = stripped_line.strip()
                    #b = a.replace("]", "")
                    #c = b.replace("[", "")
                    for i in range(len(a)):
                        array_2[p][i] = int(a[i])
                    #f = np.array(array_2)
                    #b = f.astype(np.int64)
                    array2 = np.array(array_2)
               sum = 0
               for i in range(len(array1)):
                   for j in range(len(array1[0][:])):
                      in_val_bin = bin(int(array1[i][j])).replace('0b','') #''.join('{:0>8b}'.format(c) for c in struct.pack('!f', in_val))
                      in_val_bin1 = bin(int(array2[i][j])).replace('0b','') #''.join('{:0>8b}'.format(c) for c in struct.pack('!f', in_val))
                      if in_val_bin[0] == '-' :
                          in_val_bin = in_val_bin[1:]
                          x1 = in_val_bin[::-1] #this reverses an array
                          x1 += '1'
                      else:
                          x1 = in_val_bin[::-1]
                      while len(x1) < 100:
                          x1 += '0'
                      in_bin = x1[::-1]
                      if in_val_bin1[0] == '-' :
                          in_val_bin1 = in_val_bin1[1:]
                          x1 = in_val_bin1[::-1] #this reverses an array
                          x1 += '1'
                      else:
                          x1 = in_val_bin1[::-1]
                      while len(x1) < 100:
                          x1 += '0'
                      in_bin1 = x1[::-1]
                      for l in range(len(in_bin1)):
                          if in_bin1[l] != in_bin[l]:
                            sum += 1
               average_number_of_bit_flips = sum / 450
               print(average_number_of_bit_flips)

    '''def subtract1(self):
        array1 = []
        array2 = []
        with open("exact.txt", "r") as f1:
            with open("FI_100_out1.txt", "r") as f2:
               for line in f1:
                    stripped_line = line.strip()
                    array1.append(stripped_line)
               for line in f2:
                    stripped_line = line.strip()
                    array2.append(stripped_line)
               sub_fault_exact = []
               list1_exact = []
               list2_exact = []
               list1_fault = []
               list2_fault = []
               for i in range(450):
                   if i % 2 == 0:
                       list1_exact.append(array1[i])
                   else:
                       list2_exact.append(array1[i])

               for i in range(450):
                   if i % 2 == 0:
                        list1_fault.append(array2[i])
                   else:
                        list2_fault.append(array2[i])
               a = np.array(list1_fault)
               b = np.array(list1_exact)
               c = a.astype(np.int64)
               d = b.astype(np.int64)
               e = np.array(list2_fault)
               f = np.array(list2_exact)
               g = e.astype(np.int64)
               h = f.astype(np.int64)
               MSE_neuron1 = np.square(np.subtract(c, d)).mean()
               MSE_neuron2 = np.square(np.subtract(g, h)).mean()
               print('MSE neuron1_fault_out', MSE_neuron1)
               print('MSE neuron2_fault_out', MSE_neuron2)
               for i in range(len(array1)):
                   sub_fault_exact.append(int(array1[i]) - int(array2[i]))
        return np.array(sub_fault_exact), MSE_neuron1, MSE_neuron2'''
    '''def bit_error_calculation(self):
        array1 = []
        array2 = []
        with open("exact.txt", "r") as f1:
            with open("appr_out.txt", "r") as f2:
               for line in f1:
                    stripped_line = line.strip()
                    array1.append(stripped_line)
               for line in f2:
                    stripped_line = line.strip()
                    array2.append(stripped_line)
               #array1 = np.asarray(f1.read().split('\n'))
               #array2 = np.asarray(f2.read().split('\n'))
               sub_appr_exact = []
               list1_exact = []
               list2_exact = []
               list1_appr = []
               list2_appr = []
               for i in range(450):
                   if i % 2 == 0:
                       list1_exact.append(array1[i])
                   else:
                       list2_exact.append(array1[i])

               for i in range(450):
                   if i % 2 == 0:
                        list1_appr.append(array2[i])
                   else:
                        list2_appr.append(array2[i])
               a = np.array(list1_appr)
               b = np.array(list1_exact)
               c = a.astype(np.int64)
               d = b.astype(np.int64)
               e = np.array(list2_appr)
               f = np.array(list2_exact)
               g = e.astype(np.int64)
               h = f.astype(np.int64)
               for i in range(c.shape[0]):
                    act = c[i]
                    OldRange = 2112
                    NewRange = 512
                    OldMin = 159
                    NewMin = 0 #-8
                    NewValue = (((act - OldMin) * NewRange) / OldRange) + NewMin
                    z = round(NewValue)
                    c[i] = bin(int(z)).replace('0b','')
               for i in range(d.shape[0]):
                    act = d[i]
                    OldRange = 2294
                    NewRange = 512
                    OldMin = 2113
                    NewMin = 0 #-8
                    NewValue = (((act - OldMin) * NewRange) / OldRange) + NewMin
                    z = round(NewValue)
                    d[i] = bin(int(z)).replace('0b','')
               for i in range(g.shape[0]):
                    act = g[i]
                    OldRange = 2368
                    NewRange = 512
                    OldMin = 144
                    NewMin = 0 #-8
                    NewValue = (((act - OldMin) * NewRange) / OldRange) + NewMin
                    z = round(NewValue)
                    g[i] = bin(int(z)).replace('0b','')
               for i in range(h.shape[0]):
                    act = h[i]
                    OldRange = 2386
                    NewRange = 512
                    OldMin = 2065
                    NewMin = 0 #-8
                    NewValue = (((act - OldMin) * NewRange) / OldRange) + NewMin
                    z = round(NewValue)
                    h[i] = bin(int(z)).replace('0b','')
               first_neuron_bit_flip = 0
               second_neuron_bit_flip = 0
               for i in range(g.shape[0]):
                    x = str(c[i])[::-1] #this reverses an array
                    x1 = str(d[i])[::-1] #this reverses an array
                    while len(x) < len(x1):
                        x += '0'
                    a_1 = int(x[::-1], 2)
                    while len(x1) < len(x):
                        x1 += '0'
                    a_2 = int(x1[::-1], 2)
                    x2 = str(g[i])[::-1] #this reverses an array
                    x3 = str(h[i])[::-1] #this reverses an array
                    while len(x2) < len(x3):
                        x2 += '0'
                    a_3 = int(x2[::-1], 2)
                    while len(x3) < len(x2):
                        x3 += '0'
                    a_4 = int(x3[::-1], 2)

                    compare = str(bin(a_1 ^ a_2).replace('0b',''))
                    compare1 = str(bin(a_3 ^ a_4).replace('0b',''))
                    for k in compare:
                        if k == '1':
                            first_neuron_bit_flip += 1
                    for k in compare1:
                        if k == '1':
                            second_neuron_bit_flip += 1
               print('first_neuron_bit_flip', first_neuron_bit_flip)
               print('second_neuron_bit_flip', second_neuron_bit_flip)'''


    '''def calculate_mse(self, sub_appr_exact, sub_fault_exact, layer_name):
        #exact, appr_out, FI_10000_out1 = exact, appr_out, fault
        #sub_appr_exact = np.subtract(appr_out, exact)
        #sub_fault_exact = np.subtract(FI_10000_out1, exact)
        square1 = np.square(sub_appr_exact)
        square2 = np.square(sub_fault_exact)
        list1_appr = []
        list2_appr = []
        for i in range(450):
            list1_appr.append(square1[i])
            i += 2
        for i in range(450):
            list2_appr.append(square1[i])
            i += 1
        list1_fault = []
        list2_fault = []
        for i in range(450):
            list1_fault.append(square2[i])
            i += 2
        for i in range(450):
            list2_fault.append(square2[i])
            i += 1
        neuron1_approx = np.sum(np.array(list1_appr))
        neuron2_approx = np.sum(np.array(list2_appr))
        neuron1_fault = np.sum(np.array(list1_fault))
        neuron2_fault = np.sum(np.array(list2_fault))
        #appr_mse = np.sum(np.square(sub_appr_exact), axis=0) / sub_appr_exact.shape[0]
        #fault_mse = np.sum(np.square(sub_fault_exact), axis=0) / sub_fault_exact.shape[0]
        #savetxt('./exports/fault_mse_' + layer_name + '.csv', fault_mse, delimiter=',')
        #savetxt('./exports/appr_mse_' + layer_name + '.csv', appr_mse, delimiter=',')
        #file = open('./exports/sub_appr_exact_'+ layer_name + '.txt', mode='w', encoding='utf-8')
        #np.savetxt(file, sub_appr_exact)
        #file.close()
        #file = open('./exports/sub_fault_exact_' + layer_name + '.txt', mode='w', encoding='utf-8')
        #np.savetxt(file, sub_fault_exact)
        #file.close()
        print(neuron1_approx, neuron2_approx, neuron1_fault, neuron2_fault)'''

lec = LayerErrorCalc()
'''layer_name = 'layer5'
path1 = '/home/mahdi/workspace/courses/approximation-instead of fault injection/fault-injection-MLP/quantized/exact.txt'
path2 = '/home/mahdi/workspace/courses/approximation-instead of fault injection/fault-injection-MLP/quantized/appr_out.txt'
path3 = '/home/mahdi/workspace/courses/approximation-instead of fault injection/fault-injection-MLP/quantized/FI_1000_out1.txt'
exact = lec.read_file_contents(path1)
appr_out = lec.read_file_contents(path2)
fault = lec.read_file_contents(path3)'''

#####
#MSE of approximation
#####

sub_appr_exact = lec.subtract()
bit_flip_report = lec.Pool1_bit_error_calculation()

#####
#MSE of fault injection
#####
#sub_fault_exact = lec.subtract1()

#lec.calculate_mse(sub_appr_exact, sub_fault_exact, layer_name)

#####
#bit error rate of appproximation
#####
#lec.bit_error_calculation()
