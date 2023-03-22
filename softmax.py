import numpy as np
import evoapproxlib as eal
import warnings
import math
import struct
import random

class Softmax:
  # A standard fully-connected layer with softmax activation.
  def __init__(self, input_len, nodes):
    self.input_len = input_len
    self.nodes = nodes
    self.weights =  np.random.rand(input_len, nodes) - 0.5
    self.biases = np.random.rand(1, nodes) - 0.5    #np.zeros((1, nodes))

  def forward(self, input_vect):
    input_vect = self.input_quantization(input_vect)
    self.last_input_shape = input_vect.shape
    self.last_input = input_vect
    input_len, nodes = self.weights.shape
    totals = np.dot(input_vect, self.weights) + self.biases
    self.last_totals = totals[0]

    exp = np.exp(totals)
    return exp[0] / np.sum(exp[0], axis=0)
  def fault_forward(self, input_vect, file_name):
    totals = np.dot(input_vect, self.weights) + self.biases
    self.last_totals = totals[0]
    '''with open('FI_100000_out1.txt', 'a') as f:
          for item in totals:
              f.write("%s\n" % item)
    return self.last_totals'''

  def forward_max(self, input_vect):
    input_vect = self.input_quantization(input_vect)
    self.last_input_shape = input_vect.shape
    self.last_input = input_vect
    input_len, nodes = self.weights.shape
    totals = np.dot(input_vect, self.weights) + self.biases
    self.last_totals = totals[0]

    exp = np.exp(totals)
    return exp[0] / np.sum(exp[0], axis=0)    
    

  def weights_quantization(self):
    for i in range(self.weights.shape[0]):
        for j in range(self.weights.shape[1]):
          w = self.weights[i][j]
          if w > 5:
            w = 5
          elif w < -5:
            w = -5
          OldRange = 10
          NewRange = 15
          OldMin = -5
          NewMin = 0
          NewValue = (((w - OldMin) * NewRange) / OldRange) + NewMin
          self.weights[i][j] = round(NewValue)
    
  def biases_quantization(self):
    for i in range(self.biases.shape[0]):
        for j in range(self.biases.shape[1]):
          b = self.biases[i][j]
          if b > 5:
            b = 5
          elif b < -5:
            b = -5
          OldRange = 10
          NewRange = 15
          OldMin = -5
          NewMin = 0
          NewValue = (((b - OldMin) * NewRange) / OldRange) + NewMin
          self.biases[i][j] = round(NewValue)
  
  def input_quantization(self, input_vect):
    for i in range(input_vect.shape[0]):
        for j in range(input_vect.shape[1]):
            act = input_vect[i][j]
            if act > 150:
                act = 150
            elif act < -8:
                act = -8
            OldRange = 158
            NewRange = 15
            OldMin = -8
            NewMin = 0 #-8
            NewValue = (((act - OldMin) * NewRange) / OldRange) + NewMin
            input_vect[i][j] = round(NewValue)
    return input_vect
  
  def forward_quantized(self, input_vect, file_name):
    #self.save_inputs(r"...\My shared files\cnn\cnn-catdog\fc_files\fc_input_" + file_name + ".txt", input_vect)
    #print(input_vect)
    input_vect = self.input_quantization(input_vect)
    #print(input_vect)
    #self.save_inputs(r"C:...\My shared files\cnn\cnn-catdog\fc_files\fc_quant_" + file_name + ".txt", input_vect)
    totals = np.dot(input_vect, self.weights) + self.biases
    self.last_totals = totals[0]
    return self.last_totals


  def approx(self, input_vect, file_name):
      input_vect = self.input_quantization(input_vect)
      res_lst_approx = [[0 for i in range(len(self.weights[0][:]))] for j in range(len(self.weights[:][:]))]
      res_lst_accurate = [[0 for i in range(len(self.weights[0][:]))] for j in range(len(self.weights[:][:]))]
      for j in range(len(self.weights[0][:])):
          for i in range(len(self.weights[:][:])):
              a = input_vect[0][i]
              b = self.weights[i][j]
              res_lst_approx[i][j] = eal.mul8u_JQQ.calc(a, b)
              res_lst_accurate[i][j] = (input_vect[0][i])*(self.weights[i][j])
      out_lst1 = []
      out_lst1_exact = []
      out_lst = []
      out_lst_exact = []
      ###########################################calculate approx output of layer###########################################################
      for i in range (len(res_lst_approx[0])):
          sum = 0
          for j in range (len(res_lst_approx)):
              sum += res_lst_approx[j][i]
          sum += self.biases[0][i]
          out_lst1.append(sum)
          #sig = 1.0/(1.0 + math.e ** (-sum))
          #out_lst.append(sig)
      with open('approx_out.txt', 'a') as f:
          for item in out_lst1:
              f.write("%s\n" % item)
      ###########################################calculate exact output of layer###########################################################
      for i in range (len(res_lst_accurate[0])):
          sum = 0
          for j in range (len(res_lst_accurate)):
              sum += res_lst_accurate[j][i]
          sum += self.biases[0][i]
          out_lst1_exact.append(sum)
          #sig = 1.0/(1.0 + math.e ** (-sum))
          #out_lst_exact.append(sig)
      '''with open('FC_exact_out.txt', 'a') as f:
          for item in out_lst1_exact:
              f.write("%s\n" % item)'''
      ######################## This step is to count error occurance in different output bits before sigmoid function#########################################3
      #exact_input = input_data.tolist()
      #exact_output = np.dot(exact_input, self.weights) + self.bias
      #exact_output1 = exact_output.tolist()
      compare_out_matrix = []
      for i in range(len(out_lst1_exact[:])):
          bnr = bin(int(out_lst1_exact[i])).replace('0b','')
          bnr1 = bin(int(out_lst1[i])).replace('0b','')
          x = bnr[::-1] #this reverses an array
          x1 = bnr1[::-1] #this reverses an array
          while len(x) < len(x1):
              x += '0'
          bin_exact_out = x[::-1]


          while len(x1) < len(x):
              x1 += '0'
          bin_approx_out = x1[::-1]

          compare = f'{[(ord(a) ^ ord(b)) for a,b in zip(bin_exact_out, bin_approx_out)]}'
          #[str(int(out_lst1_exact[i]) ^ int(out_lst1[i])) for i in range(len(out_lst1_exact))]
          compare1 = ''
          for k in compare:
              compare1 += k
          compare_out_matrix.append(compare1)
      '''with open('FC_compare__exact_with_approx_out.txt', 'a') as f:
          for item in compare_out_matrix:
              f.write("%s\n" % item)'''
      return np.array(out_lst1)
  '''def forward_fault_in_weights(self, input_vect, fault_rate):
    input_vect = self.input_quantization(input_vect)
    #self.last_input = input_data
    fault_num = round(fault_rate * len(self.weights) * len(self.weights[0]))
    Diff = self.weights.copy()
    fault_weights_lst = []
    fault_weights_lst1 = []
    #fault injection in random inputs
    while len(fault_weights_lst) < fault_num:
        rand_neuron = random.randint(0, len(self.weights)-1)
        rand_neuron1 = random.randint(0, len(self.weights[0])-1)
        #if rand_neuron not in fault_weights_lst:
        fault_weights_lst.append(rand_neuron)
        #if rand_neuron1 not in fault_weights_lst1:
        fault_weights_lst1.append(rand_neuron1)
    counter1 = 0
    for i in range(fault_num):
        ################################### fault_generation####################
        #counter = 0
        a = fault_weights_lst[i]
        b = fault_weights_lst1[i]
        J = self.weights[a][b]
        test = bin(struct.unpack('!I', struct.pack('!f', J))[0])[2:].zfill(32)
        in_val_bin = bin(struct.unpack('!I', struct.pack('!f', J))[0])[2:].zfill(32)
        fault_location_list = []
        for k in range(round(fault_rate * 29)):
            fault_loc = random.randint(0, 29)
            if fault_loc not in fault_location_list:
                fault_location_list.append(fault_loc)
                new_val_bin = in_val_bin[:31-fault_loc]
                if in_val_bin[31 - fault_loc] == '0':
                    new_val_bin += '1'
                else:
                    new_val_bin += '0'
                new_val_bin += in_val_bin[31-fault_loc+1:]
            in_val_bin = new_val_bin
        value = struct.unpack('!f',struct.pack('!I', int(in_val_bin, 2)))[0]
        self.weights[a][b] = value
    #diff = Diff - self.weights
    #print(diff)
    self.lin_output = np.dot(input_data, self.weights) + self.bias
    ######################## This step is to count fault occurance in different output bits before sigmoid function#########################################3
    #out_lst1 = out_lst1.astype(int)
    exact_input = input_data.tolist()
    exact_output = np.dot(exact_input, Diff) + self.bias
    exact_output1 = exact_output.tolist()
    fault_output = self.lin_output.tolist()
    #exact_output - exact_output.astype(int)
    compare_out_matrix = []
    for i in range(len(exact_output[0][:])):
        #bin_compare = bin(struct.unpack('!I', struct.pack('!f', compare))[0])[2:].zfill(32)
        bin_exact_out = bin(struct.unpack('!I', struct.pack('!f', exact_output1[0][i]))[0])[2:].zfill(32)
        bin_fault_out = bin(struct.unpack('!I', struct.pack('!f', fault_output[0][i]))[0])[2:].zfill(32)
        compare = [str(int(bin_fault_out[i]) ^ int(bin_exact_out[i])) for i in range(len(bin_fault_out))]
        compare1 = ''
        for k in compare:
            compare1 += k
        #bin_compare = bin(struct.unpack('!I', struct.pack('!f', int(compare[0]))[0])[2:].zfill(32)
        compare_out_matrix.append(compare1)
    #with open('layer4_0.01_error_rate_fault_injection_out.txt', 'a') as f:
    #    for item in compare_out_matrix:
    #        f.write("%s\n" % item)
    return self.sigmoid(self.lin_output)'''

  def backprop(self, d_L_d_out, learn_rate):
    '''
    Performs a backward pass of the softmax layer.
    Returns the loss gradient for this layer's inputs.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    # We know only 1 element of d_L_d_out will be nonzero
    for i, gradient in enumerate(d_L_d_out):
      if gradient == 0:
        continue

      # e^totals
      t_exp = np.exp(self.last_totals)

      # Sum of all e^totals
      S = np.sum(t_exp)

      # Gradients of out[i] against totals
      d_out_d_t = -t_exp[i] * t_exp / (S ** 2)
      d_out_d_t[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

      # Gradients of totals against weights/biases/input
      d_t_d_w = self.last_input
      d_t_d_b = 1
      d_t_d_inputs = self.weights

      # Gradients of loss against totals
      d_L_d_t = gradient * d_out_d_t

      # Gradients of loss against weights/biases/input
      d_L_d_w = d_t_d_w[np.newaxis].T @ d_L_d_t[np.newaxis]
      d_L_d_b = d_L_d_t * d_t_d_b
      d_L_d_inputs = d_t_d_inputs @ d_L_d_t.T
      d_L_d_w = d_L_d_w.reshape(self.weights.shape)
      
      # Update weights / biases
      self.weights -= learn_rate * d_L_d_w
      self.biases -= learn_rate * d_L_d_b

      return d_L_d_inputs.reshape(self.last_input_shape)
      
  
  def save_weights(self, addr):
    file = open(addr, 'w')
    for i in range(self.input_len):
      file.write(str(self.weights[i][0]) + " " + str(self.weights[i][1]) + "\n")
    file.close()
    
  def save_biases(self, addr):
    file = open(addr, 'w')
    for i in range(self.nodes):
      file.write(str(self.biases[0][i]) + "\n")
    file.close()
    
  def save_inputs(self, addr, input_vect):
    file = open(addr, 'w')
    for i in range(input_vect.shape[1]):
        file.write(str(input_vect[0][i]) + "\n")
    file.close()
    
  def read_weights(self, addr):
    file = open(addr, 'r')
    i = 0
    for line in file:
      s = line.split()
      self.weights[i][0] = float(s[0])
      self.weights[i][1] = float(s[1])
      i += 1
        
  def read_biases(self, addr):
    file = open(addr, 'r')
    i = 0
    for line in file:
      s = line.split()
      self.biases[0][i] = float(s[0])
      i += 1
