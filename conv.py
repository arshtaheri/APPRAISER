import numpy as np
import evoapproxlib as eal
import random
'''
Note: In this implementation, we assume the input is a 2d numpy array for simplicity, because that's
how our MNIST images are stored. This works for us because we use it as the first layer in our
network, but most CNNs have many more Conv layers. If we were building a bigger network that needed
to use Conv3x3 multiple times, we'd have to make the input be a 3d numpy array.
'''

class Conv3x3:
  # A Convolution layer using 3x3 filters.

  def __init__(self, num_filters):
    self.num_filters = num_filters

    # filters is a 3d array with dimensions (num_filters, 3, 3)
    # We divide by 9 to reduce the variance of our initial values
    self.filters = np.random.randn(num_filters, 3, 3) / 9
  
  def iterate_regions(self, image):
    '''
    Generates all possible 3x3 image regions using valid padding.
    - image is a 2d numpy array.
    '''
    h, w = image.shape
    
    for i in range(h - 2):
      for j in range(w - 2):
        im_region = image[i:(i + 3), j:(j + 3)]
        yield im_region, i, j

  def forward(self, input):
    '''
    Performs a forward pass of the conv layer using the given input.
    Returns a 3d numpy array with dimensions (h, w, num_filters).
    - input is a 2d numpy array
    '''
    self.last_input = input
    #print(input.shape)
    h, w = input.shape
    output = np.zeros((h - 2, w - 2, self.num_filters))

    for im_region, i, j in self.iterate_regions(input):
      output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

    return output

  def backprop(self, d_L_d_out, learn_rate, layer_no):
    '''
    Performs a backward pass of the conv layer.
    - d_L_d_out is the loss gradient for this layer's outputs.
    - learn_rate is a float.
    '''
    
    d_L_d_filters = np.zeros(self.filters.shape)
    if layer_no == 2:
      a, b = self.last_input.shape
      _, _, c = d_L_d_out.shape
      d_L_d_in = np.zeros((int(c / self.num_filters), a, b))
      #print(d_L_d_out.shape)
      #print(d_L_d_in.shape)
      
    for im_region, i, j in self.iterate_regions(self.last_input):
      for f in range(self.num_filters):
        d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
        if layer_no == 2:
          for k in range(int(c / self.num_filters)):
            d_L_d_in[k, i : i+3, j : j+3] += d_L_d_out[i, j, f] * self.filters[f]
    
    # Update filters
    self.filters -= learn_rate * d_L_d_filters
    if layer_no == 2:
      a, b = self.last_input.shape
      d_L_d_in = d_L_d_in[:, :a-1, :b-1]
      return d_L_d_in.reshape((a-1, b-1, int(c / self.num_filters)))
    else:
      return None 
      

#added by mojtaba
class Conv2x2:
  def __init__(self, num_filters, layer_no):
      self.num_filters = num_filters
      self.stride = 2
      self.layer_no = layer_no
      if self.layer_no == 1:
          self.filters = [[-2, -1], [1, 2]]
      elif self.layer_no == 2:
          self.filters = [[2, 1], [-1, -2]] #95.55 accuracy with these filters: [[2, 1], [1, -2]]

  def iterate_regions(self, image):
      h, w = image.shape
      if self.stride == 1:
          h -= 1
          w -= 1
      elif self.stride == 2:
          h -= 2
          w -= 2
      for i in range(0, h, self.stride):
          for j in range(0, w, self.stride):
              im_region = image[i:(i + 2), j:(j + 2)]
              yield im_region, i, j

  def forward(self, input):
      self.last_input = input
      h, w = input.shape
      output = np.zeros((h // self.stride, w // self.stride, self.num_filters))
      for im_region, i, j in self.iterate_regions(input):
        i = i // 2
        j = j // 2
        output[i, j] = np.sum(im_region * self.filters, axis=(0, 1))
      return output
  def forward_approx(self, input):
      u = 0
      q = 0
      rand = 0
      self.last_input = input
      h, w = input.shape
      output = np.zeros((h // self.stride, w // self.stride, self.num_filters))
      for im_region, i, j in self.iterate_regions(input):
        i = i // 2
        j = j // 2
        res_lst_approx = [[0 for l in range(len(im_region[0][:]))] for m in range(len(im_region[:][:]))]
        A = [[0 for l in range(len(im_region[0][:]))] for m in range(len(im_region[:][:]))]
        for l in range(len(im_region[0][:])):
            for m in range(len(im_region[:][:])):
              a = im_region[l][m]
              b = self.filters[l][m]
              #mul8s_1KX2
              if rand == 0:
                  #res_lst_approx[l][m] = eal.mul8s_1KRC.calc(a, b)
                  #res_lst_approx[l][m] = eal.mul8s_1KVA.calc(a, b) ### 86% accuracy on the first Conv layer
                  res_lst_approx[l][m] = eal.mul8s_1L1G.calc(a, b)   ### 66% accuracy on the first Conv layer
                  #res_lst_approx[l][m] = eal.mul8s_1KR6.calc(a, b)    ### 86% accuracy on the first Conv layer
                  A[l][m] = res_lst_approx[l][m]-(a*b)
              else:
                  ##### These numbers will be changed with the different filter sizes in case of applying random APPROXIMATIOn
                  if q < 4 * 4096:
                    if (q % 3) == 0:
                        res_lst_approx[l][m] = eal.mul8s_1KRC.calc(a, b)
                        q += 1
                    else:
                        res_lst_approx[l][m] = a * b
                        q += 1
                        u += 1
                  else:
                        res_lst_approx[l][m] = a * b
                        u += 1

        sum = 0
         ###########################################calculate approx output of layer###########################################################
        for l in range (len(res_lst_approx[0])):
              for m in range (len(res_lst_approx)):
                  sum += res_lst_approx[l][m]
        output[i, j] = sum
      return output, A
  def mask_gen(self, in_val):
      in_val_bin = bin(int(in_val)).replace('0b','') #''.join('{:0>8b}'.format(c) for c in struct.pack('!f', in_val))
      if in_val_bin[0] == '-' :
          in_val_bin = in_val_bin[1:]
          x1 = in_val_bin[::-1] #this reverses an array
          x1 += '1'
      else:
          x1 = in_val_bin[::-1]
      while len(x1) < 16:
          x1 += '0'
      in_val_bin1 = x1[::-1]
      mask_value = '0101011111111111'
      index = 2  # XOR the fourth character in each string (index 3 since Python uses 0-based indexing)
      result = in_val_bin1[:index] + str(int(in_val_bin1[index]) & int(mask_value[index])) + in_val_bin1[index+1:]
      if result[0] == '1':
          z = result[1:]
          new_val = -int(z, 2)
      else:
          new_val = int(result, 2)
      return new_val

  def forward_approx_Mask(self, input):
      u = 0
      q = 0
      rand = 0
      self.last_input = input
      h, w = input.shape
      output = np.zeros((h // self.stride, w // self.stride, self.num_filters))
      for im_region, i, j in self.iterate_regions(input):
        i = i // 2
        j = j // 2
        res_lst_approx = [[0 for l in range(len(im_region[0][:]))] for m in range(len(im_region[:][:]))]
        A = [[0 for l in range(len(im_region[0][:]))] for m in range(len(im_region[:][:]))]
        for l in range(len(im_region[0][:])):
            for m in range(len(im_region[:][:])):
              a = im_region[l][m]
              b = self.filters[l][m]
              #mul8s_1KX2
              if rand == 0:
                  #res_lst_approx[l][m] = eal.mul8s_1KRC.calc(a, b)
                  #res_lst_approx[l][m] = eal.mul8s_1KVA.calc(a, b) ### 86% accuracy on the first Conv layer
                  #res_lst_approx[l][m] = eal.mul8s_1L1G.calc(a, b)   ### 66% accuracy on the first Conv layer
                  #res_lst_approx[l][m] = eal.mul8s_1KR6.calc(a, b)                                 ### 86% accuracy on the first Conv layer
                  res_lst_approx[l][m] = self.mask_gen(eal.mul8s_1KVA.calc(a, b))    ### 71.77% accuracy on the first Conv layer with following mask: '0111111111111111'

                  A[l][m] = res_lst_approx[l][m]-(a*b)
              else:
                  ##### These numbers will be changed with the different filter sizes in case of applying random APPROXIMATIOn
                  if q < 4 * 4096:
                    if (q % 3) == 0:
                        res_lst_approx[l][m] = eal.mul8s_1KRC.calc(a, b)
                        q += 1
                    else:
                        res_lst_approx[l][m] = a * b
                        q += 1
                        u += 1
                  else:
                        res_lst_approx[l][m] = a * b
                        u += 1

        sum = 0
        ###########################################calculate approx output of layer###########################################################
        for l in range (len(res_lst_approx[0])):
              for m in range (len(res_lst_approx)):
                  sum += res_lst_approx[l][m]
        output[i, j] = sum
      return output, A

  def fault_gen(self, in_val):
      in_val_bin = bin(int(in_val)).replace('0b','') #''.join('{:0>8b}'.format(c) for c in struct.pack('!f', in_val))
      if in_val_bin[0] == '-' :
          in_val_bin = in_val_bin[1:]
          x1 = in_val_bin[::-1] #this reverses an array
          x1 += '1'
      else:
          x1 = in_val_bin[::-1]
      while len(x1) < 3:
          x1 += '0'
      in_val_bin1 = x1[::-1]
      fault_loc = random.randint(0, 2)
      new_val_bin = in_val_bin1[:2-fault_loc]
      if in_val_bin1[2 - fault_loc] == '0':
          new_val_bin += '1'
      else:
          new_val_bin += '0'
      new_val_bin += in_val_bin1[2-fault_loc+1:]
      if new_val_bin[0] == '1':
          z = new_val_bin[1:]
          new_val = -int(z, 2)
      else:
          new_val = int(new_val_bin, 2)
      return new_val, fault_loc
  def forward_fault(self, input):
      self.last_input = input
      h, w = input.shape
      z = 3   #number of iteration to reach the reliability confidence level
      output = np.zeros((h // self.stride, w // self.stride, self.num_filters))
      for im_region, i, j in self.iterate_regions(input):
        i = i // 2
        j = j // 2
        for z in range(z):
            for p in range(3):
                l = random.randint(0, 1)     #number of neurons
                m = random.randint(0, 1)
                neur_val = self.filters[l][m]
                faulty_val, fault_loc = self.fault_gen(neur_val)
                while faulty_val == 'nan':
                    faulty_val, fault_loc = self.fault_gen(neur_val)
                self.filters[l][m] = faulty_val
                #print(self.filters)
        output[i, j] = np.sum(im_region * self.filters, axis=(0, 1))
      return output
  def backprop(self, d_L_d_out, learn_rate, layer_no):
      d_L_d_filters = np.zeros(self.filters.shape)
      if layer_no == 2:
          a, b = self.last_input.shape
          _, _, c = d_L_d_out.shape
          d_L_d_in = np.zeros((c // self.num_filters, a, b))
          #print('backprop conv2')
          #print(self.last_input.shape)
          #print(d_L_d_out.shape)
          #print(d_L_d_in.shape)

      for im_region, i, j in self.iterate_regions(self.last_input):
          for f in range(self.num_filters):
              if self.stride == 2:
                  i = i // 2
                  j = j // 2
              d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
              if layer_no == 2:
                  for k in range(c // self.num_filters):
                      d_L_d_in[k, i : i+2, j : j+2] += d_L_d_out[i, j, f] * self.filters[f]

      # Update filters
      self.filters -= learn_rate * d_L_d_filters
      if layer_no == 2:
          if self.stride == 1:
              a, b = self.last_input.shape
              d_L_d_in = d_L_d_in[:, :a-1, :b-1]
              return d_L_d_in.reshape((a-1, b-1, int(c / self.num_filters)))
          elif self.stride == 2:
              return d_L_d_in.reshape((a, b, int(c / self.num_filters)))
      else:
          return None 
