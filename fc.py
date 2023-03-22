import numpy as np
import evoapproxlib as eal
import warnings
import math
import struct
import random

class FC:
    # input_size = number of input neurons
    # output_size = number of output neurons
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.zeros((1, output_size))

    # returns output for a given input
    def forward(self, input_data):
        self.input = input_data
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    # computes dE/dW, dE/dB for a given output_error=dE/dY. Returns input_error=dE/dX.
    def backprop(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error
        return input_error

    def approx(self, input_data):
            res_lst_approx = [[0 for i in range(len(self.weights[0][:]))] for j in range(len(self.weights[:][:]))]
            res_lst_accurate = [[0 for i in range(len(self.weights[0][:]))] for j in range(len(self.weights[:][:]))]
            for j in range(len(self.weights[0][:])):
                for i in range(len(self.weights[:][:])):
                    a = input_data[0][i]
                    b = self.weights[i][j]
                    res_lst_approx[i][j] = eal.mul8s_1KX2.calc(a, b)
                    res_lst_accurate[i][j] = (input_data[0][i])*(self.weights[i][j])
            out_lst1 = []
            out_lst1_exact = []
            out_lst = []
            out_lst_exact = []
            ###########################################calculate approx output of layer###########################################################
            for i in range (len(res_lst_approx[0])):
                sum = 0
                for j in range (len(res_lst_approx)):
                    sum += res_lst_approx[j][i]
                sum += self.bias[0][i]
                out_lst1.append(sum)
                sig = 1.0/(1.0 + math.e ** (-sum))
                out_lst.append(sig)
            with open('FC_approx_out.txt', 'a') as f:
                for item in out_lst:
                    f.write("%s\n" % item)

            ###########################################calculate exact output of layer###########################################################
            for i in range (len(res_lst_accurate[0])):
                sum = 0
                for j in range (len(res_lst_accurate)):
                    sum += res_lst_accurate[j][i]
                sum += self.bias[0][i]
                out_lst1_exact.append(sum)
                sig = 1.0/(1.0 + math.e ** (-sum))
                out_lst_exact.append(sig)
            with open('FC_exact_out.txt', 'a') as f:
                for item in out_lst_exact:
                    f.write("%s\n" % item)

            ######################## This step is to count error occurance in different output bits before sigmoid function#########################################3
            #exact_input = input_data.tolist()
            #exact_output = np.dot(exact_input, self.weights) + self.bias
            #exact_output1 = exact_output.tolist()
            compare_out_matrix = []
            for i in range(len(out_lst1_exact[0][:])):
                bnr = bin(out_lst1_exact[0][i]).replace('0b','')
                x = bnr[::-1] #this reverses an array
                while len(x) < 5:
                    x += '0'
                bin_exact_out = x[::-1]
                bnr1 = bin(out_lst1[0][i]).replace('0b','')
                x1 = bnr1[::-1] #this reverses an array
                while len(x1) < 5:
                    x1 += '0'
                bin_approx_out = x1[::-1]

                compare = [str(int(bin_approx_out[i]) ^ int(bin_exact_out[i])) for i in range(len(bin_approx_out))]
                compare1 = ''
                for k in compare:
                    compare1 += k
                compare_out_matrix.append(compare1)
            with open('FC_compare__exact_with_approx_out.txt', 'a') as f:
                for item in compare_out_matrix:
                    f.write("%s\n" % item)
            return np.array(out_lst).reshape(1, self.output_size)
