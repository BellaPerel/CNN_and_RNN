from __future__ import unicode_literals, print_function, division
# from train_q2 import *

from io import open
import glob
import os
import unicodedata
import string
import torch
import torch.nn as nn
import pickle
import matplotlib.pyplot as plt
import random


def evaluate_model_q2(language, start_letter):
    class RNN(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(RNN, self).__init__()
            self.hidden_size = hidden_size

            self.i2hidden = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
            self.i2output = nn.Linear(n_categories + input_size + hidden_size, output_size)
            self.o2output = nn.Linear(hidden_size + output_size, output_size)
            self.dropout = nn.Dropout(0.1)
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, category, input, hidden):
            input_combined = torch.cat((category, input, hidden), 1)
            hidden = self.i2hidden(input_combined)
            output = self.i2output(input_combined)
            output_combined = torch.cat((hidden, output), 1)
            output = self.o2output(output_combined)
            output = self.dropout(output)
            output = self.softmax(output)
            return output, hidden

        def init_hidden(self):
            return torch.zeros(1, self.hidden_size)
    all_letters = string.ascii_letters + " .,;'-"
    n_letters = len(all_letters) + 1

    def find_files(path):
        return glob.glob(path)

    def unicode_to_ascii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )


    def read_lines(filename):
        with open(filename, encoding='utf-8') as some_file:
            return [unicode_to_ascii(line.strip()) for line in some_file]

    category_lines = {}
    all_categories = []
    for filename in find_files('data2/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    n_categories = len(all_categories)
    max_length = 20

#grrrrrrrrrrrrrrrrrgrgrgrgr

    def random_choice(l):
        return l[random.randint(0, len(l) - 1)]


    def random_training_pair():
        category = random_choice(all_categories)
        line = random_choice(category_lines[category])
        return category, line

    # One-hot vector for category
    def category_tensor(category):
        li = all_categories.index(category)
        tensor = torch.zeros(1, n_categories)
        tensor[0][li] = 1
        return tensor

    def input_tensor(line):
        tensor = torch.zeros(len(line), 1, n_letters)
        for li in range(len(line)):
            letter = line[li]
            tensor[li][0][all_letters.find(letter)] = 1
        return tensor


    def target_tensor(line):
        letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
        letter_indexes.append(n_letters - 1)
        return torch.LongTensor(letter_indexes)


    def randomTrainingExample():
        category, line = random_training_pair()
        category_tensor2 = category_tensor(category)
        input_line_tensor = input_tensor(line)
        target_line_tensor = target_tensor(line)
        return category_tensor2, input_line_tensor, target_line_tensor


    def sample(category, start_letter='A'):

        with torch.no_grad():
            category_tensor2 = category_tensor(category)

            rnn = RNN(n_letters, 128, n_letters)
            rnn.load_state_dict(torch.load('rnn.pkl', map_location=lambda storage, loc:
            storage))

            hidden = rnn.init_hidden()

            output_name = start_letter

            for i in range(max_length):
                output, hidden = rnn(category_tensor2, input[0], hidden)
                topv, topi = output.topk(1)
                topi = topi[0][0]
                if topi == n_letters - 1:
                    break
                else:
                    letter = all_letters[topi]
                    output_name += letter
                input = input_tensor(letter)

            return output_name


    for s_letter in start_letter:
        print("Input language is: " + language + " Start letter is: " + s_letter)
        print("output generated name:")
        print(sample(language, s_letter))



"""
evaluate_model_q2('Italian', 'M')
evaluate_model_q2('German', 'A')
evaluate_model_q2('Spanish', 'K')
evaluate_model_q2('Chinese', 'C')
"""