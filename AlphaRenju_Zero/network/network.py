from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers import add
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD
from ..rules import *
from ..decorator import *
import numpy as np
import os


class Network:
    def __init__(self, conf):
        # All hyperparameters used in the model
        self._board_size = conf['board_size']  # the size of the playing board
        self._lr = conf['learning_rate']  # learning rate of SGD (2e-3)
        self._momentum = conf['momentum']  # nesterov momentum (1e-1)
        self._l2_coef = conf['l2']  # coefficient of L2 penalty (1e-4)
        self._mini_batch_size = conf['mini_batch_size']  # the size of batch when training the network
        self._fit_epochs = conf['fit_epochs']  # the number of iteration

        # Define Network
        self._build_network()

        # The location of the file which stores the parameters of the network
        self._net_para_file = conf['net_para_file']
        self._fit_history_file = conf['fit_history_file']

        # Whether we use previous model or not
        self._use_previous_model = conf['use_previous_model']
        if self._use_previous_model:
            if os.path.exists(self._net_para_file):
                self._model.load_weights(self._net_para_file)
            else:
                print('> error: [use_previous_model] = True, ' + self._net_para_file + ' not found')

    @log
    def _build_network(self):
        # Input_Layer
        init_x = Input((4, self._board_size, self._board_size))  # the input is a tensor with the shape 3*(15*15)
        x = init_x

        # First Convolutional Layer with 32 filters
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   data_format='channels_first', kernel_regularizer=l2(self._l2_coef))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # Two Residual Blocks
        x = self._residual_block(x)
        x = self._residual_block(x)
        # x = self._residual_block(x)

        # Policy Head for generating prior probability vector for each action
        policy = Conv2D(filters=2, kernel_size=(1, 1), strides=(1, 1), padding='same',
                        data_format='channels_first', kernel_regularizer=l2(self._l2_coef))(x)
        policy = BatchNormalization()(policy)
        policy = Activation('relu')(policy)
        policy = Flatten()(policy)
        policy = Dense(self._board_size*self._board_size, kernel_regularizer=l2(self._l2_coef))(policy)
        self._policy = Activation('softmax')(policy)

        # Value Head for generating value of each action
        value = Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='same',
                       data_format="channels_first", kernel_regularizer=l2(self._l2_coef))(x)
        value = BatchNormalization()(value)
        value = Activation('relu')(value)
        value = Flatten()(value)
        value = Dense(32, kernel_regularizer=l2(self._l2_coef))(value)
        value = Activation('relu')(value)
        value = Dense(1, kernel_regularizer=l2(self._l2_coef))(value)
        self._value = Activation('tanh')(value)

        # Define Network
        self._model = Model(inputs=init_x, outputs=[self._policy, self._value])

        # Define the Loss Function
        opt = SGD(lr=self._lr, momentum=self._momentum, nesterov=True)  # stochastic gradient descend with momentum
        losses_type = ['categorical_crossentropy', 'mean_squared_error']  # cross-entrophy and MSE are weighted equally
        self._model.compile(optimizer=opt, loss=losses_type)

    def _residual_block(self, x):
        x_shortcut = x
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   data_format="channels_first", kernel_regularizer=l2(self._l2_coef))(x)
        x = BatchNormalization()(x) 
        x = Activation('relu')(x)
        x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                   data_format="channels_first", kernel_regularizer=l2(self._l2_coef))(x)
        x = BatchNormalization()(x) 
        x = add([x, x_shortcut])  # Skip Connection
        x = Activation('relu')(x)
        return x
        
    def predict(self, board, color, last_move):
        if sum(sum(board)) == 0 and color == WHITE:
            print('error: network.predict')
        if sum(sum(board)) == 1 and color == BLACK:
            print('error: network.predict')
        tensor = board2tensor(board, color, last_move)
        policy, value_tensor = self._model.predict_on_batch(tensor)
        value = value_tensor[0][0]
        return policy, value

    def train(self, board_list, color_list, last_move_list, pi_list, z_list):
        size = len(color_list)
        for i in range(size):
            if sum(sum(board_list[i])) == 0 and color_list[i] == WHITE:
                print('error: network.train')
            if sum(sum(board_list[i])) == 1 and color_list[i] == BLACK:
                print('error: network.train')


        # Data Augmentation through symmetric and self-rotation transformation
        board_aug = []
        color_aug = []
        last_move_aug = []
        pi_aug = []
        z_aug = []
        for i in range(len(board_list)):
            new_board, new_color, new_last_move, new_pi, new_z = \
                data_augmentation(board_list[i], color_list[i], last_move_list[i], pi_list[i], z_list[i])
            board_aug.extend(new_board)
            color_aug.extend(new_color)
            last_move_aug.extend(new_last_move)
            pi_aug.extend(new_pi)
            z_aug.extend(new_z)
        board_list.extend(board_aug)
        color_list.extend(color_aug)
        last_move_list.extend(last_move_aug)
        pi_list.extend(pi_aug)
        z_list.extend(z_aug)

        # Regularize Data

        board_list = np.array([board2tensor(board_list[i], color_list[i], last_move_list[i], reshape_flag=False)
                               for i in range(len(board_list))])
        pi_list = np.array(pi_list)
        z_list = np.array(z_list)

        # Training

        hist = self._model.fit(board_list, [pi_list, z_list], epochs=self._fit_epochs, batch_size=self._mini_batch_size, verbose=1)
        hist_path = self._fit_history_file + '_' + str(self._fit_epochs) + '_' + str(self._mini_batch_size) + '.txt'
        with open(hist_path, 'a') as f:
            f.write(str(hist.history))
            return hist.history['loss'][0]  # only sample loss of first epoch
        
    def get_para(self):
        net_para = self._model.get_weights() 
        return net_para

    def save_model(self):
        """ save model para to file """
        self._model.save_weights(self._net_para_file)

    def load_model(self):
        if os.path.exists(self._net_para_file):
            self._model.load_weights(self._net_para_file)
        else:
            print('> error: ' + self._net_para_file + ' not found')


# Transform a board(matrix) to a tensor
def board2tensor(board, color, last_move, reshape_flag=True):

    # Current-Stone Layer
    cur = np.array(np.array(board) == color, dtype=np.int)

    # Enemy-Stone Layer
    e = np.array(np.array(board) == -color, dtype=np.int)

    # Last Step Layer
    l = np.zeros((board.shape[0], board.shape[1]))
    if last_move is not None:
        l[last_move[0]][last_move[1]] = 1

    # Color Layer
    flag = (1 if color == BLACK else 0)
    c = flag * np.ones((board.shape[0], board.shape[1]))

    # Stack cur,e,c into tensor
    tensor = np.array([cur, e, l, c])
    if reshape_flag:
        tensor = tensor.reshape(1, tensor.shape[0], tensor.shape[1], tensor.shape[2])
    return tensor


# Augment the training data pool through plane transformation
def data_augmentation(board, color, last_move, pi, z):
    new_board = []
    new_color = [color]*7
    new_last_move = []
    new_pi = []
    new_z = [z]*7
    for type in range(1, 8):
        board_t = board_transform(board, type, flag=1)
        last_move_t = coordinate_transform(last_move, type, board.shape[0], flag=1)
        pi_t = input_encode(pi, type, board.shape[0])
        new_board.append(board_t)
        new_last_move.append(last_move_t)
        new_pi.append(pi_t)
    return new_board, new_color, new_last_move, new_pi, new_z


# Transform the input vector given transformation type
def input_encode(vec, num, size):
    mat = np.reshape(vec, (size, size))  # reshape vector into matrix
    mat = board_transform(mat, num, flag=1)
    vec = np.reshape(mat, (1, size**2))
    return vec[0]


# Transform the output vector to its initial shape given the transformation type
def output_decode(vec, num, size):
    mat = np.reshape(vec, (size,size))   # reshape vector into matrix
    inv_mat = board_transform(mat, num, flag=2)
    vec = np.reshape(inv_mat, (1, size**2))
    return vec[0]


def coordinate_transform(move, type, size, flag):
    if move is None:
        return None
    board = np.zeros((size, size))
    board[move[0]][move[1]] = 1
    board_t = board_transform(board, type, flag)
    temp = np.where(board_t == 1)
    new_move = (temp[0][0], temp[1][0])
    return new_move


# Transform the input board by simple plane transformation
def board_transform(mat, num, flag=0):

    def R0(mat):
        return mat

    def R1(mat):
        mat = np.rot90(mat, 1)
        return mat

    def R2(mat):
        mat = np.rot90(mat, 2)
        return mat

    def R3(mat):
        mat = np.rot90(mat, 3)
        return mat

    def S(mat):
        mat = R0(np.fliplr(mat))
        return mat

    def SR1(mat):
        mat = R1(np.fliplr(mat))
        return mat

    def SR2(mat):
        mat = R2(np.fliplr(mat))
        return mat

    def SR3(mat):
        mat = R3(np.fliplr(mat))
        return mat

    # Random Transformation
    if flag == 0:
        num = int(np.random.randint(8, size=1))
        total_type = ['R0', 'R1', 'R2', 'R3', 'S', 'SR1', 'SR2', 'SR3']
        real_type = total_type[num]
        return eval(real_type)(mat),num

    # Encode
    elif flag == 1:  # encode
        total_type = ['R0', 'R1', 'R2', 'R3', 'S', 'SR1', 'SR2', 'SR3']
        real_type = total_type[num]
        return eval(real_type)(mat)

    # Decode
    else:
        inv_total_type = ['R0', 'R3', 'R2', 'R1', 'S', 'SR1', 'SR2', 'SR3']
        real_type = inv_total_type[num]
        return eval(real_type)(mat)
