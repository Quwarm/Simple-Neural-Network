import math, random, copy


class NeuralNetwork:
    @staticmethod
    def sigmoid(x):
        """
        A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
        :param x: argument
        :return: sigmoid(x)
        """
        return 1. / (1. + math.exp(-x))

    @staticmethod
    def dsigmoid(sigmoid_value):
        """
        Derivative of sigmoid function
        :param sigmoid_value: sigmoid(x)
        :return: derivative of sigmoid(x)
        """
        return sigmoid_value * (1 - sigmoid_value)

    def __init__(self, input, hidden_layers, output):
        """
        Initialization
        :param input: array of input data arrays of several variables
        :param hidden_layers: array of the number of neurons in hidden layers
        :param output: array of output data arrays of several variables (0 <= each value <= 1)
        """
        self.input = copy.deepcopy(input)
        self.hidden_layers = copy.deepcopy(hidden_layers)
        self.output = copy.deepcopy(output)
        if len(self.input) != len(self.output):
            raise ValueError
        self.layers = list(map(int, [len(self.input[0]), *self.hidden_layers, len(self.output[0])]))
        self.activations = [[0 for _ in range(self.layers[i])] for i in range(len(self.layers))]
        self.weights = [[[random.uniform(-5, 5) for _ in range(self.layers[i + 1])] for j in range(self.layers[i])]
                        for i in range(len(self.layers) - 1)]

    def update(self, arbitrary_input):
        """
        Update method: learning and getting results by arbitrary input
        :param arbitrary_input: arbitrary input
        :return: result
        """
        if len(arbitrary_input) != self.layers[0]:
            raise ValueError
        for i in range(len(arbitrary_input)):
            self.activations[0][i] = arbitrary_input[i]
        for i in range(len(self.layers) - 1):
            for j in range(self.layers[i + 1]):
                s = 0
                for k in range(self.layers[i]):
                    s += self.activations[i][k] * self.weights[i][k][j]
                self.activations[i + 1][j] = NeuralNetwork.sigmoid(s)
        return self.activations[-1]

    def back_propagation(self, target, learning_rate):
        """
        Error back propagation method
        :param target: target for error back propagation method
        :param learning_rate: learning rate
        :return: maximum error between results
        """
        if len(target) != self.layers[-1]:
            raise ValueError
        delta = [[0 for _ in range(self.layers[i])] for i in range(len(self.layers))]
        # hidden -> output
        o = len(self.layers) - 1
        for i in range(self.layers[o]):
            delta[o][i] = (target[i] - self.activations[o][i]) * NeuralNetwork.dsigmoid(self.activations[o][i])
        # input and hidden -> hidden
        for i in range(len(self.layers) - 2, 0, -1):
            for j in range(self.layers[i]):
                for k in range(self.layers[i + 1]):
                    delta[i][j] += delta[i + 1][k] * self.weights[i][j][k]
                delta[i][j] *= NeuralNetwork.dsigmoid(self.activations[i][j])
        # update weights
        for i in range(len(self.layers) - 1):
            for j in range(self.layers[i]):
                for k in range(self.layers[i + 1]):
                    self.weights[i][j][k] += learning_rate * delta[i + 1][k] * self.activations[i][j]
        # find error
        error = 0
        for i in range(len(target)):
            error = max(error, abs(target[i] - self.activations[o][i]))
        return error

    def train(self, iterations, learning_rate=0.5):
        """
        Training method
        :param iterations: number of iterations
        :param learning_rate: learning speed from 0 to 1
        :return: None
        """
        k = len(self.output)
        for i in range(iterations):
            self.update(self.input[i % k])
            self.back_propagation(self.output[i % k], learning_rate)


if __name__ == '__main__':
    print('XOR FUNCTION APPROXIMATION')
    n = 8
    input = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    output = [[0], [1], [1], [0], [1], [0], [0], [1]]
    network = NeuralNetwork(input, [4, 4], output)
    network.train(2000 * n - 1, 0.5)
    net_result = []
    error = 0
    for i in range(n):
        temp = network.update(input[i])
        error = max(error, abs(temp[0] - output[i][0]))
        net_result.append(temp[0])
    print('Expected results:', output)
    print('Actual results:', net_result)
    print('Maximum error:', error)
    print('Weights:', network.weights)
    print('\n\n')
    print('[SIN(X) + 1] / 2 FUNCTION APPROXIMATION, 0 <= X <= 2PI')
    n = 20
    dx = 2 * math.pi / (n - 1)
    input = [[round(i * dx, 4), 1] for i in range(n)]
    output = [[(math.sin(i * dx) + 1) / 2] for i in range(n)]
    network = NeuralNetwork(input, [6, 6, 6], output)
    network.train(650 * n - 1, 0.5)
    net_result = []
    error = 0
    for i in range(n):
        temp = network.update(input[i])
        error = max(error, abs(temp[0] - output[i][0]))
        net_result.append(temp[0])
    print('Expected results:', output)
    print('Actual results:', net_result)
    print('Maximum error:', error)
    print('Weights:', network.weights)
