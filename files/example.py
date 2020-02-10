from network import NeuralNetwork
import math

if __name__ == '__main__':
    print('XOR FUNCTION APPROXIMATION')
    n = 8
    input = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    output = [[0], [1], [1], [0], [1], [0], [0], [1]]
    network = NeuralNetwork(input, [4, 4], output)
    network.train(20000 * n - 1, 0.5)
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
