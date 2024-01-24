import numpy as np


class Module(object):

    def __init__(self):
        self.output = None
        self.grad_input = None
        self.training = True

    def forward(self, input):
        return self.update_output(input)

    def backward(self, input, grad_output):
        self.update_grad_input(input, grad_output)
        self.accumulate_grad_parameters(input, grad_output)
        return self.grad_input

    def update_output(self, input):
        pass

    def update_grad_input(self, input, grad_output):
        pass

    def accumulate_grad_parameters(self, input, grad_output):
        pass

    def zero_grad_parameters(self):
        pass

    def get_parameters(self):
        return []

    def get_grad_parameters(self):
        return []

    def train(self):
        self.training = True

    def evaluate(self):
        self.training = False

    def __repr__(self):
        return "Module"


class LinearLayer(Module):

    def __init__(self, n_in, n_out):
        super(LinearLayer, self).__init__()
        self.W = np.random.normal(loc=0, scale=(2/(n_in + n_out)) ** 0.5, size=(n_out, n_in))
        self.b = np.zeros(shape=(n_out,))
        self.gradW = np.zeros_like(self.W)
        self.gradb = np.zeros_like(self.b)

    def update_output(self, input):
        self.output = input @ self.W.T + self.b
        return self.output

    def update_grad_input(self, input, grad_output):
        self.grad_input = grad_output @ self.W
        return self.grad_input

    def accumulate_grad_parameters(self, input, grad_output):
        np.matmul(grad_output.T, input, out=self.gradW)
        np.sum(grad_output, axis=0, out=self.gradb)

    def zero_grad_parameters(self):
        self.gradW.fill(0)
        self.gradb.fill(0)

    def get_parameters(self):
        return [self.W, self.b]

    def get_grad_parameters(self):
        return [self.gradW, self.gradb]

    def __repr__(self):
        return f"Linear {self.W.shape[1]} -> {self.W.shape[0]}"


class ReLU(Module):

    def update_output(self, input):
        self.output = np.maximum(input, 0)
        return self.output

    def update_grad_input(self, input, grad_output):
        self.grad_input = np.multiply(grad_output, input > 0)
        return self.grad_input

    def __repr__(self):
        return f"ReLU"


class Sequential(Module):

    def __init__(self):
        super(Sequential, self).__init__()
        self.modules = []

    def add(self, module):
        self.modules.append(module)

    def update_output(self, input):
        self.output = input
        for i, module in enumerate(self.modules):
            self.output = module.forward(input)
            input = self.output
        return self.output

    def backward(self, input, grad_output):

        for i in range(len(self.modules) - 1, 0, -1):
            grad_output = self.modules[i].backward(self.modules[i - 1].output, grad_output)
        self.grad_input = self.modules[0].backward(input, grad_output)
        return self.grad_input

    def zero_grad_parameters(self):
        for module in self.modules:
            module.zero_grad_parameters()

    def get_parameters(self):
        return [module.get_parameters() for module in self.modules]

    def get_grad_parameters(self):
        return [module.get_grad_parameters() for module in self.modules]

    def train(self):
        self.training = True
        for module in self.modules:
            module.train()

    def evaluate(self):
        self.training = False
        for module in self.modules:
            module.evaluate()

    def __repr__(self):
        return "\n".join(str(module) for module in self.modules)


class Criterion(object):
    def __init__(self):
        self.output = None
        self.grad_input = None

    def forward(self, input, target):
        return self.update_output(input, target)

    def backward(self, input, target):
        return self.update_grad_input(input, target)

    def update_output(self, input, target):
        return self.output

    def update_grad_input(self, input, target):
        return self.grad_input

    def __repr__(self):
        return "Criterion"


class MSECriterion(Criterion):
    def __init__(self):
        super(MSECriterion, self).__init__()

    def update_output(self, input, target):
        self.output = np.sum(np.power(input - target, 2)) / input.shape[0]
        return self.output

    def update_grad_input(self, input, target):
        self.grad_input = (input - target) * 2 / input.shape[0]
        return self.grad_input

    def __repr__(self):
        return "MSECriterion"


def build_mlp(linear_sizes):
    assert len(linear_sizes) >= 2, "len(linear_sizes) >= 2"

    mlp = Sequential()
    for size_idx in range(len(linear_sizes) - 2):
        mlp.add(LinearLayer(
            n_in=linear_sizes[size_idx],
            n_out=linear_sizes[size_idx + 1]
        ))
        mlp.add(ReLU())
    mlp.add(LinearLayer(
        n_in=linear_sizes[-2],
        n_out=linear_sizes[-1]
    ))

    return mlp


def sgd(variables, gradients, config):

    for idx_lay, (current_layer_vars, current_layer_grads) in enumerate(zip(variables, gradients)):
        for idx_var, (current_var, current_grad) in enumerate(zip(current_layer_vars, current_layer_grads)):
            current_var -= config['learning_rate'] * current_grad
