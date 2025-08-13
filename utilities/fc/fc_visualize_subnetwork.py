# cf. Oliver Wilken https://stackoverflow.com/questions/29888233/how-to-visualize-a-neural-network

from matplotlib import pyplot
from math import cos, sin, atan


class Neuron:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw(self, neuron_radius):
        circle = pyplot.Circle((self.x, self.y), radius=neuron_radius, fill=False)
        pyplot.gca().add_patch(circle)


class Layer:
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer, connections, weight_colours):
        self.vertical_distance_between_layers = 6
        self.horizontal_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.y = self.__calculate_layer_y_position()
        self.neurons = self.__initialize_neurons(number_of_neurons)
        self.connections = connections
        self.weight_colours = weight_colours  # colour vector with same dimensionality as connections

    def __initialize_neurons(self, number_of_neurons):
        neurons = []
        x = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = Neuron(x, self.y)
            neurons.append(neuron)
            x += self.horizontal_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.horizontal_distance_between_neurons * (self.number_of_neurons_in_widest_layer -
                                                           number_of_neurons) / 2

    def __calculate_layer_y_position(self):
        if self.previous_layer:
            return self.previous_layer.y + self.vertical_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2, weight_colour):
        angle = atan((neuron2.x - neuron1.x) / float(neuron2.y - neuron1.y))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = pyplot.Line2D((neuron1.x - x_adjustment, neuron2.x + x_adjustment),
                             (neuron1.y - y_adjustment, neuron2.y + y_adjustment),
                             color=weight_colour)
        pyplot.gca().add_line(line)

    def draw(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw(self.neuron_radius)
        if self.previous_layer:  # If layer is not input layer
            # Draw line between neuron of previous layer and every neuron of this layer
            current_connection_idx = 0
            for j, previous_layer_neuron in enumerate(self.previous_layer.neurons):
                for i, neuron in enumerate(self.neurons):
                    if self.connections[current_connection_idx]:
                        self.__line_between_two_neurons(neuron, previous_layer_neuron,
                                                        self.weight_colours[current_connection_idx])
                    current_connection_idx += 1
        # write Text
        y = self.y - 0.1
        x_text = self.number_of_neurons_in_widest_layer * self.horizontal_distance_between_neurons
        if layerType == 0:
            pyplot.text(x_text, y, 'Input Layer', fontsize=12)
        elif layerType == -1:
            pyplot.text(x_text, y, 'Output Layer', fontsize=12)
        else:
            pyplot.text(x_text, y, 'Hidden Layer ' + str(layerType), fontsize=12)


class Subnetwork:
    def __init__(self, number_of_neurons_in_widest_layer, NN_architecture):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.NN_architecture = NN_architecture
        self.layers = []
        self.layertype = 0

    def add_layer(self, number_of_neurons, connections, weight_colours):
        layer = Layer(self, number_of_neurons, self.number_of_neurons_in_widest_layer, connections, weight_colours)
        self.layers.append(layer)

    def draw(self, save_path):
        fig = pyplot.figure()
        fig.set_size_inches(18.5, 10.5)  # TODO: Make adaptive to network size
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                i = -1
            layer.draw(i)
        pyplot.axis('scaled')
        pyplot.axis('off')
        pyplot.title(str(self.NN_architecture) + " - network", fontsize=15)
        if save_path:
            fig.savefig(save_path, dpi=100)
        else:
            pyplot.show()


class DrawSubnetwork:
    def __init__(self, NN_architecture, bit_vector, save_path=None, **kwargs):
        self.args = kwargs
        self.NN_architecture = NN_architecture
        self.number_of_connections_between_layers = [0] + [self.NN_architecture[l] * self.NN_architecture[l + 1]
                                                           for l in range(len(self.NN_architecture) - 1)]
        self.bit_vector = bit_vector
        self.colour_vector = self.args.get("colour_vector", ["purple" for _ in bit_vector])
        self.save_path = save_path

    def draw(self):
        widest_layer = max(self.NN_architecture)
        network = Subnetwork(widest_layer, self.NN_architecture)
        for layer_num, l in enumerate(self.NN_architecture):
            layer_range = (sum(self.number_of_connections_between_layers[:layer_num]),
                           sum(self.number_of_connections_between_layers[:layer_num]) +
                           self.number_of_connections_between_layers[layer_num])
            network.add_layer(l, self.bit_vector[layer_range[0]: layer_range[1]],
                              self.colour_vector[layer_range[0]: layer_range[1]])
        network.draw(self.save_path)


if __name__ == "__main__":
    net = DrawSubnetwork([2, 3, 1], [1, 1, 0, 0, 1, 0, 1, 1, 0])
    net.draw()

