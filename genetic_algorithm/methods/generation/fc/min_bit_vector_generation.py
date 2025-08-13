from genetic_algorithm.methods.generation.fc.base_generation import GenerationMethod


class MinBitVectorGeneration(GenerationMethod):

    def __init__(self, nn_architecture):
        super(MinBitVectorGeneration, self).__init__(nn_architecture=nn_architecture)
        self.nn_architecture = nn_architecture

    def generate(self):
        raise NotImplementedError()