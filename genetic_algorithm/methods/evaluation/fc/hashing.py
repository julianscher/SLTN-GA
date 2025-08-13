import xxhash


class FitnessHashTable:
    def __init__(self):
        self.table = {}

    def _calculate_position(self, bit_vector):
        h = xxhash.xxh64()
        h.update(bit_vector)
        hash_int = h.intdigest()
        h.reset()
        return hash_int

    def store_fitness(self, bit_vector, fitness_value):
        position = self._calculate_position(bit_vector)
        self.table[position] = fitness_value

    def retrieve_fitness(self, bit_vector):
        position = self._calculate_position(bit_vector)
        return self.table.get(position, None)

    def has_evaluated(self, bit_vector):
        position = self._calculate_position(bit_vector)
        return position in self.table
