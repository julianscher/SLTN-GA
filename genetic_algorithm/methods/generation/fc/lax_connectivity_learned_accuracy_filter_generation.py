import math
import time

import numpy as np

from genetic_algorithm.methods.generation.fc.base_generation import GenerationMethod
from genetic_algorithm.methods.generation.fc.lax_connectivity_generation import LaxConnectivityGeneration
from utilities.fc.fc_bit_vector_operations import check_if_bit_vector_passes_accuracy_filter
from utilities.ga_utils import randi


class LaxConnectivityLearnedAccuracyFilterGeneration(GenerationMethod):

    def __init__(self, nn_architecture, device, bound, model, train_loader, max_reas_gen_time, max_giv_gen_time, iteration_time=None, num_rec=None):
        super(LaxConnectivityLearnedAccuracyFilterGeneration, self).__init__(nn_architecture=nn_architecture, device=device,
                                                                             bound=bound, model=model, max_reas_gen_time=max_reas_gen_time, max_giv_gen_time=max_giv_gen_time,
                                                                             iteration_time=iteration_time, num_rec=num_rec)
        self.nn_architecture = nn_architecture
        self.device = device
        self.bound = bound
        self.model = model
        self.train_loader = train_loader
        self.iteration_time = iteration_time
        self.max_reas_gen_time = max_reas_gen_time
        self.max_giv_gen_time = max_giv_gen_time
        self.num_rec = num_rec

    def generate(self):
        """
            Two notes on this method:
            - Sometimes the algorithm takes some time with evaluating the accuracy of the subnetwork in NN
            - If not generation_arg for iteration_time is given, the algorithm won't be deterministic
            - TODO: rename all references and instances of this method to generate_lax_connectivity_learned_accuracy_bound
            - TODO: In its current form its only applicable to binary classification problems
            """
        dimensionality = self.model.number_of_parameters_to_be_masked()

        # measure time for creating one bit_vector
        start_time = time.time()
        # Create random bit_vector
        random_bit_vector = np.array([randi(min=0, max=2) for _ in range(dimensionality)])

        # If bound was decreased so much, that it passed 0.51 accept every bit_vector that has an input_output_path
        if self.bound == 0.51:
            return LaxConnectivityGeneration(self.nn_architecture).generate()

        # Check if bit_vector's accuracy has at least the boundary value
        if check_if_bit_vector_passes_accuracy_filter(random_bit_vector, self.bound, self.model, self.train_loader,
                                                      self.device):
            return random_bit_vector
        else:
            iteration_time = time.time() - start_time
            if self.iteration_time:
                iteration_time = self.iteration_time

            maximum_reasonable_generation_time = self.max_reas_gen_time
            maximum_given_generation_time = self.max_giv_gen_time
            unsuccessful_tries_upper_bound = math.floor(maximum_reasonable_generation_time / iteration_time)
            max_unsuccessful_tries_upper_bound = math.floor(maximum_given_generation_time / iteration_time)
            upper_bound_distance = max_unsuccessful_tries_upper_bound - unsuccessful_tries_upper_bound

            # Handle edge cases
            if unsuccessful_tries_upper_bound < 1 or max_unsuccessful_tries_upper_bound < 1 or upper_bound_distance == 0:
                return LaxConnectivityGeneration(self.nn_architecture).generate()

            number_of_unsuccessful_tries = 0
            while number_of_unsuccessful_tries <= max_unsuccessful_tries_upper_bound:
                # Create random bit_vector
                random_bit_vector = np.array([randi(min=0, max=2) for _ in range(dimensionality)])

                # Check if bit_vector's accuracy has at least the boundary value
                if check_if_bit_vector_passes_accuracy_filter(random_bit_vector, self.bound, self.model, self.train_loader,
                                                                 self.device):
                    if number_of_unsuccessful_tries > unsuccessful_tries_upper_bound:
                        # standardized distance of number_of_unsuccessful_tries to max_unsuccessful_tries_upper_bound
                        unsuccessful_tries_distance = number_of_unsuccessful_tries / upper_bound_distance

                        # Decrease accuracy bound while not passing the minimum accuracy
                        # to ensure there are input_output_paths
                        decreased_bound = max(self.bound - self._accuracy_bound_change(unsuccessful_tries_distance), 0.51)

                        # Update bound for successive generations
                        self.bound = decreased_bound

                    return random_bit_vector

                number_of_unsuccessful_tries += 1

            # If no valid bit_vector could be generated in max_unsuccessful_tries_upper_bound, decrease accuracy bound
            # with the highest change and call generate_lax_connectivity_learned_accuracy_filter recursively with the new
            # decreased_bound

            # num_rec is a value that slows down the change with increasing number of recursive calls
            if not self.num_rec:
                self.num_rec = 1

            num_rec = self.num_rec

            decreased_bound = max(self.bound - math.pow(0.1, num_rec), 0.51)

            # Update bound for recursive call and update num_rec
            self.bound = decreased_bound
            self.num_rec += 0.15

            return LaxConnectivityLearnedAccuracyFilterGeneration(self.nn_architecture, self.device, self.bound, self.model, self.train_loader, iteration_time, self.max_reas_gen_time, self.max_giv_gen_time, self.num_rec).generate()

    @staticmethod
    def _accuracy_bound_change(dist):
        return 1 / 10 * (0.051 + 0.93 * math.exp(-9.36 * dist))
