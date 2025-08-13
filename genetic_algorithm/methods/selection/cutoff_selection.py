from genetic_algorithm.methods.selection.base_selection import SelectionMethod

class CutoffSelection(SelectionMethod):
    """Selects the top individuals by fitness cutoff."""

    def select(self, population, size):
        return population[0:size]