
class Logger:
    def __init__(self, routines, log_every, log_path):
        self.tmp_log = {}
        self.routines = routines
        self.log_every = log_every
        self.log_path = log_path

        if log_path:
            self.tmp_log["log_path"] = log_path

        print("Log path:", log_path)

    def apply_init_routines(self, ga):
        # Routines to fill tmp_log with entries that are important for later routines
        for routine in self.routines["init"]:
            routine(ga, self.tmp_log)

    def apply_evol_routines(self, ga):
        if ga.population.age % self.log_every == 0:
            for routine in self.routines["evol"]:
                routine(ga, self.tmp_log)

    def apply_final_routines(self, ga):
        for routine in self.routines["final"]:
            routine(ga, self.tmp_log)