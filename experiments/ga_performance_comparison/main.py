import os
import sys
import time
import traceback
import yaml

from utilities.helper_functions import get_results_path, set_seed, get_configs_path
from args import args
from genetic_algorithm.worker import Worker

results_path = f"{get_results_path()}/ga_performance_results"

# Experiment contents overview
NN_architectures_binary = [[2, 20, 2], [2, 75, 2], [2, 100, 2], [2, 50, 50, 2]]
NN_architectures_digits = [[64, 20, 10], [64, 75, 10], [64, 100, 10], [64, 50, 50, 10]]
NN_architectures_digits_binary = [[64, 20, 2], [64, 75, 2], [64, 100, 2], [64, 50, 50, 2]]

if not os.path.isdir(results_path):
    os.mkdir(results_path)


job_tasks_binary = {"tasks0": {i: [2, 20, 2] for i in range(50)},
                    "tasks1": {i: [2, 75, 2] for i in range(50)},
                    "tasks2": {i: [2, 100, 2] for i in range(50)},
                    "tasks3": {i: [2, 50, 50, 2] for i in range(50)}}

job_tasks_digits = {"tasks0": {i: [64, 20, 10] for i in range(50)},
                    "tasks1": {i: [64, 75, 10] for i in range(50)},
                    "tasks2": {i: [64, 100, 10] for i in range(50)},
                    "tasks3": {i: [64, 50, 50, 10] for i in range(50)}}

job_tasks_digits_binary = {"tasks0": {i: [64, 20, 2] for i in range(50)},
                           "tasks1": {i: [64, 75, 2] for i in range(50)},
                           "tasks2": {i: [64, 100, 2] for i in range(50)},
                           "tasks3": {i: [64, 50, 50, 2] for i in range(50)}}


def run_job_with_checkpointing(job_id, selected_tasks, tasks, dataset_args, experiment_directory):
    # Determine starting task (for restarts)
    continue_at = _read_interrupt_checkpoint(job_id, selected_tasks[0], experiment_directory)
    task_keys = selected_tasks[selected_tasks.index(continue_at):]
    sub_tasks = {k: tasks.get(k, None) for k in task_keys}

    #load_config()
    base_yaml = getattr(args, 'config', None)
    if not base_yaml or not os.path.isfile(base_yaml):
        raise FileNotFoundError("Base YAML not found. Ensure args.config points to your YAML.")

    for counter, NN_architecture in sub_tasks.items():
        _write_interrupt_checkpoint(job_id, counter, experiment_directory)

        job_name = f"{experiment_directory}/job_{job_id}_{counter}"
        job_root = _job_dir(job_id, counter, experiment_directory)

        if _is_task_done(job_id, counter, experiment_directory):
            # Already finished successfully; skip
            continue

        # Compose a per-task YAML that overrides just the job name & architecture
        task_yaml = os.path.join(job_root, "task.config.yaml")
        overrides = {
            "name": job_name,
            "architecture": NN_architecture,
            "dataset_args": dataset_args,
            "results_path": results_path,
        }
        _compose_task_yaml(base_yaml, overrides, task_yaml)

        run_job_with_timeout(name=job_name, task_yaml_path=task_yaml)

def run_job_with_timeout(name, task_yaml_path):
    """ This method only works with Unix systems """

    """def handler(signum, frame):
        print("Forever is over!")
        raise Exception("end of time")

    signal.signal(signal.SIGALRM, handler)
    # Give every task 1800 sec to finish
    signal.alarm(1800)"""

    start = time.time()
    print(f"Starting experiment {name} at {start:.2f}")

    log_dir = os.path.join(results_path, name)

    try:
        _call_worker_with_config(task_yaml_path)

        runtime = time.time() - start
        print(f"Time to run: {runtime:.2f} seconds")
        _store_runtime(runtime, log_dir)

        # Mark the task as completed. Restarts can skip it
        with open(os.path.join(os.path.dirname(task_yaml_path), "DONE"), "w") as f:
            f.write("ok\n")
    except Exception:
        os.makedirs(log_dir, exist_ok=True)

        with open(os.path.join(log_dir, "error_log.txt"), "w") as f:
            print(traceback.format_exc(), file=f)

        _store_runtime(1800.0, log_dir)


##### helper functions #####

def _interrupt_path(job_id, experiment_directory):
    return os.path.join(results_path, experiment_directory, "interrupt_logs", f"interrupt_job{job_id}.txt")

def _job_dir(job_id, counter, experiment_directory):
    return os.path.join(results_path, experiment_directory, f"job_{job_id}_{counter}")

def _read_interrupt_checkpoint(job_id, default_task, experiment_directory):
    p = _interrupt_path(job_id, experiment_directory)
    if os.path.isfile(p):
        try:
            with open(p, "r") as f:
                line = f.readline().strip()
                return int(line)
        except Exception:
            pass
    return default_task

def _write_interrupt_checkpoint(job_id, counter, experiment_directory):
    p = _interrupt_path(job_id, experiment_directory)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        print(counter, file=f)

def _is_task_done(job_id, counter, experiment_directory):
    # We mark a task done by writing a sentinel file into the job directory.
    return os.path.isfile(os.path.join(_job_dir(job_id, counter, experiment_directory), "DONE"))

def _mark_task_done(job_id, counter, experiment_directory):
    os.makedirs(_job_dir(job_id, counter, experiment_directory), exist_ok=True)
    with open(os.path.join(_job_dir(job_id, counter, experiment_directory), "DONE"), "w") as f:
        f.write("ok\n")

def _store_runtime(seconds, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "runtime.txt"), "w") as f:
        f.write(f"{seconds:.3f}\n")

def _deep_update(dst, src):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _compose_task_yaml(base_yaml_path, overrides_dict, out_yaml_path):
    with open(base_yaml_path, "r") as f:
        base = yaml.safe_load(f) or {}
    merged = _deep_update(base, overrides_dict)
    os.makedirs(os.path.dirname(out_yaml_path), exist_ok=True)
    with open(out_yaml_path, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)
    return out_yaml_path

def _call_worker_with_config(task_yaml_path):
    # Point the worker at our task-specific YAML and run in-process.
    prev = getattr(args, "config", None)
    try:
        args.config = task_yaml_path
        set_seed(getattr(args, "seed", 0), getattr(args, "only_model", False))
        w = Worker()
        w.run()
    finally:
        # Restore original args.config to avoid side-effects for subsequent calls
        if prev is None:
            if hasattr(args, "config"):
                delattr(args, "config")
        else:
            args.config = prev

def _create_experiment_directory_with_interrupt_logs(target_directory, job_count):
    if not os.path.isdir(f"{results_path}/{target_directory}"):
        os.mkdir(f"{results_path}/{target_directory}")
        os.mkdir(f"{results_path}/{target_directory}/interrupt_logs")
        # create interrupt logs
        for i in range(job_count):
            open(f"{results_path}/{target_directory}/interrupt_logs/interrupt_job{i}.txt", "x")


##### conduction function #####
def run_tasks_job(job_id, selected_tasks, job_tasks, dataset_args, experiment_directory):
    run_job_with_checkpointing(job_id, selected_tasks, job_tasks, dataset_args, experiment_directory)


##### Experiment settings #####
def _cfg(rel_path: str) -> str:
    return f"{get_configs_path()}/ga/experiment_configs/{rel_path}.yml"

# === make_moons ===
def get_GA_moons():
    # You have to create the directory make_moons manually in GA_performance_results
    experiment_directory = "make_moons/GA_evolutions_top_speed"
    job_tasks = job_tasks_binary
    args.config = f"{get_configs_path()}/ga/experiment_configs/moons_experiment/top_speed.yml"
    dataset_args = {}
    return experiment_directory, job_tasks, dataset_args


def get_GA_adaptive_AB_moons():
    # You have to create the directory make_moons manually in GA_performance_results
    experiment_directory = "make_moons/GA_evolutions_max_acc_bound"
    job_tasks = job_tasks_binary
    args.config = _cfg("moons_experiment/max_acc_bound")
    dataset_args = {}
    return experiment_directory, job_tasks, dataset_args


# === make_circles ===
def get_GA_circles():
    # You have to create the directory make_circles manually in GA_performance_results
    experiment_directory = "make_circles/GA_evolutions_top_speed"
    job_tasks = job_tasks_binary
    args.config = _cfg("circles_experiment/top_speed")
    dataset_args = {}
    return experiment_directory, job_tasks, dataset_args


def get_GA_adaptive_AB_circles():
    # You have to create the directory make_circles manually in GA_performance_results
    experiment_directory = "make_circles/GA_evolutions_max_acc_bound"
    job_tasks = job_tasks_binary
    args.config = _cfg("circles_experiment/max_acc_bound")
    dataset_args = {}
    return experiment_directory, job_tasks, dataset_args


# === load_digits (10 classes) ===
def get_GA_digits_old():
    # You have to create the directory load_digits manually in GA_performance_results
    experiment_directory = "load_digits_old/GA_evolutions_top_speed"
    job_tasks = job_tasks_digits
    args.config = _cfg("digits_experiment_old/top_speed")
    dataset_args = {"classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "with_validation": False}
    return experiment_directory, job_tasks, dataset_args


def get_GA_static_AB_digits_old():
    # You have to create the directory load_digits manually in GA_performance_results
    experiment_directory = "load_digits_old/GA_evolutions_max_acc"
    job_tasks = job_tasks_digits
    args.config = _cfg("digits_experiment_old/max_acc")
    dataset_args = {"classes": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "with_validation": False}
    return experiment_directory, job_tasks, dataset_args


# === load_digits_binary (2 classes) ===
def get_GA_binary_digits_old():
    # You have to create the directory load_digits_binary manually in GA_performance_results
    experiment_directory = "load_digits_binary/GA_evolutions_top_speed"
    job_tasks = job_tasks_digits_binary
    args.config = _cfg("digits_experiment_old/top_speed")
    dataset_args = {"classes": [0, 1], "with_validation": False}
    return experiment_directory, job_tasks, dataset_args


def get_GA_static_AB_binary_digits_old():
    # You have to create the directory load_digits_binary manually in GA_performance_results
    experiment_directory = "load_digits_binary/GA_evolutions_max_acc_bound"
    job_tasks = job_tasks_digits_binary
    args.config = _cfg("digits_experiment_old/max_acc_bound")
    dataset_args = {"classes": [0, 1], "with_validation": False}
    return experiment_directory, job_tasks, dataset_args


# === load_digits_fewer_classes ===
def get_GA_digits(n_classes: int, norm=False):
    # You have to create the directory load_digits_fewer_classes manually in GA_performance_results
    digits_datasets = {2: "load_digits_binary", 3: "load_digits_ternary", 4: "load_digits_quaternary",
                       5: "load_digits_quinary"}
    selected_dataset = digits_datasets[n_classes]

    job_tasks = {"tasks0": {i: [64, 75, n_classes] for i in range(50)}}

    if norm:
        experiment_directory = f"load_digits_normalized/{selected_dataset}"
        args.config = _cfg("digits_experiment/top_speed_normalized")
    else:
        experiment_directory = f"load_digits/{selected_dataset}"
        args.config = _cfg("digits_experiment/top_speed")
    dataset_args = {"classes": list(range(n_classes)), "with_validation": False}
    return experiment_directory, job_tasks, dataset_args


# === make_blobs ===
def get_GA_blobs(n_classes: int, norm=False):
    # You have to create the directory make_blobs manually in GA_performance_results
    blob_datasets = {2: "make_two_blobs", 3: "make_three_blobs", 4: "make_four_blobs", 5: "make_five_blobs",
                     6: "make_six_blobs", 7: "make_seven_blobs", 8: "make_eight_blobs", 9: "make_nine_blobs",
                     10: "make_ten_blobs"}
    selected_dataset = blob_datasets[n_classes]

    job_tasks = {"tasks0": {i: [2, 100, n_classes] for i in range(50)}}

    if norm:
        experiment_directory = f"make_blobs_normalized/{selected_dataset}"
        args.config = _cfg(f"blobs_experiment/top_speed_normalized")
    else:
        experiment_directory = f"make_blobs/{selected_dataset}"
        args.config = _cfg(f"blobs_experiment/top_speed")
    dataset_args = {"centers": n_classes, "with_validation": False}
    return experiment_directory, job_tasks, dataset_args


if __name__ == "__main__":
    sys.setrecursionlimit(50000) # To prevent RecursionError
    experiment_directory, job_tasks, dataset_args = get_GA_moons()
    run_tasks_job(0, range(25), job_tasks["tasks0"], dataset_args, experiment_directory)
    # run_tasks_job(0, range(25, 50), job_tasks["tasks0"], dataset_args, experiment_directory)
    # run_tasks_job(1, range(25), job_tasks["tasks1"], dataset_args, experiment_directory)
    # run_tasks_job(1, range(25, 50), job_tasks["tasks1"], dataset_args, experiment_directory)
    # run_tasks_job(2, range(25), job_tasks["tasks2"], dataset_args, experiment_directory)
    # run_tasks_job(2, range(25, 50), job_tasks["tasks2"], dataset_args, experiment_directory)
    # run_tasks_job(3, range(25), job_tasks["tasks3"], dataset_args, experiment_directory)
    # run_tasks_job(3, range(25, 50), job_tasks["tasks3"], dataset_args, experiment_directory)
