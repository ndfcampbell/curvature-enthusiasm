# run_batch.py
import os
import sys
import csv
import time
import math
import signal
import argparse
import threading
import queue
import subprocess
from pathlib import Path

MAX_RETRIES_DEFAULT = 3

stop_event = threading.Event()  # set on SIGINT to stop launching new work

def tee_stream(stream, *writers):
    for line in iter(stream.readline, ''):
        for w in writers:
            w.write(line)
            w.flush()
    stream.close()

def run_calc_interpolation(fileA, fileB, config_file, gpu_id, log_dir, mem_frac, timeout_s):
    """
    Launch calc_interpolation.py on a specific GPU.
    Returns True on success, False on failure.
    """
    task_name = f"{Path(fileA).stem}__{Path(fileB).stem}"
    log_path = Path(log_dir) / f"gpu{gpu_id}_{task_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, 'calc_interpolation.py',
        '--start_pose', fileA,
        '--end_pose', fileB,
        '--config', config_file,
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Set XLA memory policy if provided (calc_interpolation should use `setdefault`)
    if mem_frac is not None:
        env["XLA_PYTHON_CLIENT_MEM_FRACTION"] = f"{mem_frac:.2f}"
        env.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)  # prefer fraction over preallocate

    # Strongly request GPU platform
    env.setdefault("JAX_PLATFORMS", "gpu")
    env.setdefault("JAX_PLATFORM_NAME", "gpu")  # legacy

    cwd = Path(__file__).resolve().parent
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write(f"[LAUNCH] GPU {gpu_id} | {fileA} -> {fileB} | config={config_file}\n")
        lf.flush()

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )

            # Tee stdout to console and log file
            t = threading.Thread(target=tee_stream, args=(proc.stdout, sys.stdout, lf), daemon=True)
            t.start()

            try:
                ret = proc.wait(timeout=timeout_s) if timeout_s else proc.wait()
            except subprocess.TimeoutExpired:
                lf.write(f"\n[TIMEOUT] Killing task after {timeout_s}s\n")
                lf.flush()
                proc.kill()
                proc.wait()
                return False

            if ret != 0:
                lf.write(f"\n[ERROR] Exit code {ret}\n")
                lf.flush()
                return False

            lf.write("\n[SUCCESS] Completed\n")
            lf.flush()
            return True

        except Exception as e:
            lf.write(f"\n[EXCEPTION] {e}\n")
            lf.flush()
            return False

def worker(task_queue, gpu_id, mem_frac, timeout_s, max_retries):
    while not stop_event.is_set():
        try:
            fileA, fileB, config_file, log_dir = task_queue.get(block=False)
        except queue.Empty:
            break

        # Retry with exponential backoff
        delay = 1.0
        for attempt in range(1, max_retries + 1):
            ok = run_calc_interpolation(fileA, fileB, config_file, gpu_id, log_dir, mem_frac, timeout_s)
            if ok:
                break
            if attempt < max_retries:
                print(f"[GPU {gpu_id}] Retry {attempt}/{max_retries-1} for {fileA} -> {fileB} after {delay:.1f}s")
                time.sleep(delay)
                delay = min(delay * 2, 30.0)  # cap backoff
        task_queue.task_done()

def parse_batch_file(batch_file: Path):
    """
    Batch file format:
      each line: <pairs_csv_path>,<config_yaml_path>
    pairs CSV format:
      each line: <start_pose>,<end_pose>
    """
    jobs = []
    with open(batch_file, "r", encoding="utf-8") as bf:
        reader = csv.reader((ln for ln in bf if ln.strip() and not ln.strip().startswith("#")))
        for row in reader:
            if len(row) < 2:
                raise ValueError(f"Bad line in batch file {batch_file}: {row}")
            pairs_file, config_file = map(str.strip, row[:2])
            pairs_file = Path(pairs_file)
            config_file = Path(config_file)
            if not pairs_file.exists():
                raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
            if not config_file.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")

            with open(pairs_file, "r", encoding="utf-8") as pf:
                pairs_reader = csv.reader((ln for ln in pf if ln.strip() and not ln.strip().startswith("#")))
                for prow in pairs_reader:
                    if len(prow) < 2:
                        raise ValueError(f"Bad line in pairs file {pairs_file}: {prow}")
                    fileA, fileB = map(str.strip, prow[:2])
                    jobs.append((fileA, fileB, str(config_file)))
    return jobs

def run_testing(batch_file, gpu_list, max_jobs_per_gpu, log_dir, mem_frac_arg, timeout_s, max_retries):
    # If user didn’t pass mem fraction, estimate safe default per job on a GPU
    # e.g., reserve ~85% of the card divided by concurrent jobs on that card.
    mem_frac = None
    if mem_frac_arg is None:
        if max_jobs_per_gpu > 0:
            mem_frac = max(0.1, 0.85 / float(max_jobs_per_gpu))
    else:
        mem_frac = float(mem_frac_arg)

    jobs = parse_batch_file(Path(batch_file))
    task_queue = queue.Queue()
    for fileA, fileB, config_file in jobs:
        task_queue.put((fileA, fileB, config_file, log_dir))

    print(f"Queued {task_queue.qsize()} tasks across GPUs {gpu_list} "
          f"with up to {max_jobs_per_gpu} job(s)/GPU. "
          f"Mem fraction per job: {mem_frac if mem_frac is not None else '(unset)'}")

    total_workers = len(gpu_list) * max_jobs_per_gpu
    threads = []

    for i in range(total_workers):
        gpu_id = gpu_list[i % len(gpu_list)]
        t = threading.Thread(target=worker, args=(task_queue, gpu_id, mem_frac, timeout_s, max_retries), daemon=True)
        t.start()
        threads.append(t)

    # SIGINT handler: stop launching new work, let current finish/timeout
    def on_sigint(sig, frame):
        print("\n[CTRL-C] Stopping new tasks; waiting for running tasks to finish...")
        stop_event.set()
    signal.signal(signal.SIGINT, on_sigint)

    # Wait
    for t in threads:
        t.join()

    print("All tasks completed (or stopped).")

# python run_batch.py batches/my_batch.csv --gpus 0,1 --jobs-per-gpu 2 --timeout 18000
# logs written to logs/gpu{n}_{start}__{end}.log

# each line: <pairs_csv>,<config_yaml>
# pairs/run1_pairs.csv,configs/mano_config.yaml
# pairs/run2_pairs.csv,configs/mano_config.yaml

# each line: <start_pose>,<end_pose>
# 01_01r,01_02r
# 01_02r,01_03r
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Batch interpolation across multiple GPUs.")
    p.add_argument("batch_file", help="CSV with lines: <pairs_csv>,<config_yaml>")
    p.add_argument("--gpus", required=True, help="Comma-separated GPU IDs to use, e.g. 0,1,2,3")
    p.add_argument("--jobs-per-gpu", type=int, default=1, help="Concurrent jobs per GPU (default: 1)")
    p.add_argument("--log-dir", default="logs", help="Directory to write per-task logs")
    p.add_argument("--mem-frac", type=float, default=None,
                   help="XLA_PYTHON_CLIENT_MEM_FRACTION per job; overrides auto (e.g., 0.30)")
    p.add_argument("--timeout", type=int, default=0,
                   help="Per-task timeout seconds (0 disables)")
    p.add_argument("--retries", type=int, default=MAX_RETRIES_DEFAULT,
                   help=f"Retries per task (default: {MAX_RETRIES_DEFAULT})")
    args = p.parse_args()

    gpu_list = [int(x.strip()) for x in args.gpus.split(",") if x.strip() != ""]
    run_testing(args.batch_file, gpu_list, args.jobs_per_gpu, args.log_dir,
                args.mem_frac, args.timeout if args.timeout > 0 else None, args.retries)



