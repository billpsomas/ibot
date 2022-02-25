# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A script to run multinode training with submitit.
Almost copy-paste from https://github.com/facebookresearch/deit/blob/main/run_with_submitit.py
"""
import argparse
import os
import uuid
import traceback
from pathlib import Path

import eval_cls
import submitit


def parse_args():
    parser = argparse.ArgumentParser("Submitit for iBOT downstream tasks", parents=[eval_cls.get_args_parser()])
    parser.add_argument("--ngpus", default=2, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=2800, type=int, help="Duration of the job")

    parser.add_argument("--partition", default="gpu", type=str, help="Partition where to submit")
    parser.add_argument("--use_volta32", action='store_true', help="Big models? Use this")
    parser.add_argument('--comment', default="", type=str,
                        help='Comment to pass to scheduler, e.g. priority message')
    parser.add_argument('--mem_per_cpu', default=5, type=int, help='Needed memory per CPU')
    parser.add_argument('--mem_per_gpu', default=12, type=int, help='Needed memory per GPU')
    parser.add_argument('--ntasks', default=10, type=int, help='Total number of tasks to run')
    parser.add_argument('--account', default='pr011028_gpu', type=str, help='ARIS account to use')
    parser.add_argument('--name', default='mulnodeibotdown', type=str, help='Name of run')
    parser.add_argument('--mem', default=0, type=int, help='Total memory used')
    parser.add_argument('--cpus_per_task', default=10, type=int, help='CPUS to use')
    parser.add_argument('--submitit_slurm', default='/users/pa17/{user}/bill/logs/submitit/', type=str, help='Shared path to create submitit slurm file')
    return parser.parse_args()


def get_shared_folder(shared_path) -> Path:
    user = os.getenv("USER")
    if Path(shared_path.format(user=user)).is_dir():
        p = Path(shared_path.format(user=user))
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file(shared_path):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(shared_path)), exist_ok=True)
    init_file = get_shared_folder(shared_path) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        print(os.getcwd())
        from eval_cls import main as main_eval_cls

        self._setup_gpu_args()

        try:
            #print(os.system("ps aux | grep submitit | grep -v grep"))
            #print(os.system("ps aux | grep submitit | grep -v grep | awk '{print $2}' | sort -n"))        
            for checkpoint_key in self.args.checkpoint_key.split(','):
                print("Starting evaluating {}.".format(checkpoint_key))
                import copy
                args_copy = copy.deepcopy(self.args)
                args_copy.checkpoint_key = checkpoint_key
                main_eval_cls(args_copy)
        except:
            print(traceback.format_exc())
            os.system("for pid in $(ps aux | grep submitit | grep -v grep | awk '{print $2}' | sort -n); do kill -9 $pid; done")

        #try:
            #print(os.system("ps aux | grep submitit | grep -v grep"))
            #print(os.system("ps aux | grep submitit | grep -v grep | awk '{print $2}' | sort -n"))
            #main(args_copy)

        #except:
            #print(traceback.format_exc())
            #os.system("for pid in $(ps aux | grep submitit | grep -v grep | awk '{print $2}' | sort -n); do kill -9 $pid; done")

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(self.args.submitit_slurm).as_uri()
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.output_dir == "":
        args.output_dir = get_shared_folder(args.submitit_slurm) / "%j"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=args.output_dir, slurm_max_num_timeout=60)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    partition = args.partition
    kwargs = {}

    executor.update_parameters(
        slurm_exclusive=True,
        slurm_mem_per_cpu=args.mem_per_cpu,
        slurm_mem_per_gpu=args.mem_per_gpu,
        slurm_job_name=args.name,
        slurm_additional_parameters={"account": args.account, "ntasks": args.ntasks},
        mem_gb=args.mem,
        slurm_gres='gpu:' +str(num_gpus_per_node),
        slurm_ntasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=args.cpus_per_task,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        slurm_setup=["""export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)-ib""", """export GLOO_SOCKET_IFNAME=ib0""", """NODES=($( scontrol show hostname $SLURM_NODELIST | uniq ))""", """export NUM_NODES=${#NODES[@]}""", """WORKERS=$(printf '%s-ib:'${SLURM_NTASKS_PER_NODE}',' "${NODES[@]}" | sed 's/,$//')"""],
        **kwargs
    )
    
    executor.update_parameters(name=args.name)
    print(executor.parameters)
    args.dist_url = get_init_file(args.submitit_slurm).as_uri()

    trainer = Trainer(args)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at: {args.output_dir}")


if __name__ == "__main__":
    main()
