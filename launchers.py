import math
from typing import Set
from experimaestro.launcherfinder import (
    HostRequirement,
    HostSpecification,
    CudaSpecification,
    CPUSpecification,
    MatchRequirement,
)
from experimaestro.launchers.slurm import SlurmLauncher, SlurmOptions

DURATION_100H = 100 * 3600
DURATION_20H = 20 * 3600
GIGA = 1024**3

MEM_PER_CPU = 2048 * 1024**2  # 2 Go per core


def slurm_launcher(match: MatchRequirement, **options):
    cpus_per_task = int(max(
        match.requirement.cpu.cores or 0,
        math.ceil((match.requirement.cpu.memory or 0) / MEM_PER_CPU),
    ))
    
    return SlurmLauncher(
        binpath="/gpfslocalsys/slurm/current/bin",
        options=SlurmOptions(
            cpus_per_task=str(cpus_per_task) if cpus_per_task > 0 else None,
            time=SlurmOptions.format_time(match.requirement.duration),
            **options,
        ),
    )


def find_launcher(requirements: HostRequirement, tags: Set[str] = set()):
    """Find a launcher"""

    # Partitions CPU
    # http://www.idris.fr/jean-zay/cpu/jean-zay-cpu-exec_partition_slurm.html

    # pre_post
    if match := requirements.match(
        HostSpecification(
            max_duration=DURATION_100H,
            cpu=CPUSpecification(cores=32, memory=129 * GIGA),
        )
    ):
        return slurm_launcher(
            match,
            account="cdt@cpu",
            partition="prepost",
        )

    # cpu_p1
    if match := requirements.match(
        HostSpecification(
            max_duration=DURATION_100H,
            cpu=CPUSpecification(cores=32, memory=129 * GIGA),
        )
    ):
        return slurm_launcher(
            match,
            account="cdt@cpu",
            qos="qos_cpu-t3",
            partition="cpu_p1" if match.requirement.duration <= DURATION_20H else "qos_cpu-t4",
        )

    # Partitions GPU (v100)
    # http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm.html
    for cuda_mem, constraint in [(16, "v100-16g"), (32, "v100-32g")]:
        if match := requirements.match(
            HostSpecification(
                max_duration=DURATION_100H,
                cpu=CPUSpecification(cores=32, memory=129 * GIGA),
                cuda=[CudaSpecification(memory=cuda_mem * GIGA) for _ in range(8)],
            )
        ):
            if len(match.requirement.cuda_gpus) > 0:
                return slurm_launcher(
                    match,
                    account="cdt@v100",
                    gpus_per_node=len(match.requirement.cuda_gpus),
                    constraint=constraint,
                    qos="qos_gpu-t3" if match.requirement.duration <= DURATION_20H else "qos_gpu-t4",
                )

    # Partitions GPU (a100)
    # http://www.idris.fr/jean-zay/gpu/jean-zay-gpu-exec_partition_slurm.html
    for cuda_mem, constraint in [(80, "a100")]: 
        if match := requirements.match(
            HostSpecification(
                max_duration=DURATION_20H,
                cpu=CPUSpecification(cores=32, memory=129 * GIGA),
                cuda=[CudaSpecification(memory=cuda_mem * GIGA) for _ in range(8)],
            )
        ):
            if len(match.requirement.cuda_gpus) > 0:
                return slurm_launcher(
                    match,
                    account="cdt@a100",
                    gpus_per_node=len(match.requirement.cuda_gpus),
                    constraint=constraint,
                    # qos="qos_gpu-t4"
                )

    # Could not find a host
    return None
