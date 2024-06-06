from typing import Set
from experimaestro.launcherfinder import (
    HostRequirement,
    HostSpecification,
    CudaSpecification,
    CPUSpecification,
)
from experimaestro.launchers.slurm import SlurmLauncher, SlurmOptions
from experimaestro.connectors.local import LocalConnector


def find_launcher(requirements: HostRequirement, tags: Set[str] = set()):
    """Find a launcher"""

    if match := requirements.match(HostSpecification(cuda=[])):
        # No GPU: run directly
        return LocalConnector.instance()

    if match := requirements.match(
        HostSpecification(
            max_duration=100 * 3600,
            cpu=CPUSpecification(cores=32, memory=129 * (1024**3)),
            cuda=[CudaSpecification(memory=24 * (1024**3)) for _ in range(8)],
        )
    ):
        if len(match.requirement.cuda_gpus) > 0:
            return SlurmLauncher(
                connector=LocalConnector.instance(),
                options=SlurmOptions(gpus_per_node=len(match.requirement.cuda_gpus)),
            )

    # Could not find a host
    return None