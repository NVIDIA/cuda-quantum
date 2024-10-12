import cudaq
import os

remote_config = os.environ.get("FERMIONIQ_REMOTE_CONFIG_ID", "")
project_id = os.environ.get("FERMIONIQ_PROJECT_ID", "")

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
cudaq.set_target("fermioniq", **{
    "remote-config": remote_config,
    "project-id": project_id
})


# Create the kernel we'd like to execute on Fermioniq.
@cudaq.kernel
def kernel():
    qvector = cudaq.qvector(2)
    h(qvector[0])
    x.ctrl(qvector[0], qvector[1])
    mz(qvector[0])
    mz(qvector[1])


# Submit to Fermioniq's endpoint and confirm the program is valid.
result = cudaq.sample(kernel)

print(result)
