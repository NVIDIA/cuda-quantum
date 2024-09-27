import cudaq

# You only have to set the target once! No need to redefine it
# for every execution call on your kernel.
cudaq.set_target("fermioniq", 
                 **{"remote-config": "95ab08ab-7b8c-480a-914b-c1206dd6f373",
                    "project-id": "943977db-7264-4b66-addf-c9d6085d9d8f"})


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
