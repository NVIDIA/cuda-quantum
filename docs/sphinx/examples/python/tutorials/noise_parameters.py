import cudaq
import numpy as np

### Hidden qubit parameters
T1 = 1000 #ns
T2 = 200 #ns
detuning_rate = np.pi/250

### Define Kraus Channels

def error_prob(t,T12):
        error = 1-np.exp(-t/T12) #nanoseconds
        return error

def t1_kraus(t,T1):

    t1_decay = error_prob(t,T1)
    kraus_0 = np.array([[1.0,0.0],[0.0,np.sqrt(1-t1_decay)]],dtype=np.complex128,order='F')
    kraus_1 = np.array([[0.0,np.sqrt(t1_decay)],[0.0,0.0]],dtype=np.complex128,order='F')
    kraus_op = cudaq.KrausChannel([kraus_0,kraus_1])
    return kraus_op

def coherent_z_kraus(angle):
    kraus_0 = np.array([[np.exp(-1j*angle/2),0.0],[0.0,np.exp(1j*angle/2)]],dtype="complex128",order="F")
    kraus_op = cudaq.KrausChannel([kraus_0])
    return kraus_op

def t2_kraus(t,T2):
    
    t2_decay = error_prob(t,T2)
    kraus_0 = np.array([[1.0,0.0],[0.0,np.sqrt(1-t2_decay)]],dtype=np.complex128,order='F')
    kraus_1 = np.array([[0.0,0.0],[0.0,np.sqrt(t2_decay)]],dtype=np.complex128,order='F')
    kraus_op = cudaq.KrausChannel([kraus_0,kraus_1])
    return kraus_op

### Define Noise model and gates
    
def hidden_noise_model(t):
    #initialize the noise model
    noise_model=cudaq.NoiseModel()

    ##add noise to r1 "delay"
    noise_model.add_channel("r1",[0],t1_kraus(t,T1))
    noise_model.add_channel("r1",[0],t2_kraus(t,T2))
    noise_model.add_channel("r1",[0],coherent_z_kraus(t*detuning_rate))

    ##add noise to rx
    noise_model.add_channel("rx",[0],t1_kraus(t,T1))
    noise_model.add_channel("rx",[0],t2_kraus(t,T2))
    noise_model.add_channel("rx",[0],coherent_z_kraus(t*detuning_rate))

    ##add noise to rz 
    noise_model.add_channel("rz",[0],t1_kraus(t,T1))
    noise_model.add_channel("rz",[0],t2_kraus(t,T2))
    noise_model.add_channel("rz",[0],coherent_z_kraus(t*detuning_rate))

    return noise_model

## define some custom gates

def Id(self, time, qubit) -> None:
    self.r1(0.0, qubit)

def custom_rx(self, time, amp, qubit):
     ##hidden frequency
     f = 5.3012314 #GHz
     amp_adjust = 0.1
     angle = 2*np.pi*time*amp*f*amp_adjust
     self.rx(angle,qubit)