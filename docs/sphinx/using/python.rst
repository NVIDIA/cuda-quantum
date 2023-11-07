.. meta::
   :thebe-kernel: ipython

CUDA Quantum in Python
======================

Welcome to CUDA Quantum!
This is a introduction by example for using CUDA Quantum in Python. 

.. raw:: html

    <div id="interactive-frame">
      <iframe id="jupyterlab-vqe" src="../_static/cuda_quantum_icon.svg" name="JupyterLab"></iframe>
    </div>
    <button id="jupyter-lab-launch" title="Open JupyterLab" class="jupyterlab-button" onclick="launchJupyterLab('jupyterlab-vqe', 'tutorial/vqe.ipynb')">
       Open JupyterLab
    </button>

Introduction
--------------------------------

We're going to take a look at how to construct quantum programs through CUDA Quantum's `Kernel` API.

.. raw:: html

   <button title="Launch Interactive" class="thebelab-button thebe-launch-button" onclick="initThebe()">
      Launch Interactive
   </button>

When you create a `Kernel` and invoke its methods, a quantum program is constructed that can then be executed by calling, for example, `cudaq::sample`. Let's take a closer look!

.. code-block::
   :class: thebe, thebe-init
   
   %pip install cuda-quantum

.. container:: thebe

   .. literalinclude:: ../examples/python/intro.py
      :language: python

Bernstein-Vazirani
--------------------------------

Bernstein Vazirani is an algorithm for finding the bitstring encoded in a given function. 

For the original source of this algorithm, see 
`this publication <https://epubs.siam.org/doi/10.1137/S0097539796300921>`__.

In this example, we generate a random bitstring, encode it into an inner-product oracle, then we simulate the kernel and return the most probable bitstring from its execution.

If all goes well, the state measured with the highest probability should be our hidden bitstring!

.. literalinclude:: ../examples/python/bernstein_vazirani.py
   :language: python

Variational Quantum Eigensolver
--------------------------------

Let's take a look at how we can use CUDA Quantum's built-in `vqe` module to run our own custom VQE routines! Given a parameterized quantum kernel, a system spin Hamiltonian, and one of CUDA Quantum's optimizers, `cudaq.vqe` will find and return the optimal set of parameters that minimize the energy, <Z>, of the system.

.. literalinclude:: ../examples/python/simple_vqe.py
   :language: python

Let's look at a more advanced examples.

As an alternative to `cudaq.vqe`, we can also use the `cudaq.optimizers` suite on its own to write custom variational algorithm routines. Much of this can be slightly modified for use with third-party optimizers, such as `scipy`.

.. literalinclude:: ../examples/python/advanced_vqe.py
   :language: python

Quantum Approximate Optimization Algorithm
-------------------------------------------

Let's now see how we can leverage the VQE algorithm to compute the Max-Cut of a rectangular graph.

.. literalinclude:: ../examples/python/qaoa_maxcut.py
   :language: python

Noisy Simulation
-----------------

CUDA Quantum makes it simple to model noise within the simulation of your quantum program.
Let's take a look at the various built-in noise models we support, before concluding with a brief example of a custom noise model constructed from user-defined Kraus Operators.

The following code illustrates how to run a simulation with depolarization noise.

.. literalinclude:: ../examples/python/noise_depolarization.py
   :language: python

The following code illustrates how to run a simulation with amplitude damping noise.

.. literalinclude:: ../examples/python/noise_amplitude_damping.py
   :language: python

The following code illustrates how to run a simulation with bit-flip noise.

.. literalinclude:: ../examples/python/noise_bit_flip.py
   :language: python

The following code illustrates how to run a simulation with phase-flip noise.

.. literalinclude:: ../examples/python/noise_phase_flip.py
   :language: python

The following code illustrates how to run a simulation with a custom noise model.

.. literalinclude:: ../examples/python/noise_kraus_operator.py
   :language: python

.. _python-examples-for-hardware-providers:

Using Quantum Hardware Providers
-----------------------------------

CUDA Quantum contains support for using a set of hardware providers. 
For more information about executing quantum kernels on different hardware backends, please take a look at :doc:`hardware`.

The following code illustrates how run kernels on Quantinuum's backends.

.. literalinclude:: ../examples/python/providers/quantinuum.py
   :language: python

The following code illustrates how run kernels on IonQ's backends.

.. literalinclude:: ../examples/python/providers/ionq.py
   :language: python


Jupyterlab
-----------------------------------

Launch the dev container with

.. code-block:: console

   docker run -it --gpus all -p 8888:8888 -p 443:443 -p 3000:3000 --name cudaq-jupyter ghcr.io/nvidia/cuda-quantum:preview
   #ip_addr=`docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' cudaq-jupyter`

Within the dev container, execute:

setupProxy.js:
   const express = require('express');
   const { createProxyMiddleware } = require('http-proxy-middleware');
   const app = express();
   app.use(
   '/',
   createProxyMiddleware({
      target: 'http://localhost:8080',
      changeOrigin: true
   })
   );
   app.listen(8888);

Testing:
   const WebSocket = require('ws');

   let ws = new WebSocket("ws://localhost:8888", [], { })
   ws.onopen = () => {
   console.log('Connection opened ok! ')
   }

sudo apt-get install xz-utils 
wget https://nodejs.org/dist/v20.9.0/node-v20.9.0-linux-x64.tar.xz
tar -xf node-v20.9.0-linux-x64.tar.xz && rm node-v20.9.0-linux-x64.tar.xz
sudo mv node-v20.9.0-linux-x64/ /usr/local/node-20.9
export PATH="$PATH:/usr/local/node-20.9/bin"
npm install http-proxy-middleware express
node setupProxy.js &

jupyter-lab --no-browser --ip=* --port=8888 --ServerApp.tornado_settings='{"headers":{"Content-Security-Policy": "frame-ancestors self http://localhost:8080 * http://localhost:8888/ https://bettinaheim.github.io/cuda-quantum https://bettinaheim.github.io/cuda-quantum/*"},"cookie_options":{"SameSite":"None","Secure":True}}' --IdentityProvider.token="my-custom-token" --ServerApp.allow_origin='*' --ServerApp.websocket_url=ws://localhost:8888


THIS WORKS:
jupyter-lab --no-browser --ip=* --port=8888 --ServerApp.tornado_settings='{"headers":{"Content-Security-Policy": "frame-ancestors self http://localhost:8080 * http://localhost:8888/ https://bettinaheim.github.io/cuda-quantum https://bettinaheim.github.io/cuda-quantum/*"},"cookie_options":{"SameSite":"None","Secure":True}}' --IdentityProvider.token="my-custom-token" --ServerApp.allow_origin='*' --IdentityProvider.cookie_options='{"SameSite":"None","Secure":True}'

# maybe we can --ip=localhost instead of --ip=* , 
# but allow_origin would have to be http://localhost:8888 (not sure why the proxy isn't changing that...)

# jupyter-lab --no-browser --ip=* --port=5802 --ServerApp.tornado_settings='{"headers":{"Content-Security-Policy": "frame-ancestors self http://localhost:5802 https://htmlpreview.github.io/?https%3A%2F%2Fgithub.com%2Fbettinaheim%2Fcuda-quantum%2Fblob%2Fdocs-preview%2Fv1"}}' --ServerApp.password="$(cat .jupyter/jupyter_server_config.json | egrep -o "hashed_password.*$" | cut -d ':' -f 2-)" --ServerApp.disable_check_xsrf=True

-> instead get password with:  python -c "from jupyter_server.auth import passwd; pw=passwd(algorithm='sha1'); print(pw)"

# could add "xsrf_cookie_kwargs":{"path":"_xsrf"} to the tornado_settings, but is not much help, similar "cookie_options":{"SameSite":"None"}

Notes: 
In the jupyterlab config, take a look at c.ServerApp.allow_remote_access, c.ServerApp.local_hostnames,

Workspace config:

{
  "data": {
    "layout-restorer:data": {
      "main": {
        "dock": {
          "type": "tab-area",
          "currentIndex": 0,
        }
      },
      "down": {
        "size": 0,
      },
      "left": {
        "collapsed": true,
        "visible": false,
      },
      "right": {
        "collapsed": true,
        "visible": false,
      },
      "relativeSizes": [
        0,
        1,
        0
      ],
      "top": {
        "simpleVisibility": false
      }
    },
    "file-browser-filebrowser:cwd": {
      "path": "tutorial/vqe.ipynb"
    }
  }
}

