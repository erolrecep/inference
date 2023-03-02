# Inference
---




## Setup

        $ conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
        $ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
        $ pip install tensorflow==2.11.0

*To automate pytorch and tensorflow sees cuda libraries installed by conda, follow these steps,*

        $ mkdir -p $CONDA_PREFIX/etc/conda/activate.d
        $ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

*To verify both library installations ..*

        $ import torch
        $ torch.cuda.is_available()

        $ python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"



## Set TensorFlow GPU Consumption


        $ import tensorflow as tf
        $ gpus = tf.config.experimental.list_physical_devices('GPU')
        $ for gpu in gpus:
        $   tf.config.experimental.set_memory_growth(gpu, True)

In my case, without this code block, TensorFlow was using almost all the available resources in the device memory.
After this code, it decreases the total memory consumption by 1 GB. The next step to limit the memory consumption
would be write a data loader to load the data optimized to the device.



