# Inference
---




## Setup for Linux

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


## Setup for Apple Silicon

        # Create a conda environment
        $ conda create -n inference python 3.10
        $ conda activate inference

        # For TensorFlow
        $ conda install -c apple tensorflow-deps
        $ pip install tensorflow-macos
        $ pip install tensorflow-metal

        # For PyTorch
        $ conda install pytorch torchvision torchaudio -c pytorch-nightly

*TensorFlow reference --> [link](https://developer.apple.com/metal/tensorflow-plugin/)*
</br>
*PyTorch reference --> [link](https://developer.apple.com/metal/pytorch/)*


## Set TensorFlow GPU Consumption


        $ import tensorflow as tf
        $ gpus = tf.config.experimental.list_physical_devices('GPU')
        $ for gpu in gpus:
        $   tf.config.experimental.set_memory_growth(gpu, True)

In my case, without this code block, TensorFlow was using almost all the available resources in the device memory.
After this code, it decreases the total memory consumption by 1 GB. The next step to limit the memory consumption
would be write a data loader to load the data optimized to the device.

---

### TO-DO List

 1. Learn more about grad-cam visualization and it's implementations.
    + https://github.com/jacobgil/pytorch-grad-cam 
    + https://pyimagesearch.com/2020/03/09/grad-cam-visualize-class-activation-maps-with-keras-tensorflow-and-deep-learning/ 
 2. Learn more about feature maps and filters visualizations --> 
    + https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/ 
 3. Learn more about model pruning
    + https://pytorch.org/tutorials/intermediate/pruning_tutorial.html 
 4. Learn more about model post-training quantization
    + https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html 
    + https://pytorch.org/TensorRT/tutorials/ptq.html 
    + https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/index.html
 5. Learn more about NVIDIA TensorRT tool for model quantization
    + https://docs.nvidia.com/deeplearning/tensorrt/sample-support-guide/index.html#onnx_mnist_sample 
    + https://developer.nvidia.com/tensorrt 
    + https://github.com/NVIDIA/TensorRT

