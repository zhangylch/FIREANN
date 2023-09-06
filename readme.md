Field-Induced Recursively embedded atom neural network 
=================================================
**Introduction:**
___________________________
Field-induced Recursively embedded atom neural network (FIREANN) is a PyTorch-based end-to-end multi-functional Deep Neural Network Package for Molecular, Reactive and Periodic Systems under the presence of the external field with rigorous rotational equivariance. As a results, FIREANN framework intrinsically describes the response of the potential energy to an external field up to an arbitrary order (dipole moments, polarizabilities …) by taking the analytical gradients of the potential energy with respect to the field vector. FIREANN is developed based on the [REANN package] (https://github.com/zhangylch/REANN.git) and inherits all features of the REANN package including Distributed DataParallel parallelized training on both GPU and CPU and a efficient interface with LAMMPS for GPU and CPU.

**Requirements:**
___________________________________
* PyTorch 2.0.0
* LibTorch 2.0.0
* cmake 3.1.0
* opt_einsum 3.2.0

** Training Workflow**
______________________________________________
The training process can be divided into four parts: information loading, initialization, dataloader and optimization. First, the "src.read" will load the information about the systems and NN structures from the dataset and input files (“input_nn” and “input_density”) respectivrly. Second, the "run.train" module utilizes the loaded information to initialize various classes, including property calculator, dataloader, and optimizer. For each process, an additional thread will be activated in the "src.dataloader" module to prefetch data from CPU to GPU in an asynchronous manner. Meanwhile, the optimization will be activated in the "src.optimize" module once the first set of data is transferred to the GPU. During optimization, a learning rate scheduler, namely "ReduceLROnPlateau" provided by PyTorch, is used to decay the learning rate. Training is stopped when the learning rate drops below "end_lr" and the model that performs best on the validation set is saved for further investigation.

**References:**
__________________________________________________
If you use this package, please cite these works.
1. FIREANN model: Yaolong Zhang and Bin Jiang arXiv:2304.07712.
2. The original EANN model: Yaolong Zhang, Ce Hu and Bin Jiang *J. Phys. Chem. Lett.* 10, 4962-4967 (2019).
3. The theory of REANN model: Yaolong Zhang, Junfan Xia and Bin Jiang *Phys. Rev. Lett.* 127, 156002 (2021).
4. The details about the implementation of REANN: Yaolong Zhang, Junfan Xia and Bin Jiang *J. Chem. Phys.* 156, 114801 (2022).
