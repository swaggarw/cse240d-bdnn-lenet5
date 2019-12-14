# CSE240D, Fall 2019


## In this project we have implemented LeNet-5 as a Binary Deep Neural Network, synthesized and verified the design using Vivado HLS

### LeNet-5
![LeNet-5](https://cdn-images-1.medium.com/max/800/1*lvvWF48t7cyRWqct13eU0w.jpeg)

Final Results (for inference on 100 images):

| ------------- | Utilization      | Timing (ns)     | Latency (ms) |
| ------------- | ---------------- | --------------- | ------------ |
| 1-PE          |  11%             | 3.717           | 41.9         | 
| 2-PE          |  78%             | 3.717           | 21.86        |


### Training Weights
The training weights and parameter names are included in mnist-cnn-params.h and mnist-cnn-config.h. The Vivado HLS libraries automatically include these files if the path to this repo is added in the include path.

### Testbench Verification

In order to perform Testbench Verification, run the command: $ t_1W1A mnist.t/mnist.t <#Number of images to infer>
```bash
t_1W1A mnist.t/mnist.t 100 # Performs inference on 100 images and prints accuracy. mnist.t is the dataset file.
```

### Synthesis
In order to perform Synthesis:
  - Clone the repository
  - Add repo root folder as an include library in Vivado HLS
  - Make sure the include statements match with the include path in your Vivado HLS environment (line: 6-8 in mnist-cnn-1W1A.cpp)
  - Select DoCompute as top module (for 1 PE inference) and set the numReps variable to the number of images you want to infer in one run (line 11).
  - Select ArbitrateCompute (for 2 PE inference) and set the numReps variable to the number of images you want to infer in one run (line 110)
  - Set target clock in Solution Settings and run C Synthesis

### Directory structure

- hls-nn-lib/ - Contains the header libraries for BNN implementation
- mnist.t/ - Contains dataset file for MNIST
- ArbitrateCompute_100imgs_2PE_3ns.html: Synthesis results for 2 PE system for 100 images
- Makefile: Makefile to compile your testbench for accuracy calculation. Export your Vivado library path as $XILINX_VIVADO variable before running this.
- loader.h: loader helper libraries to load the MNIST dataset
- mnist-cnn-1W1A.cpp: Main design file for the B-DNN with 1-bit weight and 1-bit activation
- mnist-cnn-config.h: LeNet-5 Configuration file for various layer parameters
- mnist-cnn-params.h: LeNet-5 parameter file containing weight and activation values  
- t_1W1A: Compiled testbench file for accuracy calculation

## A project by - 

- Swapnil Aggarwal
- Utkarsh Singh
