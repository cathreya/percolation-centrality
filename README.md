# percolation-centrality
Programs corresponding to the work "Efficient Parallel Algorithms for Computing Percolation Centrality"

 We  present  parallel  algorithms  that  compute  thesource-based  and  source-destination  variants  of  the  percolationcentrality  values  of  nodes  in  a  network.  Our  algorithms  extend the   algorithm   of   Brandes,   introduce   optimizations   aimed   atexploiting  the  structural  properties  of  graphs,  and  extend  thealgorithmic  techniques  introduced  by  Sariyuce  et  al.  [26]  in  thecontext  of  centrality  computation.  Experimental  studies  of  ouralgorithms on an Intel Xeon(R) Silver 4116 CPU and an NvidiaTesla V100 GPU on a collection of 12 real-world graphs indicatethat  our  algorithmic  techniques  offer  a  significant  speedup.

This repositary contains the implementations for both the versions of percolation on CPU and GPU respectively.

## Dependencies

- CUDA  Version  11.3 forprogramming the V100 GPU.
- g++ version 7.3.0 compiler
- C++ along with OpenMP version 4.5

## Dataset

The graph instances used in the experiment can be found here : https://drive.google.com/drive/folders/1YAmc3DTl94kMpcWGQ1wUIS5AR43eYeao?usp=sharing

## Run

The compile instructions for the respective codes can be found as a comment inside the codes. 

To run the CPU codes use :
```
./cpu_src_only <input_file> <output_file> <num_threads>
./cpu_src_dest <input_file> <output_file> <num_threads>
```

To run the GPU codes use :
```
./gpu_src_only <input_file> <output_file>
./gpu_src_dest <input_file> <output_file>
```
