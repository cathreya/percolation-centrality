# percolation-centrality
Programs corresponding to the work "Efficient Parallel Algorithms for Computing Percolation Centrality"

We  present  parallel  algorithms  that  compute  the source-based  and  source-destination  variants  of  the  percolation centrality  values  of  nodes  in  a  network.  Our  algorithms  extend the   algorithm   of   Brandes,   introduce   optimizations   aimed   at exploiting  the  structural  properties  of  graphs,  and  extend  the algorithmic  techniques  introduced  by  Sariyuce  et  al. in  the context  of  centrality  computation.  

This repo contains the implementations for both the versions of percolation centrality on CPU and GPU respectively.

## Dependencies

- python 3.7+
- CUDA  Version  11.3
- g++ version 7.3.0 compiler
- OpenMP version 4.5

## Run

The helper script `run.py` can be used to execute the programs on a Linux Environment. 
```
usage: run.py [-h] [-a ALGO] [-d DATASET] [-o OUTFILE] [-t THREADS] [-g] [-r]

Find source-based/source-destination-based percolation centrality

optional arguments:
  -h, --help            show this help message and exit
  -a ALGO, --algorithm ALGO
                        pcso - source-based percolation centrality, pcsd -
                        source-destination-based percolation centrality
                        pcsobcc - source-based percolation centrality with bcc
                        decomposition pcsdbcc - source-destination-based
                        percolation centrality with bcc decomposition
  -d DATASET, --dataset DATASET
                        The dataset to run on. Dataset must be present in
                        ./datasets subdirectory. Look at existing datasets for
                        the input format
  -o OUTFILE, --outfile OUTFILE
                        Output file name
  -t THREADS, --cputhreads THREADS
                        Number of OpenMP threads (max 48)
  -g, --gpu             Run experiment of GPU (default: multicore CPU)
  -r, --recompile       Recompile executables
```

For example, to run the source only version on the dataset `PGPgiantcompo.in` on CPU,
```
python run.py --algorithm pcso --dataset PGPgiantcompo.in
```

To run the source destination version with BCC decomposition on the dataset `PGPgiantcompo.in` on GPU and store the output in `my_outfile.txt`,
```
python run.py --algorithm pcsdbcc --dataset PGPgiantcompo.in --gpu -o my_outfile.txt 
```