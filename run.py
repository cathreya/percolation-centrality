from subprocess import run
import argparse
from sys import exit
from os.path import isfile, exists
from os import mkdir

parser = argparse.ArgumentParser(description='Find source-based/source-destination-based percolation centrality')
parser.add_argument('-a', '--algorithm', dest='algo',
                    default=['pcso'], nargs=1,
                    help='pcso - source-based percolation centrality, \
                    	  pcsd - source-destination-based percolation centrality \
                    	  pcsobcc - source-based percolation centrality with bcc decomposition \
                    	  pcsdbcc - source-destination-based percolation centrality with bcc decomposition')

parser.add_argument('-d', '--dataset', dest='dataset',
                    default=['none'], nargs=1,
                    help='The dataset to run on. Dataset must be present in ./datasets subdirectory.\
                    	  Look at existing datasets for the input format')

parser.add_argument('-o', '--outfile', dest='outfile',
                    default=['out.txt'], nargs=1,
                    help='Output file name')

parser.add_argument('-t', '--cputhreads', dest='threads',
                    default=[32], nargs=1,
                    help='Number of OpenMP threads (max 48)')


parser.add_argument('-g', '--gpu', dest='gpuflag', action='store_true',
                    help='Run experiment of GPU (default: multicore CPU)')

parser.add_argument('-r', '--recompile', dest='compileflag', action='store_true',
                    help='Recompile executables')



args = parser.parse_args()
algo = args.algo[0]
dataset = args.dataset[0]
outfile = args.outfile[0]
threads = args.threads[0]	
gpuflag = args.gpuflag
compileflag = args.compileflag


if algo not in ['pcso', 'pcsd', 'pcso', 'pcsdbcc']:
	exit('Error: Invalid algorithm.')


if not isfile("./datasets/"+dataset):
	exit('Error: Dataset not found.')

command_map = {
	('pcso',False) : 'g++ -O2 -fopenmp -static-libstdc++ ./src/CPU/src_only_PC.cpp -o ./exec/CPU/pcso',
	('pcsd',False) : 'g++ -O2 -fopenmp -static-libstdc++ ./src/CPU/src_dest_PC.cpp -o ./exec/CPU/pcsd',
	('pcsobcc',False) : 'g++ -O2 -fopenmp -static-libstdc++ ./src/CPU/src_only_PC.cpp -o ./exec/CPU/pcsobcc',
	('pcsdbcc',False) : 'g++ -O2 -fopenmp -static-libstdc++ ./src/CPU/src_dest_PC.cpp -o ./exec/CPU/pcsdbcc',
	('pcso',True) : 'nvcc ./src/GPU/.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3 -o ./exec/GPU/pcso',
	('pcsd',True) : 'nvcc ./src/GPU/.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3 -o ./exec/GPU/pcsd',
	('pcsobcc',True) : 'nvcc ./src/GPU/.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3 -o ./exec/GPU/pcsobcc',
	('pcsdbcc',True) : 'nvcc ./src/GPU/.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3 -o ./exec/GPU/pcsdbcc'
}

exec_map = {
	('pcso',False) : './exec/CPU/pcso',
	('pcsd',False) : './exec/CPU/pcsd',
	('pcsobcc',False) : './exec/CPU/pcsobcc',
	('pcsdbcc',False) : './exec/CPU/pcsdbcc',
	('pcso',True) : './exec/GPU/pcso',
	('pcsd',True) : './exec/GPU/pcsd',
	('pcsobcc',True) : './exec/GPU/pcsobcc',
	('pcsdbcc',True) : './exec/GPU/pcsdbcc'
}

if not exists('./exec'):
	mkdir('./exec')

if not exists('./exec/CPU'):
	mkdir('./exec/CPU')

if not exists('./exec/GPU'):
	mkdir('./exec/GPU')

if not exists('./output'):
	mkdir('./output')


if not isfile(exec_map[(algo,gpuflag)]) or compileflag:
	print("Didn't find executable or recompile flag set.")
	print("Compiling ...")
	run(command_map[(algo,gpuflag)], shell=True, check=True)
	print("Finished Compiling")
else:
	print("Found existing executables. Skipping compilation.")

thread_arg = ["", str(threads)][int(gpuflag)]
print("Running Algorithm {} on dataset {} on {}".format(algo, dataset, ['CPU','GPU'][int(gpuflag)]))
run(exec_map[(algo,gpuflag)]+ " ./datasets/{} ./output/{} {}".format(dataset, outfile, threads), shell=True, check=True)
print("Successfully completed execution. Output can be found at ./output/{}".format(outfile))








