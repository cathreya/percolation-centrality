#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <set>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <assert.h>

// compile : nvcc <file_name>.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3

using namespace std;

#define NUM_THREADS 32
#define NUM_BLOCKS 1024


typedef struct
{
	int child;
	int parent;
} node;


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void brandes(int V, int E, int *dColumn, int *dRow, int *Distance, int *Queue,
float *Paths, float *dDelta, node *Parents, float *dCentrality, int *crr, float *pc)
{
	__shared__ int arr[2];
	int *QLen = arr, *parIndex = arr+1;
	
	int rootIndex = blockIdx.x + 1;
	int *Q = Queue + (blockIdx.x)*(V+1);
	float *dPaths = Paths + (blockIdx.x)*(V+1);
	float *delta = dDelta + (blockIdx.x)*(V+1);
	int *done = crr + (blockIdx.x)*(V+1);
	int *dDistance = Distance + (blockIdx.x)*(V+1);
	node *dParent = Parents + (blockIdx.x)*(E+1);

	while(rootIndex <= V)
	{
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) dPaths[i] = 0;
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) dDistance[i] = -1;

		if(threadIdx.x==0)
		{
			*QLen = *parIndex = 1;
			int root = rootIndex;
			Q[0] = root;
			dPaths[root] = 1.0f;
			dDistance[root] = 0;
			dParent[0].child = root;
			dParent[0].parent = 0;
		}
		__syncthreads();

		int oldQLen = 0;
		while(oldQLen < *QLen)
		{
			int id = threadIdx.x;
			int	source = Q[oldQLen++];
			int degree = dRow[source+1] - dRow[source];

			while(id < degree)
			{
				int neighbour = dColumn[dRow[source]+id];
				if(dDistance[neighbour] == -1)
				{
					dDistance[neighbour] = dDistance[source]+1;
					Q[atomicAdd(QLen, 1)] = neighbour;
				}
				if(dDistance[neighbour] == dDistance[source]+1)
				{
					dPaths[neighbour] += dPaths[source];
					int tmp = atomicAdd(parIndex, 1);
					dParent[tmp].child = neighbour;
					dParent[tmp].parent = source;
				}
				id += NUM_THREADS;
			}
			__syncthreads();
		}

		if(threadIdx.x==0)
		{
			arr[0] = dDistance[Q[*QLen-1]];
			*parIndex -= 1;
		}
		__syncthreads();


		float *deltaG = (float*)Q, factor = 1.0f;
		int id = *parIndex - threadIdx.x, *reach = arr;

		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) done[i] = 0;	
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) delta[i] = 0.0f;
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) deltaG[i] = 0.0f;

		if(crr[0]=='-') factor = -1.0f;
		while(*reach > 0)
		{
			if(id > 0 && dDistance[dParent[id].child] == *reach)
			{
				node n = dParent[id];

				atomicAdd(&delta[n.parent], (dPaths[n.parent]/dPaths[n.child])*(max(0.0f,pc[rootIndex]-pc[n.child])+(delta[n.child])));

				if(atomicExch(&done[n.child], 1) == 0)
				{
					atomicAdd(&dCentrality[n.child], factor*delta[n.child]);
				}

				bool flag = dDistance[dParent[id-1].child] == *reach-1;
				if(threadIdx.x==NUM_THREADS-1 || flag)
				{
					*parIndex = id-1;
					*reach -= (flag);
				}
			}
			__syncthreads();
			id = *parIndex - threadIdx.x;
		}
		rootIndex += NUM_BLOCKS;
	}
}

int main( int argc, char **argv )
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

   string input = argv[1];
	string output = argv[2];

   ifstream fin(input);
   ofstream fout(output);

   int V,E;

   fin >> V >> E;

	vector <vector <int> > graph(V+1);
	for(int i=0; i<E; ++i)
	{
		int u, v;
		fin >> u >> v;
		if(u == v) continue;
		graph[u].push_back(v);
		graph[v].push_back(u);
	}
	auto t1 = std::chrono::high_resolution_clock::now();

	int *hColumn = new int[2*E];
	int *hRow	 = new int[V+2];
	float *perc  = new float[V+2];

	for(int i=1;i<=V;++i)
		perc[i] = 1.0/(float)(i);
	perc[0] = perc[V+1] = 1.0;

	for(int index=0, i=1; i<=V; ++i) 
	{
		for(int j=0;j<(int)graph[i].size();++j)
		{
			int n = graph[i][j]; 
			hColumn[index++] = n;
		}
	}
	
	// Filling row array
	long count = 0;
	for(int i=0; i<=V;)
	{
		for(int j=0;j<(int)graph.size();++j)
		{
			vector<int> v = graph[i];
			hRow[i++] = count;
			count += v.size();
		}
	}
	hRow[V+1] = count;

	float *delta, *Paths, *dCentrality, *pc;
	node *Parents;
	int *dColumn, *dRow, *Distance, *Queue, *crr;

	cudaMalloc((void**)&dRow,    		sizeof(int)*(V+2));
	cudaMalloc((void**)&dCentrality,	sizeof(int)*(V+2));
	cudaMalloc((void**)&pc,				sizeof(int)*(V+2));
	cudaMalloc((void**)&dColumn, 		sizeof(int)*(2*E));
	cudaMalloc((void**)&crr,    		sizeof(int)*(V+2)*NUM_BLOCKS);
	cudaMalloc((void**)&Queue,    		sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Distance,		sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Paths,			sizeof(float)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&delta,			sizeof(float)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Parents,		sizeof(node)*(E+1)*NUM_BLOCKS);

	cudaMemcpy(dRow, hRow, sizeof(int)*(V+2),cudaMemcpyHostToDevice);
	cudaMemcpy(dColumn, hColumn, sizeof(int)*(2*E), cudaMemcpyHostToDevice);
	cudaMemcpy(pc, perc, sizeof(int)*(V+2),cudaMemcpyHostToDevice);
	gpuErrchk( cudaPeekAtLastError() );

	clock_t start = clock();

	vector<pair<float,int> > perc_pair(V+1);
	vector<float> contrib(V+1);
	perc_pair[0].first = 0;
	perc_pair[0].second = 0;
    for(int i=1;i<=V;++i)
    {
		perc_pair[i].first = perc[i];
		perc_pair[i].second = i;
    }
	sort(perc_pair.begin(),perc_pair.end());
	float carry = 0,sum_x = 0;
	for(int i=1;i<=V;++i)
	{
		contrib[perc_pair[i].second] = (float)(i-1)*perc_pair[i].first-carry;
		carry += perc_pair[i].first;
		sum_x += contrib[perc_pair[i].second];
	}
	carry = 0;
	for(int i=V;i>=1;i--)
	{
		contrib[perc_pair[i].second] += carry-(float)(V-i)*perc_pair[i].first;
		carry += perc_pair[i].first;
	}

	brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
				Parents, dCentrality, crr, pc);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
	clock_t stop = clock();
	// printf("Time = %f\n", (double)(stop-start)/CLOCKS_PER_SEC);

	cudaDeviceSynchronize();

	float *Centrality = new float[V+1];
	cudaMemcpy(Centrality, dCentrality, sizeof(float)*(V+1), cudaMemcpyDeviceToHost);
	for(int i=1; i<=V; ++i)
	{
		Centrality[i] /= (sum_x-contrib[i]);
	}

	delete[] hRow;
	delete[] hColumn;

	cudaFree(Queue);
	cudaFree(dRow);
	cudaFree(dColumn);
	cudaFree(Distance);

	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	cerr << "total time : " << duration << " mu.s." <<endl;

	for(int i=1; i<=V; ++i)
		fout << Centrality[i] << "\n";

	return 0;
}


void printfGraph(vector <vector<int> > &graph)
{
	printf("Graph is\n");
	for(int i=1; i<graph.size(); ++i)
	{
		printf("\n%d\t", i);
		for(int j=0;j<(int)graph[i].size();++j)
		{
			int n = graph[i][j];
			printf("%d ", n);
		}
	}
}

__global__ void dPrintGraph(int V, int E, int *dRow, int *dColumn)
{
	printf("printing from device:\nRow is\n");
	for(int i=0; i<=V+1; ++i) printf("%d ", dRow[i]);
	printf("\nCol is\n");
	for(int i=0; i<2*E; ++i) printf("%d ", dColumn[i]);
	printf("\n");
}

__global__ void dPrintDist(int V, int E, int *dDistance)
{
	printf("Distances \n");
	for(int i=1; i<(V+1)*V; ++i)
	{
		if(i%(V+1)==0) { ++i; printf("\n"); }
		printf("%d ", dDistance[i]);
	}
}
