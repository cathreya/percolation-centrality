#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>
#include <set>
#include <time.h>
#include <iostream>
#include <algorithm>
#include <assert.h>
#include <iostream>
#include <vector>
#include <stack>
#include <set>
#include <queue>
#include <list>
#include <iomanip>
#include <chrono>
#include <fstream>
using namespace std;

// compile : nvcc <file_name>.cu -arch=sm_70 -std=c++11 -Xcompiler -fopenmp -O3

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
float *Paths, float *dDelta, node *Parents, float *dCentrality, int *crr, float *pc, float *orig_delta)
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
			dParent[0] = {root, 0};
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
					dParent[atomicAdd(parIndex, 1)] = {neighbour,source};
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

		int id = *parIndex - threadIdx.x, *reach = arr;

		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) done[i] = 0;	
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) delta[i] = orig_delta[i];

		while(*reach > 0)
		{
			if(id > 0 && dDistance[dParent[id].child] == *reach)
			{
				node n = dParent[id];

				atomicAdd(&delta[n.parent], (dPaths[n.parent]/dPaths[n.child])*(1.0+(delta[n.child])));

				if(atomicExch(&done[n.child], 1) == 0)
				{
					atomicAdd(&dCentrality[n.child], pc[rootIndex]*delta[n.child]);
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

int n,m;
int timer;
vector<double> x, reach, reachcnt;
vector<double> pre_orig_delta;
vector<pair<int,int> > st;
vector<int> vis,vis1,low,entry;
vector<vector<int> > g, tmp_g;
double sum_x;

int vertices;
vector<int> rep;
vector<double> global_pc,tmp_pc;

void dfs(int u, int par, int &compsz, double &compwt){
	timer++;
	entry[u] = timer;
	vis[u] = 2;
	reach[u] = x[u];
	reachcnt[u] = 1;
	low[u] = entry[u];
	double wt_children = x[u];
	int cnt_children = 1;
	for(int v:g[u]){
		if(vis[v]< 2){
			st.push_back({u,v});
			dfs(v, u, compsz, compwt);
			low[u] = min(low[u], low[v]);
			if(low[v] >= entry[u]){
				vector<int> unique_vertices;
				++vertices;
				rep.push_back(u);
				x[vertices] = x[u];
				while(st.back() != make_pair(u,v)){
					int p = st.back().first;
					int q = st.back().second;
					st.pop_back();
					if(p == u)
						p = vertices;
					if(q == u)
						q = vertices;
					unique_vertices.push_back(p);
					unique_vertices.push_back(q);
					tmp_g[p].push_back(q);
					tmp_g[q].push_back(p);
				}
				tmp_g[vertices].push_back(v);
				tmp_g[v].push_back(vertices);
				unique_vertices.push_back(v);
				unique_vertices.push_back(vertices);
				sort(unique_vertices.begin(), unique_vertices.end());
				unique_vertices.erase(unique(unique_vertices.begin(), unique_vertices.end()),unique_vertices.end());
				
				st.pop_back();
				double su = 0;
				int cnt = 0;
				for(int p:unique_vertices){
					if(p==vertices) continue;
					su += reach[p];
					cnt += reachcnt[p];
					pre_orig_delta[p] = (reachcnt[p] - 1); 
				}

				cnt_children += cnt;
				wt_children += su;
				reach[vertices] = compwt - su;
				reachcnt[vertices] = compsz - cnt;
				pre_orig_delta[vertices] = (reachcnt[vertices]-1);
				
				reach[u] = x[u];
				reachcnt[u] = 1;
			}
		}
		else if(v != par && entry[v] < entry[u]){
			st.push_back({u,v});
			low[u] = min(low[u], entry[v]);
		}
	}
	reachcnt[u] = cnt_children;
	reach[u] = wt_children;
	vis[u] = 3;
}

void prelim_dfs(int u, int &cnt, double &wt){
	vis[u] = 1;
	cnt += 1;
	wt += x[u];
	for(int v:g[u]){
		if(!vis[v]){
			prelim_dfs(v,cnt,wt);
		}
	}
}

int main( int argc, char **argv )
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

   string input = argv[1];
	string output = argv[2];

   ifstream fin (input);
   ofstream fout (output);
   fin >> n >> m;
	vertices = n;
	for(int i=0;i<=n;++i)
		rep.push_back(i);
	g.resize(n+1);
	tmp_g.resize(10*n+1);
	global_pc.resize(n+1);
	x.resize(10*n+1);
	reach.resize(10*n+1);
	reachcnt.resize(10*n+1);
	vis.resize(n+1);
	low.resize(n+1);
	entry.resize(n+1);
	pre_orig_delta.resize(10*n+1,0);
	
	timer = 0;
	sum_x = 0;
   int cnt = 0;
   double wt = 0;

	for(int i=1;i<=n;i++){
		x[i] = 1.0/(double)(i);
		sum_x += x[i];
	}
	for(int i=0;i<m;i++){
		int u,v;
		fin>>u>>v;
		if(u == v)
			continue;
		g[u].push_back(v);
		g[v].push_back(u);
	}

   auto t1 = std::chrono::high_resolution_clock::now();

	for(int i=1;i<=n;i++){
		if(!vis[i]){
			cnt = 0;
			wt = 0;
			prelim_dfs(i,cnt,wt);
			dfs(i,0, cnt, wt);
		}
	}

	tmp_g.resize(vertices+1);
	tmp_pc.resize(vertices+1);
	x.resize(vertices+1);
	reach.resize(vertices+1);
	reachcnt.resize(vertices+1);
	pre_orig_delta.resize(vertices+1);

	int V, E;
	V = vertices;
	for(int i=1;i<=vertices;++i)
		E += (int)(tmp_g[i].size());
	E = E/2;

	float *perc  = new float[V+2];
	float *pre_delta = new float[V+2];

	for(int i=1;i<=V;++i)
	{
		perc[i] = reach[i];
		pre_delta[i] = pre_orig_delta[i];
	}
	
	perc[0] = perc[V+1] = 1.0;
	pre_delta[0] = pre_delta[V+1] = 0.0;

	int *hColumn = new int[2*E];
	int *hRow	 = new int[V+2];

	for(int index=0, i=1; i<=V; ++i) 
	{
		for(int j=0;j<(int)tmp_g[i].size();++j)
		{
			int n = tmp_g[i][j]; 
			hColumn[index++] = n;
		}
	}

	long count = 0;
	for(int i=0; i<=V;)
	{
		for(int j=0;j<(int)tmp_g.size();++j)
		{
			vector<int> v = tmp_g[i];
			hRow[i++] = count;
			count += v.size();
		}
	}
	hRow[V+1] = count;
	
	float *delta, *Paths, *dCentrality, *pc, *orig_delta;
	node *Parents;
	int *dColumn, *dRow, *Distance, *Queue, *crr;

	cudaMalloc((void**)&dRow,    		sizeof(int)*(V+2));
	cudaMalloc((void**)&dCentrality,	sizeof(float)*(V+2));
	cudaMalloc((void**)&pc,				sizeof(float)*(V+2));
	cudaMalloc((void**)&orig_delta,	sizeof(float)*(V+2));
	cudaMalloc((void**)&dColumn, 		sizeof(int)*(2*E));
	cudaMalloc((void**)&crr,    		sizeof(int)*(V+2)*NUM_BLOCKS);
	cudaMalloc((void**)&Queue,    	sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Distance,		sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Paths,			sizeof(float)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&delta,			sizeof(float)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Parents,		sizeof(node)*(E+1)*NUM_BLOCKS);

	cudaMemcpy(dRow, hRow, sizeof(int)*(V+2),cudaMemcpyHostToDevice);
	cudaMemcpy(dColumn, hColumn, sizeof(int)*(2*E), cudaMemcpyHostToDevice);
	cudaMemcpy(pc, perc, sizeof(float)*(V+2),cudaMemcpyHostToDevice);
	cudaMemcpy(orig_delta, pre_delta, sizeof(float)*(V+2),cudaMemcpyHostToDevice);
	gpuErrchk( cudaPeekAtLastError() );

	auto t3 = std::chrono::high_resolution_clock::now();

	brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
				Parents, dCentrality, crr, pc, orig_delta);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
	cudaDeviceSynchronize();

	float *Centrality = new float[V+1];
	cudaMemcpy(Centrality, dCentrality, sizeof(float)*(V+1), cudaMemcpyDeviceToHost);
	for(int i=1;i<=V;++i)
		global_pc[rep[i]] += Centrality[i];

	for(int i=1;i<=n;i++){
		global_pc[i] /= (n-2);
		global_pc[i] /= (sum_x-x[i]);
	}

	delete[] hRow;
	delete[] hColumn;

	cudaFree(Queue);
	cudaFree(dRow);
	cudaFree(dColumn);
	cudaFree(Distance);

	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t3 - t1 ).count();
	cerr << "preprocessing time : " << duration << " mu.s." <<endl;
	duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	cerr << "total time : " << duration << " mu.s." <<endl;

	for(int i=1;i<=n;i++)
		fout << global_pc[i] << "\n";
	
	return 0;
}