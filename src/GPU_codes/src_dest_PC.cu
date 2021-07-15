#include<cuda.h>
#include<cuda_runtime.h>
#include<stdio.h>
#include<vector>
#include<set>
#include<time.h>
#include<iostream>
#include<vector>
#include<stack>
#include<set>
#include<queue>
#include<list>
#include<iomanip>
#include<algorithm>
#include<chrono>
#include<fstream>
#include<omp.h>
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

__device__ void merge(float A[], float temp[], int from, int mid, int to, int N)
{
    int k = from, i = from, j = mid + 1;
    while (i <= mid && j <= to)
    {
        if (A[i] < A[j]) {
            temp[k++] = A[i++];
        }
        else {
            temp[k++] = A[j++];
        }
    }

    while (i < N && i <= mid) {
        temp[k++] = A[i++];
    }

    for (int i = from; i <= to; i++) {
        A[i] = temp[i];
    }
}

__global__ void brandes(int V, int E, int *dColumn, int *dRow, int *Distance, int *Queue,
float *Paths, float *dDelta, node *Parents, float *dCentrality, int *crr, float *perc_state, 
float *reach_vec, float *reach_suffixsum, int *starters)
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
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) done[i] = 0;	
		for(int i=threadIdx.x; i<=V; i+=NUM_THREADS) delta[i] = 0;

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
		int l = starters[rootIndex];
		int r = starters[rootIndex+1]-1;
		for (int m = 1; m <= r - l; m = 2*m)
	    {
	    	int i = l+2*m*threadIdx.x;
	        while(i<r)
	        {
	            int from = i;
	            int mid = i + m - 1;
	            int to = min(i + 2*m - 1, r);
	 
	            merge(reach_vec, reach_suffixsum, from, mid, to, r+1);
	            i += 2*m*NUM_THREADS;
	        }
	        __syncthreads();
	    }

		if(threadIdx.x==0)
		{
			reach_suffixsum[r] = reach_vec[r];
			for(int i=r-1; i>=l;i--)
				reach_suffixsum[i] = reach_vec[i] + reach_suffixsum[i+1];
		}
		__syncthreads();

		int oldQLen = 0;
		while(oldQLen < *QLen)
		{
			int	source = Q[oldQLen++];
			int degree = dRow[source+1] - dRow[source];
			if(source != rootIndex)
			{
				int id = starters[source]+threadIdx.x;
				while(id < starters[source+1])
				{
					float xi = reach_vec[id];
					if(xi == perc_state[source])
					{
						id += NUM_THREADS;
						continue;
					}
					int l = starters[rootIndex];
					int r = starters[rootIndex+1]-1;
					if(reach_vec[r] <= xi) 
					{
						id += NUM_THREADS;
						continue;
					}
					r++;
					while(l<r)
					{
						int m = (l+r)/2;
						if(reach_vec[m] <= xi) l = m+1;
						else r = m;
					}
					int cnt = starters[rootIndex+1]-l;
					atomicAdd(&delta[source], reach_suffixsum[l] - xi*cnt);
					id += NUM_THREADS;
				}
				__syncthreads();
			}

			int id = threadIdx.x;
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

		while(*reach > 0)
		{
			if(id > 0 && dDistance[dParent[id].child] == *reach)
			{
				node n = dParent[id];

				float add = 0;
				float xi = perc_state[n.child];
				int l = starters[rootIndex];
				int r = starters[rootIndex+1]-1;
				if(reach_vec[r] > xi)
				{
					r++;
					while(l<r)
					{
						int m = (l+r)/2;
						if(reach_vec[m] <= xi) l = m+1;
						else r = m;
					}
					int cnt = starters[rootIndex+1]-l;
					add = reach_suffixsum[l] - xi*cnt; 
				}
			
				atomicAdd(&delta[n.parent], (dPaths[n.parent]/dPaths[n.child])*(add+(delta[n.child])));
				
				if(atomicExch(&done[n.child], 1) == 0)
				{
					atomicAdd(&dCentrality[n.child], delta[n.child]);
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
int vertices;
int timer;
vector<double> pc,x,contrib,global_pc;
vector<vector<double> > reach, reachsuf;
vector<vector<int> > reachv;
vector<pair<int,int> > st;
vector<int> vis,vis1,low,entry, cur_comp;
vector<vector<int> > g, tmp_g;
vector<bool> in_otherbccs;
vector<pair<long double,int> > perc;
vector<int> rep;
vector<int> to_reach;
vector<vector<int> > unique_vertices;

double bccramp;
double watch;
double sum_x;

vector<vector<int> > p;
vector<int> sigma, d;
vector<double> delta;
vector<int> qs;

double R(double u){
	if(u>0) return u;
	return 0;
}

void compute_source_perc(int src, std::vector<std::vector<int> > &g){
	qs.push_back(src);
	int it = 0;
	d[src] = 0;
	sigma[src] = 1;
	while(it != (int)qs.size()){
		int u = qs[it];
		it += 1;
		for(int v:g[u]){
			if(d[v] < 0){
				qs.push_back(v);
				d[v] = d[u] + 1;
			}
			if(d[v] == d[u] + 1){

				sigma[v] += sigma[u];
				p[v].push_back(u);
			}
		}
	}
	while(!qs.empty()){
		int v = qs.back();
		qs.pop_back();
		bool fl = 1;
		for(double xi:reach[v])
		{
			if(fl == 1 && xi == x[v])
			{
				fl = 0;
				continue;
			}
			int ind = lower_bound(reach[src].begin(), reach[src].end(), xi) - reach[src].begin();
			if(ind < (int)(reach[src].size())){
				int cnt = (int)reach[src].size() - ind;
				delta[v] += reachsuf[src][ind] - xi*cnt; 
			}
		}
		for(int u:p[v]){
			int ind = lower_bound(reach[src].begin(), reach[src].end(), x[v]) - reach[src].begin();
			double add = 0;
			if(ind < (int)reach[src].size()){
				int cnt = (int)reach[src].size() - ind;
				add = (reachsuf[src][ind] - x[v]*cnt);
			}
			delta[u] += (double)(sigma[u])/sigma[v] * (delta[v]+add);
		}
		if(v != src){
			pc[v] += delta[v];
		}
		d[v] = -1;
		sigma[v] = 0;
		p[v].clear();
		delta[v] = 0;
	}
}


void dfs(int u, int par){
	timer++;
	entry[u] = timer;
	vis[u] = 2;
	low[u] = entry[u];
	vector<int> children = {u};

	for(int v:g[u])
	{
		if(vis[v]< 2)
		{
			st.push_back({u,v});
			dfs(v, u);
			low[u] = min(low[u], low[v]);

			if(low[v] >= entry[u])
			{
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
					unique_vertices[vertices].push_back(p);
					unique_vertices[vertices].push_back(q);
					tmp_g[p].push_back(q);
					tmp_g[q].push_back(p);
				}
				st.pop_back();
				tmp_g[vertices].push_back(v);
				tmp_g[v].push_back(vertices);
				unique_vertices[vertices].push_back(v);
				unique_vertices[vertices].push_back(vertices);
				sort(unique_vertices[vertices].begin(), unique_vertices[vertices].end());
				unique_vertices[vertices].erase(unique(unique_vertices[vertices].begin(), unique_vertices[vertices].end()),unique_vertices[vertices].end());
				
				int sz = cur_comp.size();
				for(int uv:unique_vertices[vertices]){
					if(uv != vertices)
					{
						for(auto v:reachv[uv])
						{
							sz--;
							children.push_back(v);
						}
					}
				}
				to_reach.push_back(vertices);
			}
		}
		else if(v != par && entry[v] < entry[u])
		{
			st.push_back({u,v});
			low[u] = min(low[u], entry[v]);
		}
	}
	reachv[u] = children;
	reach[u].clear();
	sort(reachv[u].begin(), reachv[u].end());
	reachv[u].erase(unique(reachv[u].begin(), reachv[u].end()),reachv[u].end());
	for(int i:reachv[u]) 
		reach[u].push_back(x[i]);
	
	vis[u] = 3;
}

void prelim_dfs(int u){
	vis[u] = 1;
	cur_comp.push_back(u);
	for(int v:g[u]){
		if(!vis[v]){
			prelim_dfs(v);
		}
	}
}

int main(int argc, char **argv)
{
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

	omp_set_num_threads(10);

	string input = argv[1];
    string output = argv[2];

    ifstream fin (input);
    ofstream fout (output);

	fin >> n >> m;
	vertices = n;
	for(int i=0;i<=n;++i)
		rep.push_back(i);
	g.resize(n+1);
	tmp_g.resize(2*m+1);
	x.resize(2*m+1);
	contrib.resize(n+1);
	reach.resize(2*m+1);
	reachsuf.resize(2*m+1);
	reachv.resize(n+1);
	vis.resize(n+1);
	low.resize(n+1);
	entry.resize(n+1);
	in_otherbccs.resize(n+1,0);
	perc.resize(n+1);
	global_pc.resize(n+1,0);
	unique_vertices.resize(2*n);
	
	timer = 0;
	sum_x = 0;

	for(int i=1;i<=n;i++){
		x[i] = 1.0/(double)(i);
	}
	for(int i=0;i<m;i++){
		int u,v;
		fin >> u >> v;
		if(u != v)
		g[u].push_back(v);
		g[v].push_back(u);
	}
	
    auto t1 = std::chrono::high_resolution_clock::now();

    for(int i=1;i<=n;++i)
		perc[i] = {x[i],i};
	sort(perc.begin(),perc.end());
	long double carry = 0;
	for(int i=1;i<=n;++i)
	{
		contrib[perc[i].second] = (long double)(i-1)*perc[i].first-carry;
		carry += perc[i].first;
		sum_x += contrib[perc[i].second];
	}
	carry = 0;
	for(int i=n;i>=1;i--)
	{
		contrib[perc[i].second] += carry-(long double)(n-i)*perc[i].first;
		carry += perc[i].first;
	}
	for(int i=1;i<=n;i++){
		if(!vis[i]){
			cur_comp.clear();
			prelim_dfs(i);
			dfs(i,0);
			#pragma omp parallel
			{
				vector<bool> mrk(n);
				#pragma omp for
				for(int j=0;j<to_reach.size();++j)
				{
					int u = to_reach[j];
					int sz = cur_comp.size();
					for(int uv:unique_vertices[u]){
						if(uv != u)
						{
							for(auto v:reachv[uv])
							{
								sz--;
								mrk[v] = 1;
							}
						}
					}
					
					reach[u].resize(sz);
					int index = 0;
					for(int i:cur_comp)
					{
						if(!mrk[i])
							reach[u][index++] = x[i];
						else
							mrk[i] = 0;
					}
				}
			}
			to_reach.clear();
		}
	}

	tmp_g.resize(vertices+1);
	pc.resize(vertices+1,0);
	d.resize(vertices+1,-1);
	p.resize(vertices+1);
	sigma.resize(vertices+1,0);
	delta.resize(vertices+1,0);
	
	int V, E = 0;
	V = vertices;
	int cnt_reach_vec = 0;
	for(int i=1;i<=V;++i)
	{
		cnt_reach_vec += (int)(reach[i].size());
		E += (int)(tmp_g[i].size());
	}
	E = E/2;

	int *hColumn = new int[2*E];
	int *hRow	 = new int[V+2];
	float *perc  = new float[V+2];

	for(int i=1;i<=V;++i)
		perc[i] = x[i];
	perc[0] = perc[V+1] = 1.0;

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
	float *intm_reach = new float[cnt_reach_vec];
	int *pointers = new int[V+2];

	int index = 0;
	for(int i=1;i<=V;++i)
	{
		pointers[i] = index;
		for(int j=0;j<(int)reach[i].size();++j)
		{
			intm_reach[index] = reach[i][j];
			index++;
		}
	}
	pointers[V+1] = index;
	
	float *delta, *Paths, *dCentrality, *perc_state, *reach_vec, *reach_suffixsum;
	node *Parents;
	int *dColumn, *dRow, *Distance, *Queue, *crr, *starters;

	cudaMalloc((void**)&dRow,    		 sizeof(int)*(V+2));
	cudaMalloc((void**)&dCentrality,	 sizeof(int)*(V+2));
	cudaMalloc((void**)&perc_state,		 sizeof(float)*(V+2));
	cudaMalloc((void**)&dColumn, 		 sizeof(int)*(2*E));
	cudaMalloc((void**)&crr,    		 sizeof(int)*(V+2)*NUM_BLOCKS);
	cudaMalloc((void**)&Queue,    		 sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Distance,		 sizeof(int)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Paths,			 sizeof(float)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&delta,			 sizeof(float)*(V+1)*NUM_BLOCKS);
	cudaMalloc((void**)&Parents,		 sizeof(node)*(E+1)*NUM_BLOCKS);
	cudaMalloc((void**)&reach_vec,		 sizeof(float)*(cnt_reach_vec));
	cudaMalloc((void**)&reach_suffixsum, sizeof(float)*(cnt_reach_vec));
	cudaMalloc((void**)&starters, 		 sizeof(int)*(V+2));

	cudaMemcpy(dRow, hRow, sizeof(int)*(V+2),cudaMemcpyHostToDevice);
	cudaMemcpy(dColumn, hColumn, sizeof(int)*(2*E), cudaMemcpyHostToDevice);
	cudaMemcpy(perc_state, perc, sizeof(float)*(V+2),cudaMemcpyHostToDevice);
	cudaMemcpy(reach_vec, intm_reach, sizeof(float)*(cnt_reach_vec),cudaMemcpyHostToDevice);
	cudaMemcpy(starters, pointers, sizeof(int)*(V+2),cudaMemcpyHostToDevice);
	gpuErrchk( cudaPeekAtLastError() );

	auto k1 = std::chrono::high_resolution_clock::now();
	auto preprocessing_duration = std::chrono::duration_cast<std::chrono::microseconds>( k1 - t1 ).count();
	cerr << "preprocessing time : " << preprocessing_duration << " mu.s." << endl;
	brandes <<<NUM_BLOCKS, NUM_THREADS, 32>>> (V, E, dColumn, dRow, Distance, Queue, Paths, delta,
				Parents, dCentrality, crr, perc_state, reach_vec, reach_suffixsum, starters);

	cudaDeviceSynchronize();
	gpuErrchk( cudaPeekAtLastError() );
	cudaDeviceSynchronize();
	float *Centrality = new float[V+1];
	cudaMemcpy(Centrality, dCentrality, sizeof(float)*(V+1), cudaMemcpyDeviceToHost);

	for(int i=1;i<=V;++i)
		global_pc[rep[i]] += Centrality[i];
	for(int i=1;i<=n;++i)
		global_pc[i] /= (sum_x - contrib[i]);
	
	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	cerr << "total time : " << duration << " mu.s." << endl;

	for(int i=1;i<=n;i++){
		fout << global_pc[i] << endl;
	}
	
	delete[] hRow;
	delete[] hColumn;

	cudaFree(Queue);
	cudaFree(dRow);
	cudaFree(dColumn);
	cudaFree(Distance);
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

	