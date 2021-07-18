#include<bits/stdc++.h>
#include<omp.h>
#include<chrono>
using namespace std;

// compile : mpic++ -fopenmp -static-libstdc++ <file_name>.cpp

const int INF = 1e9;

int N,M;
vector<vector<int> > adj;
vector<double> percolation,contri,global_pc;
vector<double> x; 

double sum; 

string input,output;

void traverse(int src,vector<double> x, vector<vector<int> > &adj,double *ptr)
{	
	int N = (int)x.size()-1;
	queue<int> q;
	stack<int> st;
	vector<int> dist(N+1,-1);
	vector<long double> sig(N+1,0.0),delta(N+1,0.0);
	vector<vector<int> > pr(N+1);

	int u = src;
	q.push(u);
	dist[u] = 0;
	sig[u] = 1.0;

	while(!q.empty())
	{
		u = q.front();
		q.pop();
		st.push(u);

		for(auto v:adj[u])
		{
			if(dist[v] < 0)
			{
				dist[v] = dist[u]+1;
				q.push(v);
			}
			if(dist[v] == dist[u]+1)
			{
				pr[v].push_back(u);
				sig[v] = sig[u]+sig[v];
			}
		}
	}

	while(!(st.empty()))
	{
		u = st.top();
		st.pop();
		for(auto p:pr[u])
		{
			double g;
			g = sig[p]/sig[u];
			g = g*(max(x[src]-x[u],(double)(0.0))+delta[u]);
			delta[p] = delta[p]+g;
		}
		if(u != src)
			ptr[u] += delta[u];
		pr[u].clear();
		delta[u] = 0;
		sig[u] = 0;
		dist[u] = -1;
	}
}

int main( int argc, char **argv ) {
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

    input = argv[1];
    output = argv[2];
    int numthreads = atoi(argv[3]);
    omp_set_num_threads(numthreads);

    ifstream fin(input);
    ofstream fout(output);
	fin >> N >> M; 
	int u,v;
	double p;
	adj.resize(N+1);
	x.push_back(0);
	for(int i=0;i<N;++i)
	{
		double prc = 1.0/(double)(i+1);
		x.push_back(prc);
	}
	for(int i=0;i<M;++i)
	{
		fin >> u >> v;
		adj[u].push_back(v);
		adj[v].push_back(u);
	}

	global_pc.resize(N+1);
	contri.resize(N+1);
	percolation.resize(N+1);
	fill(contri.begin(),contri.end(),0);
	fill(percolation.begin(),percolation.end(),0);
	fill(global_pc.begin(),global_pc.end(),0);

    auto t1 = std::chrono::high_resolution_clock::now();

	double *ptr = &percolation[0];
	#pragma omp parallel for reduction (+:ptr[:N+1]) 
	for(int i=1;i<=N;++i)
	{
		traverse(i,x,adj,ptr);
	}
	
	ptr = &contri[0];
	#pragma omp parallel for reduction (+:ptr[:N+1],sum) collapse(2)
	for(int i=1;i<=N;++i)
	{
		for(int j=1;j<=N;++j)
		{
			long double r = max(x[i]-x[j],(double)(0.0));
			sum += r;
			contri[i] += r;
			contri[j] += r;
		}
	}

	for(int i=1;i<=N;++i)
		percolation[i] /= (sum-contri[i]);


	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	cerr << "total time : " << duration << " mu.s." <<endl;
	for(int i=1;i<=N;++i)
	{
		percolation[i] /= (sum-contri[i]);
		fout << percolation[i] << "\n";
	}

	return 0;
}