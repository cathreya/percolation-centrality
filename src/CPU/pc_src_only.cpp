#include<iostream>
#include<vector>
#include<stack>
#include<set>
#include<queue>
#include<list>
#include<iomanip>
#include<chrono>
#include<omp.h>
#include<fstream>
using namespace std;

// compile : mpic++ -fopenmp -static-libstdc++ <file_name>.cpp

int n,m;
vector<double> pc,x;
vector<vector<int> > g;
double sum_x;

int vertices;
vector<double> global_pc;

void compute_source_perc(int src, vector<double> x, double *ptr, vector<int> qs, 
	vector<int> sigma, vector<int> d, vector<vector<int> > p, 
	vector<double> del,vector<vector<int> > &g){
	qs.push_back(src);
	int it = 0;
	d[src] = 0;
	sigma[src] = 1;
	while(it != (int)qs.size()){
		int u = qs[it];
		it += 1;
		for(int v:g[u]){
			if(d[v] < 0){
				// q.push(v);
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
		double inc = (double)(del[v]+1)/sigma[v];
		for(int u:p[v]){
			del[u] += sigma[u]*inc;
		}
		if(v != src){
			ptr[v] += (del[v]*x[src]);
		}
		d[v] = -1;
		sigma[v] = 0;
		p[v].clear();
		del[v] = 0;
	}
}

int main( int argc, char **argv ) {
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

    string input = argv[1];
    string output = argv[2];
    int numthreads = atoi(argv[3]);
    omp_set_num_threads(numthreads);

    ifstream fin(input);
    ofstream fout(output);
    fin >> n >> m;
    vertices = n;
	g.resize(n+1);
	pc.resize(n+1,0);
	global_pc.resize(n+1);
	x.resize(n+1);
	
	sum_x = 0;

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

	double *ptr = &pc[0];

	#pragma omp parallel
	{
		vector<int> qs,d(vertices+1,-1),sigma(vertices+1,0);
		vector<vector<int> > p(vertices+1);
		vector<double> del(n+1,0);
		#pragma omp for reduction (+:ptr[:vertices+1]) 
		for(int i=1;i<=vertices;++i)
		{
			if(g[i].empty())
				continue;
			compute_source_perc(i, x, ptr, qs, sigma, d, p, del, g);
		}
	}
	
	for(int i=1;i<=n;i++){
		pc[i] /= (n-2);
		pc[i] /= (sum_x-x[i]);
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	cerr << "total time : " << duration << " mu.s." <<endl;

	for(int i=1;i<=n;i++)
		fout << pc[i] << "\n";

	
	return 0;
}