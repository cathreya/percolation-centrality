#include <bits/stdc++.h>
#include <chrono>
#include <omp.h>
using namespace std;

// compile : mpic++ -fopenmp -static-libstdc++ <file_name>.cpp

const int MAX_THREADS = 48;

int n,m;
int numthreads;
int timer[MAX_THREADS];
vector<double> pc[MAX_THREADS],x, reach[MAX_THREADS], reachcnt[MAX_THREADS];
vector<int> procs;
vector<pair<int,int> > st[MAX_THREADS];
vector<int> vis[MAX_THREADS],low[MAX_THREADS],entry[MAX_THREADS],cur_comp[MAX_THREADS];
vector<vector<int> > g, tmp_g[MAX_THREADS];
double sum_x;

std::vector<std::vector<int> > p[MAX_THREADS];
std::vector<int> sigma[MAX_THREADS], d[MAX_THREADS];
std::vector<double> delta[MAX_THREADS], orig_delta[MAX_THREADS];
std::vector<int> qs[MAX_THREADS];

std::vector<vector<int> > BCC;
int cnt;
int second_count[MAX_THREADS];
std::vector<int> whl,resp;

int divp;
int total;
long long bcc_sum;
int bcc_mx;
int second_total[MAX_THREADS];

double watch[MAX_THREADS];

void compute_source_perc(int src, std::vector<std::vector<int> > &g,int world_rank){
	qs[world_rank].push_back(src);
	int it = 0;
	d[world_rank][src] = 0;
	sigma[world_rank][src] = 1;
	while(it != (int)qs[world_rank].size()){
		int u = qs[world_rank][it];
		it += 1;
		for(int v:g[u]){
			if(d[world_rank][v] < 0){
				qs[world_rank].push_back(v);
				d[world_rank][v] = d[world_rank][u] + 1;
			}
			if(d[world_rank][v] == d[world_rank][u] + 1){
				sigma[world_rank][v] += sigma[world_rank][u];
				p[world_rank][v].push_back(u);
			}
		}
	}
	while(!qs[world_rank].empty()){
		int v = qs[world_rank].back();
		qs[world_rank].pop_back();
		for(int u:p[world_rank][v]){
			delta[world_rank][u] += (double)(sigma[world_rank][u])/sigma[world_rank][v] * (delta[world_rank][v]+1);
		}
		if(v != src){
			pc[world_rank][v] += (delta[world_rank][v] * reach[world_rank][src]);
		}
		d[world_rank][v] = -1;
		sigma[world_rank][v] = 0;
		p[world_rank][v].clear();
		delta[world_rank][v] = orig_delta[world_rank][v];
	}
}

void distributive_dfs(int u, int par){
	timer[0]++;
	entry[0][u] = timer[0];
	vis[0][u] = 2;
	low[0][u] = entry[0][u];
	for(int v:g[u]){
		if(vis[0][v]< 2){
			st[0].push_back({u,v});
			distributive_dfs(v, u);
			low[0][u] = min(low[0][u], low[0][v]);
			if(low[0][v] >= entry[0][u]){
				vector<int> unique_vertices;
				while(st[0].back() != make_pair(u,v)){
					int p = st[0].back().first;
					int q = st[0].back().second;
					st[0].pop_back();
					unique_vertices.push_back(p);
					unique_vertices.push_back(q);
				}
				unique_vertices.push_back(v);
				unique_vertices.push_back(u);
				sort(unique_vertices.begin(), unique_vertices.end());
				unique_vertices.erase(unique(unique_vertices.begin(), unique_vertices.end()),unique_vertices.end());
				st[0].pop_back();
				for(auto uv:unique_vertices)
				{
					BCC[cnt].push_back(uv);
					total++;
				}
				++cnt;
			}
		}
		else if(v != par && entry[0][v] < entry[0][u]){ 
			st[0].push_back({u,v});
			low[0][u] = min(low[0][u], entry[0][v]);
		}
	}
	vis[0][u] = 3;
}


void distributive_prelim_dfs(int u){
	vis[0][u] = 1;
	cur_comp[0].push_back(u);
	for(int v:g[u]){
		if(!vis[0][v]){
			distributive_prelim_dfs(v);
		}
	}
}

void dfs(int u, int par, int world_rank,int &compsz, double &compwt){
	timer[world_rank]++;
	entry[world_rank][u] = timer[world_rank];
	vis[world_rank][u] = 2;
	reach[world_rank][u] = x[u];
	reachcnt[world_rank][u] = 1;
	low[world_rank][u] = entry[world_rank][u];
	double wt_children = x[u];
	int cnt_children = 1;
	for(int v:g[u]){
		if(vis[world_rank][v]< 2){
			st[world_rank].push_back({u,v});
			dfs(v, u, world_rank, compsz, compwt);
			low[world_rank][u] = min(low[world_rank][u], low[world_rank][v]);
			if(low[world_rank][v] >= entry[world_rank][u]){
				vector<int> unique_vertices;
				while(st[world_rank].back() != make_pair(u,v)){
					int p = st[world_rank].back().first;
					int q = st[world_rank].back().second;
					st[world_rank].pop_back();
					unique_vertices.push_back(p);
					unique_vertices.push_back(q);
					tmp_g[world_rank][p].push_back(q);
					tmp_g[world_rank][q].push_back(p);
				}
				tmp_g[world_rank][u].push_back(v);
				tmp_g[world_rank][v].push_back(u);
				unique_vertices.push_back(v);
				unique_vertices.push_back(u);
				sort(unique_vertices.begin(), unique_vertices.end());
				unique_vertices.erase(unique(unique_vertices.begin(), unique_vertices.end()),unique_vertices.end());
				st[world_rank].pop_back();

				double su = 0;
				int cnt = 0;
				for(int p:unique_vertices){
					if(p==u) continue;
					su += reach[world_rank][p];
					cnt += reachcnt[world_rank][p]; 
					delta[world_rank][p] = (reachcnt[world_rank][p] - 1); 
					orig_delta[world_rank][p] = (reachcnt[world_rank][p] - 1); 
				}

				if(whl[second_count[world_rank]] >= 0 && whl[second_count[world_rank]] != world_rank)
				{
					second_total[world_rank] += (int)(unique_vertices.size());
					cnt_children += cnt;
					wt_children += su;
					reach[world_rank][u] = x[u];
					reachcnt[world_rank][u] = 1;
					tmp_g[world_rank][u].clear();
					++second_count[world_rank];
					continue;
				}

				cnt_children += cnt;
				wt_children += su;
				reach[world_rank][u] = compwt - su;
				reachcnt[world_rank][u] = compsz - cnt;
				delta[world_rank][u] = (reachcnt[world_rank][u]-1);
				orig_delta[world_rank][u] = (reachcnt[world_rank][u]-1);

				if(whl[second_count[world_rank]] == -1)
				{
					for(int src:unique_vertices)
					{
						if(resp[second_total[world_rank]] == world_rank)
							compute_source_perc(src, tmp_g[world_rank], world_rank);
						second_total[world_rank]++;
					}
				}	
				else
				{
					for(int src:unique_vertices)
					{
						compute_source_perc(src, tmp_g[world_rank], world_rank);
						second_total[world_rank]++;
					}
				}

				reach[world_rank][u] = x[u];
				reachcnt[world_rank][u] = 1;
				tmp_g[world_rank][u].clear();
				second_count[world_rank]++;
			}
		}
		else if(v != par && entry[world_rank][v] < entry[world_rank][u]){ //back edge
			st[world_rank].push_back({u,v});
			low[world_rank][u] = min(low[world_rank][u], entry[world_rank][v]);
		}
	}
	reachcnt[world_rank][u] = cnt_children;
	reach[world_rank][u] = wt_children;
	vis[world_rank][u] = 3;
}

void prelim_dfs(int u, int world_rank, int &cnt, double &wt){
	vis[world_rank][u] = 1;
	cnt += 1;
	wt += x[u];
	for(int v:g[u]){
		if(!vis[world_rank][v]){
			prelim_dfs(v,world_rank,cnt,wt);
		}
	}
}

int main( int argc, char **argv ){
	ios::sync_with_stdio(false);
	cin.tie(0);
	cout.tie(0);

    string input = argv[1];
    string output = argv[2];
    int numthreads = atoi(argv[3]);
    omp_set_num_threads(numthreads);

    ifstream fin (input);
    ofstream fout (output);
	fin >> n >> m;
	g.resize(n+1);
	BCC.resize(n+1);
	whl.resize(n+1);
	
	for(int i=0;i<numthreads;++i)
	{
		vis[i].resize(n+1,0);
		timer[i] = 0;
		entry[i].resize(n+1);
		low[i].resize(n+1);
		tmp_g[i].resize(n+1);
		reach[i].resize(n+1);
		reachcnt[i].resize(n+1);
		delta[i].resize(n+1,0);
		orig_delta[i].resize(n+1,0);
		d[i].resize(n+1,-1);
		sigma[i].resize(n+1);
		p[i].resize(n+1);
		pc[i].resize(n+1,0);
	}
	
	double sum_x = 0;
	x.push_back(0.0);
	for(int i=0;i<n;++i)
	{
		double prc = 1.0/(double)(i+1);
		x.push_back(prc);
		sum_x += prc;
	}
	for(int i=0;i<m;i++){
		int u,v;
		fin>>u>>v;
		if(u != v)
		{
			g[u].push_back(v);
			g[v].push_back(u);
		}
	}

    auto t1 = std::chrono::high_resolution_clock::now();

	vector<int> roots;
	for(int i=1;i<=n;i++){
		if(!vis[0][i]){
			roots.push_back(i);
			cur_comp[0].clear();
			distributive_prelim_dfs(i);
			distributive_dfs(i,0);
		}
	}

	resp.resize(total+1,0);
	divp = (total+numthreads-1)/numthreads;
	
	timer[0] = 0;
	for(int i=1;i<=n;++i)
		vis[0][i] = low[0][i] = entry[0][i] = 0;
	while(!st[0].empty())
		st[0].pop_back();

	int it = 0;

	vector<int> dis(numthreads);
	vector<int> procs;
	int sm = 0;
	for(int i=0;i<numthreads-1;++i)
	{
		procs.push_back(i);
		dis[i] = divp;
		sm += divp;
	}
	procs.push_back(numthreads-1);
	dis[numthreads-1] = total-sm;
	int local_cnt = 0, bcc_cnt = 0;

	for(int j=0;j<cnt;++j)
	{
		vector<int> bc = BCC[j];
		int sz = bc.size();
		if(sz == 0)
		{
			it++;
			continue;
		}
		random_shuffle(procs.begin(),procs.end());
		whl[it] = -1;
		for(int i=0;i<numthreads;++i)
		{
			if(dis[procs[i]] >= sz)
			{
				whl[it] = procs[i];
				dis[procs[i]] -= sz;
				local_cnt += sz;
				break;
			}
		}
		if(whl[it] == -1)
		{
			int i = 0;
			for(auto u:bc)
			{
				while(1)
				{
					if(i == numthreads)
						i = 0;
					if(dis[procs[i]] > 0)
					{
						resp[local_cnt] = procs[i];
						local_cnt++;
						dis[procs[i]]--;
						i++;
						break;
					}
					i++;
				}
			}
		}
		it++;
	}
	cur_comp[0].clear();
	
	#pragma omp parallel for 
	for(int i=0;i<numthreads;++i)
	{
		int world_rank = i;
		for(auto u:roots)
		{
			int cnt = 0;
    		double wt = 0;
			prelim_dfs(u,world_rank,cnt,wt);
			dfs(u,0,world_rank,cnt,wt);
			cur_comp[world_rank].clear();
		}
	}

	for(int i=1;i<=n;i++){
		for(int j=1;j<numthreads;++j)
			pc[0][i] += pc[j][i];
		pc[0][i] /= (n-2);
		pc[0][i] /= (sum_x-x[i]);
	}
	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	cerr << "total time : " << duration<< " mu.s." << endl;

	for(int i=1;i<=n;i++){
		fout<<pc[0][i]<<endl;
	}

	return 0;
}