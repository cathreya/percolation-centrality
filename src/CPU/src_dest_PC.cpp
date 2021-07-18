#include<bits/stdc++.h>
#include<chrono>
#include<omp.h>
using namespace std;

// compile : mpic++ -fopenmp -static-libstdc++ <file_name>.cpp

const int MAX_THREADS = 48;

int n,m;
int timer[MAX_THREADS];
vector<long double> pc[MAX_THREADS],x, contrib, global_pc;
vector<vector<double> > reach[MAX_THREADS], reachsuf[MAX_THREADS];
vector<vector<int> > reachv[MAX_THREADS];
vector<pair<int,int> > st[MAX_THREADS];
vector<int> vis[MAX_THREADS],low[MAX_THREADS],entry[MAX_THREADS], cur_comp[MAX_THREADS];
vector<vector<int> > g, tmp_g[MAX_THREADS];
double sum_x;

std::vector<std::vector<int> > p[MAX_THREADS];
std::vector<int> sigma[MAX_THREADS], d[MAX_THREADS];
std::vector<double> delta[MAX_THREADS];
std::vector<int> qs[MAX_THREADS];
std::vector<pair<long double,int> > perc;

std::vector<vector<int> > BCC;
int cnt;
int second_count[MAX_THREADS];
std::vector<int> whl,resp;
std::vector<bool> in_otherbccs[MAX_THREADS];

int divp;
int total;
int second_total[MAX_THREADS];

double watch[MAX_THREADS];

double R(double u){
	if(u>0) return u;
	return 0;
}

void compute_source_perc(int src, std::vector<std::vector<int> > &g, int world_rank){
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
			int ind = lower_bound(reach[world_rank][src].begin(), reach[world_rank][src].end(), x[v]) - reach[world_rank][src].begin();
			double add = 0;
			if(ind < (int)reach[world_rank][src].size()){
				int cnt = (int)reach[world_rank][src].size() - ind;
				add = (reachsuf[world_rank][src][ind] - x[v]*cnt);
			}
			delta[world_rank][u] += (double)(sigma[world_rank][u])/sigma[world_rank][v] * (delta[world_rank][v]+add	);
		}
		if(v != src){
			pc[world_rank][v] += delta[world_rank][v];
		}
		d[world_rank][v] = -1;
		sigma[world_rank][v] = 0;
		p[world_rank][v].clear();
		delta[world_rank][v] = 0;
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


void dfs(int u, int par, int world_rank){
	timer[world_rank]++;
	entry[world_rank][u] = timer[world_rank];
	vis[world_rank][u] = 2;
	low[world_rank][u] = entry[world_rank][u];
	vector<int> children = {u};
	for(int v:g[u]){
		if(vis[world_rank][v]< 2){
			st[world_rank].push_back({u,v});
			dfs(v, u, world_rank);
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

				if(whl[second_count[world_rank]] >= 0 && whl[second_count[world_rank]] != world_rank)
				{
					for(int uv:unique_vertices){
						second_total[world_rank]++;
						if(uv != u)	children.insert(children.begin(), reachv[world_rank][uv].begin(), reachv[world_rank][uv].end());
					}
					st[world_rank].pop_back();	
					tmp_g[world_rank][u].clear();
					++second_count[world_rank];
					continue;
				}

				for(int uv:unique_vertices){
					if(uv != u)
					{
						for(auto v:reachv[world_rank][uv])
						{
							children.push_back(v);
							in_otherbccs[world_rank][v] = 1;
						}
					}
				}

				reach[world_rank][u].clear();
				for(int i:cur_comp[world_rank])
				{
					if(!in_otherbccs[world_rank][i])
						reach[world_rank][u].push_back(x[i]);
					else
						in_otherbccs[world_rank][i] = 0;
				}
				
				sort(reach[world_rank][u].begin(),reach[world_rank][u].end());
				reachsuf[world_rank][u] = reach[world_rank][u];
				for(int i=(int)reach[world_rank][u].size()-2; i>=0;i--){
					reachsuf[world_rank][u][i] += reachsuf[world_rank][u][i+1];
				}
				st[world_rank].pop_back();	

				if(whl[second_count[world_rank]] == -1)
				{
					for(int src:unique_vertices)
					{
						if(resp[second_total[world_rank]] == world_rank)
						{
							for(int repr:unique_vertices){
								for(double xi:reach[world_rank][repr]){
									int ind = lower_bound(reach[world_rank][src].begin(), reach[world_rank][src].end(), xi) - reach[world_rank][src].begin();
									if(ind < (int)reach[world_rank][src].size()){
										int cnt = (int)reach[world_rank][src].size() - ind;
										delta[world_rank][repr] += reachsuf[world_rank][src][ind] - xi*cnt; 
									}
								}
								int ind = lower_bound(reach[world_rank][src].begin(), reach[world_rank][src].end(), x[repr]) - reach[world_rank][src].begin();
								if(ind < (int)reach[world_rank][src].size()){
									int cnt = (int)reach[world_rank][src].size() - ind;
									delta[world_rank][repr] -= reachsuf[world_rank][src][ind] - x[repr]*cnt; 
								}
							}
							compute_source_perc(src, tmp_g[world_rank], world_rank);
						}
						second_total[world_rank]++;
					}
				}	
				else
				{
					for(int src:unique_vertices)
					{
						for(int repr:unique_vertices){
							for(double xi:reach[world_rank][repr]){
								int ind = lower_bound(reach[world_rank][src].begin(), reach[world_rank][src].end(), xi) - reach[world_rank][src].begin();
								if(ind < (int)reach[world_rank][src].size()){
									int cnt = (int)reach[world_rank][src].size() - ind;
									delta[world_rank][repr] += reachsuf[world_rank][src][ind] - xi*cnt; 
								}
							}
							int ind = lower_bound(reach[world_rank][src].begin(), reach[world_rank][src].end(), x[repr]) - reach[world_rank][src].begin();
							if(ind < (int)reach[world_rank][src].size()){
								int cnt = (int)reach[world_rank][src].size() - ind;
								delta[world_rank][repr] -= reachsuf[world_rank][src][ind] - x[repr]*cnt; 
							}
						}
						compute_source_perc(src, tmp_g[world_rank], world_rank);
						second_total[world_rank]++;
					}
				}

				tmp_g[world_rank][u].clear();
				second_count[world_rank]++;
			}
		}
		else if(v != par && entry[world_rank][v] < entry[world_rank][u]){ 
			st[world_rank].push_back({u,v});
			low[world_rank][u] = min(low[world_rank][u], entry[world_rank][v]);
		}
	}

	reachv[world_rank][u] = children;
	reach[world_rank][u].clear();
	sort(reachv[world_rank][u].begin(), reachv[world_rank][u].end());
	reachv[world_rank][u].erase(unique(reachv[world_rank][u].begin(), reachv[world_rank][u].end()),reachv[world_rank][u].end());
	for(int i:reachv[world_rank][u]) {
		reach[world_rank][u].push_back(x[i]);
	}
	sort(reach[world_rank][u].begin(), reach[world_rank][u].end());
	reachsuf[world_rank][u] = reach[world_rank][u];
	for(int i=(int)reach[world_rank][u].size()-2; i>=0;i--){
		reachsuf[world_rank][u][i] += reachsuf[world_rank][u][i+1];
	}
	vis[world_rank][u] = 3;
}


void prelim_dfs(int u,int world_rank){
	vis[world_rank][u] = 1;
	cur_comp[world_rank].push_back(u);
	for(int v:g[u]){
		if(!vis[world_rank][v]){
			prelim_dfs(v,world_rank);
		}
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

    ifstream fin (input);
    ofstream fout (output);
	fin >> n >> m;
	g.resize(n+1);
	contrib.resize(n+1);
	global_pc.resize(n+1,0);
	BCC.resize(n+1);
	whl.resize(n+1);
	perc.resize(n+1);
	
	for(int i=0;i<numthreads;++i)
	{
		vis[i].resize(n+1,0);
		timer[i] = 0;
		entry[i].resize(n+1);
		low[i].resize(n+1);
		tmp_g[i].resize(n+1);
		in_otherbccs[i].resize(n+1,0);
		reach[i].resize(n+1);
		reachsuf[i].resize(n+1);
		reachv[i].resize(n+1);
		delta[i].resize(n+1,0);
		d[i].resize(n+1,-1);
		sigma[i].resize(n+1);
		p[i].resize(n+1);
		pc[i].resize(n+1,0);
	}
	
	sum_x = 0;
	x.push_back(0.0);
	for(int i=0;i<n;++i)
	{
		double prc = 1.0/(double)(i+1);
		x.push_back(prc);
	}
	for(int i=0;i<m;i++){
		int u,v;
		fin>>u>>v;
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
	int local_cnt = 0;

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
			prelim_dfs(u,world_rank);
			dfs(u,0,world_rank);
			cur_comp[world_rank].clear();
		}
	}
	

	for(int i=1;i<=n;i++)
	{
		for(int j=0;j<numthreads;++j)
			global_pc[i] += pc[j][i];
		global_pc[i] /= (sum_x - contrib[i]);
	}

	auto t2 = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();
	cerr << "total time : " << duration<< " mu.s." << endl;

	for(int i=1;i<=n;i++)
		fout<<global_pc[i]<<endl;

	return 0;
}