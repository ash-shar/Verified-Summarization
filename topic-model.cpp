#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#include <bits/stdc++.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <map>
#include <queue>
#include <random>
#include <tr1/cmath>
#include <ctime>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

using namespace std;

typedef struct interaction{
	vector<int> c_text;
	vector<int> e_text;
	int behav;
	double ts;
}interaction;

double MIN_LOG = log(DBL_MIN);
int RESOLUTION = 100;

int num_top_words = 50;
// int num_top_e_words = 20;

// Model Parameters
double alpha_k = 0.01;
double alpha_w = 0.01;
double alpha_b = 0.01;			// Changed Later to 50/B
double alpha_s = 0.01;
double alpha_sk = 0.01;
double alpha_sp = 0.01;


vector<double> alpha_kt;
vector<double> alpha_st;
vector<double> beta_kt;
vector<double> beta_st;

int K;
int T;

int numIter;


// Behavior and Word Mappings to Idx
map<string, int> c_wordMap;
map<int, string> c_wordMapRev;
map<string, int> e_wordMap;
map<int, string> e_wordMapRev;
map<string, int> behavMap;
map<int, string> behavMapRev;


// Vocabulary and Behavior Set
set<string> c_vocab;
set<string> e_vocab;
set<string> behav_set;

// Dataset
vector<interaction> corpus;	// List of Interactions in the courpus

int V;						// Size of Vocabulary
int B;						// Number of Behavior
int C;						// Number of Interactions (Size of the corpus)
int S;						// Expression Words Vocab


// Count Matrices
vector<vector<int>> nkb;
vector<vector<int>> nkw;
vector<vector<int>> nsw;
vector<vector<int>> nsk;

// Count Arrays
vector<int> nkwsum;
vector<int> nkbsum;
vector<int> nswsum;
vector<int> nsksum;
vector<int> nk;
vector<int> ns;

vector<int> intr_w_topic;
vector<int> intr_s_topic;


// Probability Matrices
vector<vector<double>> PHI_K_W_num;
vector<vector<double>> PHI_K_B_num;
vector<vector<double>> PHI_S_W_num;
vector<vector<double>> PHI_S_K_num;
vector<double> EXRPESSION_PRIOR;
vector<double> TOPIC_PRIOR;

vector<double> PHI_K_W_denom;
vector<double> PHI_S_W_denom;
vector<double> PHI_K_B_denom;
vector<double> PHI_S_K_denom;


vector<vector<double>> pkt;
vector<vector<double>> pst;

unsigned seed = chrono::system_clock::now().time_since_epoch().count();
// random_device rd;
mt19937 gen(seed);


void make_dir(string dir_path)
{
	struct stat st = {0};

	if (stat(dir_path.c_str(), &st) == -1) {
		mkdir(dir_path.c_str(), 0700);
	}
}

void split(const std::string &s, char delim, std::vector<std::string> &elems) {
	std::stringstream ss;
	ss.str(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
}

std::vector<std::string> split(const std::string &s, char delim) {
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}


double mean(vector<double> v)
{
	double sum=0;
	for(int i=0;i<v.size();i++) sum+=v[i];

	return (1.0*sum)/v.size();
}

double standard_deviation(vector<double> v, double ave)
{

	double E=0;
	// Quick Question - Can vector::size() return 0?
	double inverse = 1.0 / static_cast<double>(v.size());
	for(int i=0; i<v.size(); i++)
	{
		E += pow( (v[i] - ave), 2);
	}
	return sqrt(inverse * E);
}


void quicksort(vector<pair<int, double> > & vect, int left, int right) {
	int l_hold, r_hold;
	pair<int, double> pivot;

	l_hold = left;
	r_hold = right;    
	int pivotidx = left;
	pivot = vect[pivotidx];

	while (left < right) {
		while (vect[right].second <= pivot.second && left < right) {
			right--;
		}
		if (left != right) {
			vect[left] = vect[right];
			left++;
		}
		while (vect[left].second >= pivot.second && left < right) {
			left++;
		}
		if (left != right) {
			vect[right] = vect[left];
			right--;
		}
	}

	vect[left] = pivot;
	pivotidx = left;
	left = l_hold;
	right = r_hold;

	if (left < pivotidx) {
		quicksort(vect, left, pivotidx - 1);
	}
	if (right > pivotidx) {
		quicksort(vect, pivotidx + 1, right);
	}
}


int sample_from_prob(vector<double> probs)
{
	srand(time(0));

	discrete_distribution<> d(probs.begin(), probs.end());

	int sampled_idx = d(gen);

	return sampled_idx;
}

vector<double> handle_underflow(vector<double> probs)
{
	int num = probs.size();

	double max_prob = *max_element(probs.begin(), probs.begin()+num);

	for(int j = 0; j < num; ++j)
	{
		
		if(probs[j] > 1) probs[j] = 0.0;
		else
		{
			probs[j] = probs[j]-max_prob;

			if(probs[j] < MIN_LOG)
			{
				probs[j] = 0.0;
			}

			else
			{
				probs[j] = exp(probs[j]);
			}
		}
	}

	return probs;
}



void compute_beta_distribution()
{

	vector<vector<double>> topic_ts, express_ts;

	topic_ts.resize(K);

	express_ts.resize(T);

	for(int c = 0; c < C; ++c)
	{
		interaction curr_intr = corpus[c];

		int curr_topic = intr_w_topic[c];
		int curr_express = intr_s_topic[c];

		topic_ts[curr_topic].push_back(curr_intr.ts);
		express_ts[curr_express].push_back(curr_intr.ts);
	}


	for(int k = 0; k < K; ++k)
	{
		if(topic_ts[k].size() == 0)
		{
			alpha_kt[k] = 1.0;
			beta_kt[k] = 1.0;
			continue;
		}

		double t_g = mean(topic_ts[k]);

		double s_g = standard_deviation(topic_ts[k], t_g);

		if(s_g != 0)
		{
			double curr_alpha = t_g * ( ((t_g * (1 - t_g)) / (s_g*s_g)) - 1);

			double curr_beta = (1- t_g) * ( ((t_g * (1 - t_g)) / (s_g*s_g)) - 1);

			alpha_kt[k] = curr_alpha;
			beta_kt[k] = curr_beta;
		}
	}

	for(int s = 0; s < T; ++s)
	{
		if(express_ts[s].size() == 0)
		{
			alpha_st[s] = 1.0;
			beta_st[s] = 1.0;
			continue;	
		}

		double t_g = mean(express_ts[s]);

		double s_g = standard_deviation(express_ts[s], t_g);

		if(s_g != 0)
		{
			double curr_alpha = t_g * ( ((t_g * (1 - t_g)) / (s_g*s_g)) - 1);

			double curr_beta = (1- t_g) * ( ((t_g * (1 - t_g)) / (s_g*s_g)) - 1);

			alpha_st[s] = curr_alpha;
			beta_st[s] = curr_beta;
		}
	}


	// Compute the probabilities

	for(int k = 0; k < K; ++k)
	{
		for(int t = 1; t < RESOLUTION; ++t)
		{
			double ts = (1.0*t)/(1.0 * RESOLUTION);

			pkt[k][t] = log(pow(ts, alpha_kt[k]-1)) + log(pow(1-ts, beta_kt[k]-1)) - log(tr1::beta(alpha_kt[k], beta_kt[k]));
		}
	}


	for(int s = 0; s < T; ++s)
	{
		for(int t = 1; t < RESOLUTION; ++t)
		{
			double ts = (1.0*t)/(1.0 * RESOLUTION);

			pst[s][t] = log(pow(ts, alpha_st[s]-1)) + log(pow(1-ts, beta_st[s]-1)) - log(tr1::beta(alpha_st[s], beta_st[s]));
		}
	}


}


void decrease_cnts(int c)
{
	int k = intr_w_topic[c];
	int s = intr_s_topic[c];

	ns[s] -= 1;
	nk[k] -= 1;

	TOPIC_PRIOR[k] =  log((nk[k] + alpha_k) / (C - 1 + (K * alpha_k)));
	EXRPESSION_PRIOR[s] = log((ns[s] + alpha_sp) / (C - 1 + T * alpha_sp));

	nsk[s][k] -= 1;
	nsksum[s] -= 1;

	PHI_S_K_num[s][k] = log((nsk[s][k] + alpha_sk));
	PHI_S_K_denom[s] = log((nsksum[s] + K * alpha_sk));

	interaction curr_intr = corpus[c];

	int b = curr_intr.behav;

	nkb[k][b] -= 1;
	nkbsum[k] -= 1;

	PHI_K_B_num[k][b] = log((nkb[k][b] + alpha_b));
	PHI_K_B_denom[k] = log((nkbsum[k] + (B * alpha_b)));


	int W = curr_intr.c_text.size();

	for(int j = 0; j < W; ++j)
	{
		int w = curr_intr.c_text[j];
		nkw[k][w] -= 1;
		nkwsum[k] -= 1;

		PHI_K_W_num[k][w] = log((nkw[k][w] + alpha_w));
	}		

	PHI_K_W_denom[k] = log((nkwsum[k] + (V * alpha_w)));

	W = curr_intr.e_text.size();

	for(int j = 0; j < W; j++)
	{
		int w = curr_intr.e_text[j];
		nsw[s][w] -= 1;
		nswsum[s] -= 1;

		PHI_S_W_num[s][w] = log((nsw[s][w] + alpha_s));		
	}

	PHI_S_W_denom[s] = log((nswsum[s] + (S * alpha_s)));
}


void increase_cnts(int c, int k, int s)
{
	intr_w_topic[c] = k;
	intr_s_topic[c] = s;

	ns[s] += 1;
	nk[k] += 1;

	TOPIC_PRIOR[k] =  log((nk[k] + alpha_k) / (C + (K * alpha_k)));
	EXRPESSION_PRIOR[s] = log((ns[s] + alpha_sp) / (C - 1 + T * alpha_sp));

	nsk[s][k] += 1;
	nsksum[s] += 1;

	PHI_S_K_num[s][k] = log((nsk[s][k] + alpha_sk));
	PHI_S_K_denom[s] = log((nsksum[s] + K * alpha_sk));

	interaction curr_intr = corpus[c];

	int b = curr_intr.behav;

	nkb[k][b] += 1;
	nkbsum[k] += 1;

	PHI_K_B_num[k][b] = log((nkb[k][b] + alpha_b));
	PHI_K_B_denom[k] = log((nkbsum[k] + (B * alpha_b)));


	int W = curr_intr.c_text.size();

	for(int j = 0; j < W; ++j)
	{
		int w = curr_intr.c_text[j];
		nkw[k][w] += 1;
		nkwsum[k] += 1;

		PHI_K_W_num[k][w] = log((nkw[k][w] + alpha_w));
	}		

	PHI_K_W_denom[k] = log((nkwsum[k] + (V * alpha_w)));


	W = curr_intr.e_text.size();

	for(int j = 0; j < W; j++)
	{
		int w = curr_intr.e_text[j];
		nsw[s][w] += 1;
		nswsum[s] += 1;

		PHI_S_W_num[s][w] = log((nsw[s][w] + alpha_s));		
	}

	PHI_S_W_denom[s] = log((nswsum[s] + (S * alpha_s)));
}

void gibbs_sampling_iteration(int itr)
{
	for(int c = 0; c < C; ++c)
	{
		decrease_cnts(c);

		vector<double> p_k, p_s;

		p_k.resize(K, -1);
		p_s.resize(T, -1);

		interaction curr_intr = corpus[c];

		int b = curr_intr.behav;
		
		double ts = curr_intr.ts;

		int t = int(ts*RESOLUTION);

		int W = curr_intr.c_text.size();

		for(int k = 0; k < K; ++k)
		{
			p_k[k] = TOPIC_PRIOR[k] + pkt[k][t] + PHI_K_B_num[k][b] - PHI_K_B_denom[k];

			for(int j = 0; j < W; ++j)
			{
				int w = curr_intr.c_text[j];
				p_k[k] += PHI_K_W_num[k][w];
				p_k[k] -= PHI_K_W_denom[k];
			}
		}

		p_k = handle_underflow(p_k);

		int new_topic = sample_from_prob(p_k);

		W = curr_intr.e_text.size();

		for(int s = 0; s < T; ++s)
		{
			p_s[s] = EXRPESSION_PRIOR[s] + PHI_S_K_num[s][new_topic] - PHI_S_K_denom[s]; // + pst[s][t] 

			for(int j = 0; j < W; ++j)
			{
				int w = curr_intr.e_text[j];
				p_s[s] += PHI_S_W_num[s][w];
				p_s[s] -= PHI_S_W_denom[s];
			}
		}

		p_s = handle_underflow(p_s);

		int new_express = sample_from_prob(p_s);

		if(new_express != 0 && itr>450)
		{
			cout<<"SET: "<<new_express<<endl;
		}

		increase_cnts(c, new_topic, new_express);
	}
}

void run_topic_model()
{
	for(int itr = 0; itr < numIter; itr++)
	{
		compute_beta_distribution();

		gibbs_sampling_iteration(itr);

		cout<<"\rIter Number: "<<itr<<" / "<<numIter<<endl;
	}
	cout<<"Topic Model Done"<<endl;
}


void readVocab(const char* filename)
{
	int counter = 0;
	double ts;
	ifstream infile(filename);
	string line;

	while(getline(infile,line)){
		// Tweet Id, Content Text, Expression Text, Behav, Timestamp
		vector<string> cols = split(line, '\t');
		string c_text = cols[1];
		string e_text = cols[2];
		string behav = cols[3];
		stringstream(cols[4]) >> ts;

		vector<string> text_units = split(c_text, ' ');

		for(int i = 0; i < text_units.size(); i++){
			c_vocab.insert(text_units[i]);
		}

		text_units.clear();

		text_units = split(e_text, ' ');

		for(int i = 0; i < text_units.size(); i++){
			e_vocab.insert(text_units[i]);
		}

		behav_set.insert(behav);
	}

	// Create Word Map
	set<string>::iterator it;
	for (it = c_vocab.begin(); it != c_vocab.end(); it++){
		c_wordMap.insert(pair<string,int>(*it, counter));
		c_wordMapRev.insert(pair<int,string>(counter, *it));
		counter++;
	}

	counter = 0;

	for (it = e_vocab.begin(); it != e_vocab.end(); it++){
		e_wordMap.insert(pair<string,int>(*it, counter));
		e_wordMapRev.insert(pair<int,string>(counter, *it));
		counter++;
	}

	// Create Behavior Map
	counter = 0;
	for(it = behav_set.begin(); it != behav_set.end(); it++)
	{
		behavMap.insert(pair<string,int>(*it, counter));
		behavMapRev.insert(pair<int,string>(counter, *it));
		counter++;
	}

}

void readCorpus(const char *corpus_path)
{
	ifstream infile(corpus_path);
	string line;

	int idx = 0;

	while (getline(infile,line)){
		// Tweet Id, Content Text, Expression Text, Behav, Timestamp
		interaction curr_intr;
		vector<string> cols = split(line, '\t');
		string c_text = cols[1];
		string e_text = cols[2];
		curr_intr.behav = behavMap.find(cols[3])->second;
		stringstream(cols[4]) >> curr_intr.ts;

		curr_intr.c_text.clear();
		curr_intr.e_text.clear();

		vector<string> text_units = split(c_text, ' ');

		// cout<<text_units.size()<<endl;

		for (int i = 0; i < text_units.size(); i++){
			curr_intr.c_text.push_back(c_wordMap.find(text_units[i])->second);
		}

		text_units.clear();
		text_units = split(e_text, ' ');

		for (int i = 0; i < text_units.size(); i++){
			curr_intr.e_text.push_back(e_wordMap.find(text_units[i])->second);
		}

		corpus.push_back(curr_intr);
		
		idx+=1;
	}
}

void initialize(const char *corpus_path)
{
	cout<<"Initialization Started"<<endl;

	// Load the Behavior and Word Vocabulary
	readVocab(corpus_path);

	B = behav_set.size();
	S = e_vocab.size();
	V = c_vocab.size();

	cout<<"B: "<<B<<endl;
	cout<<"S: "<<S<<endl;
	cout<<"V: "<<V<<endl;


	// Load the Corpus
	readCorpus(corpus_path);

	cout<<"Corpus reading done"<<endl;

	// Determine the size of vocabulary, number of behaviors, number of Users and the size of the corpus
	C = corpus.size();


	// Resize count matrices and arrays (nkb, nkw, nsw)

	nkb.resize(K);
	nkw.resize(K);

	PHI_K_B_num.resize(K);
	PHI_K_W_num.resize(K);

	PHI_K_W_denom.resize(K, 0.0);
	PHI_K_B_denom.resize(K, 0.0);

	pkt.resize(K);

	for(int k = 0; k < K; ++k)
	{
		nkb[k].resize(B, 0);
		PHI_K_B_num[k].resize(B, 0.0);

		pkt[k].resize(RESOLUTION, 0.0);

		nkw[k].resize(V, 0);
		PHI_K_W_num[k].resize(V, 0.0);

	}

	nkbsum.resize(K, 0);
	nkwsum.resize(K, 0);

	nk.resize(K, 0);

	intr_w_topic.resize(C, -1);
	intr_s_topic.resize(C, -1);

	double topic_prior_prob = log(1.0/K);

	TOPIC_PRIOR.resize(K, topic_prior_prob);

	nsw.resize(T);

	nsk.resize(T);
	nsksum.resize(T, 0.0);

	ns.resize(T, 0);

	nswsum.resize(T, 0);

	PHI_S_W_num.resize(T);
	PHI_S_K_num.resize(T);

	PHI_S_W_denom.resize(T, 0.0);
	PHI_S_K_denom.resize(T, 0.0);

	pst.resize(T);

	for(int s = 0; s < T; ++s)
	{
		nsw[s].resize(S, 0);
		nsk[s].resize(K, 0);

		PHI_S_W_num[s].resize(S, 0.0);
		PHI_S_K_num[s].resize(K, 0.0);

		pst[s].resize(RESOLUTION, 0.0);
	}

	EXRPESSION_PRIOR.resize(T, 0.0);


	alpha_kt.resize(K, 1.0);
	beta_kt.resize(K, 1.0);

	alpha_st.resize(T, 1.0);
	beta_st.resize(T, 1.0);

	// Modify Parameters

	alpha_b = 50.0/B;
	alpha_k = 50.0/K;
	alpha_sk = 50.0/K;

	double k_topic_prob = 1.0/K;
	double s_topic_prob = 1.0/T;

	vector<double> p_s;
	vector<double> p_k;

	p_s.resize(T, s_topic_prob);
	p_k.resize(K, k_topic_prob);

	intr_w_topic.resize(C, -1);
	intr_s_topic.resize(C, -1);

	// Initial Assignment of topic 

	for(int c = 0; c < C; c++)
	{
		int k = sample_from_prob(p_k);

		interaction curr_intr = corpus[c];

		int b = curr_intr.behav;

		nk[k] += 1;

		nkb[k][b] += 1;
		nkbsum[k] += 1;

		int W = curr_intr.c_text.size();


		for(int j = 0; j < W; ++j)
		{
			int w = curr_intr.c_text[j];
			nkw[k][w] += 1;
			nkwsum[k] += 1;
		}

		intr_w_topic[c] = k;

		int s = sample_from_prob(p_s);

		ns[s] += 1;
		nsk[s][k] += 1;
		nsksum[s] += 1;

		W = curr_intr.e_text.size();

		for(int j = 0; j < W; j++)
		{
			int w = curr_intr.e_text[j];
			nsw[s][w] += 1;
			nswsum[s] += 1;
		}

		intr_s_topic[c] = s;

	}


	// Compute PHIs using initial cnts


	for(int k = 0; k < K; ++k)
	{

		TOPIC_PRIOR[k] = log((nk[k] + alpha_k) / (C + (K * alpha_k)));

		for(int b = 0; b < B; ++b)
		{
			PHI_K_B_num[k][b] = log((nkb[k][b] + alpha_b));
		}

		PHI_K_B_denom[k] = log((nkbsum[k] + (B * alpha_b)));

		for(int v = 0; v < V; ++v)
		{
			PHI_K_W_num[k][v] = log((nkw[k][v] + alpha_w));
		}

		PHI_K_W_denom[k] = log((nkwsum[k] + (V * alpha_w)));

	}

	for(int s = 0; s < T; ++s)
	{
		EXRPESSION_PRIOR[s] = log((ns[s] + alpha_sp) / (C - 1 + T * alpha_sp));

		for(int w = 0; w < S; ++w)
		{
			PHI_S_W_num[s][w] = log((nsw[s][w] + alpha_s));
		}

		for(int k = 0; k < K; ++k)
		{
			PHI_S_K_num[s][k] = log((nsk[s][k] + alpha_sk));
		}

		PHI_S_K_denom[s] = log((nsksum[s] + K * alpha_sk));

		PHI_S_W_denom[s] = log((nswsum[s] + (S * alpha_s)));
	}

	cout<<"Initialization Ended"<<endl;
}


void output_data_stats()
{
	cout<<endl;

	cout<<"Size of Corpus: "<<C<<endl;

	cout<<"Vocab Size: "<<V<<endl;

	cout<<"Number of Behaviors: "<<B<<endl;

	cout<<"Expression Vocab: "<<S<<endl;
}

void output_result(string destpath)
{
	// Output Mappings


	// Vocab Mapping

	string c_vocab_map_filename = "c-vocab-mapping.txt";

	ofstream c_vocab_map_file;
	c_vocab_map_file.open((destpath+c_vocab_map_filename).c_str(), ofstream::out);

	for(int j = 0; j < V; j++)
	{
		c_vocab_map_file<<j<<"\t"<<c_wordMapRev[j]<<endl;
	}

	c_vocab_map_file.close();


	string e_vocab_map_filename = "e-vocab-mapping.txt";

	ofstream e_vocab_map_file;
	e_vocab_map_file.open((destpath+e_vocab_map_filename).c_str(), ofstream::out);

	for(int j = 0; j < S; j++)
	{
		e_vocab_map_file<<j<<"\t"<<e_wordMapRev[j]<<endl;
	}

	e_vocab_map_file.close();

	
	// Behav Mapping
	string behavior_map_filename = "behavior-mapping.txt";

	ofstream behav_map_file;
	behav_map_file.open((destpath+behavior_map_filename).c_str(), ofstream::out);

	for(int j = 0; j < B; ++j)
	{
		behav_map_file<<j<<"\t"<<behavMapRev[j]<<endl;
	}

	behav_map_file.close();	


	// TOPIC PRIORS

	string topic_prior_filename = "topic-priors.txt";

	ofstream topic_prior_file;

	topic_prior_file.open((destpath+topic_prior_filename).c_str(), ofstream::out);

	for(int k = 0; k < K; ++k)
	{
		double prob = ((nk[k] + alpha_k) / (C + (K * alpha_k)));

		topic_prior_file<<prob<<endl;
	}

	topic_prior_file.close();


	// Topic Word Distribution

	string c_topic_word_filename = "c-topic-word-distribution.txt";

	ofstream c_topic_word_file;
	c_topic_word_file.open((destpath+c_topic_word_filename).c_str(), ofstream::out);

	for(int k = 0; k < K; ++k)
	{
		for(int v = 0; v < V; ++v)
		{
			double prob = ((nkw[k][v] + alpha_w)/(nkwsum[k] + (V * alpha_w)));

			c_topic_word_file<<prob<<" ";
		}

		c_topic_word_file<<endl;
	}

	c_topic_word_file.close();


	// Topic Behavior Distribution
	string topic_behav_filename = "topic-behavior-distribution.txt";

	ofstream topic_behav_file;
	topic_behav_file.open((destpath+topic_behav_filename).c_str(), ofstream::out);

	for(int k = 0; k < K; ++k)
	{
		for(int b = 0; b < B; ++b)
		{
			double prob = ((nkb[k][b] + alpha_b) / (nkbsum[k] + (B * alpha_b)));
			topic_behav_file<<prob<<" ";
		}

		topic_behav_file<<endl;
	}

	topic_behav_file.close();

	// Topic Time Alpha & Beta

	string c_topic_time_alpha_filename = "c-topic-time-alpha.txt";
	string c_topic_time_beta_filename = "c-topic-time-beta.txt";

	ofstream c_topic_time_alpha_file, c_topic_time_beta_file;
	
	c_topic_time_alpha_file.open(((destpath+c_topic_time_alpha_filename)).c_str(), ofstream::out);
	c_topic_time_beta_file.open(((destpath+c_topic_time_beta_filename)).c_str(), ofstream::out);


	for(int k = 0; k < K; ++k)
	{
		c_topic_time_alpha_file<<alpha_kt[k]<<endl;
		c_topic_time_beta_file<<beta_kt[k]<<endl;
	}

	c_topic_time_alpha_file.close();
	c_topic_time_beta_file.close();


	// Expression Prior

	string expression_prior_filename = "expression-priors.txt";

	ofstream expression_prior_file;

	expression_prior_file.open((destpath+expression_prior_filename).c_str(), ofstream::out);

	for(int s = 0; s < T; ++s)
	{
		double prob = (ns[s] + alpha_sp) / (C + T * alpha_sp);

		expression_prior_file<<prob<<endl;
	}

	expression_prior_file.close();


	// Expression Word Distribution

	string e_topic_word_filename = "e-topic-word-distribution.txt";

	ofstream e_topic_word_file;
	e_topic_word_file.open((destpath+e_topic_word_filename).c_str(), ofstream::out);

	for(int s = 0; s < T; ++s)
	{
		for(int w = 0; w < S; ++w)
		{
			double prob = ((nsw[s][w] + alpha_s) / (nswsum[s] + (S * alpha_s)));

			e_topic_word_file<<prob<<" ";
		}

		e_topic_word_file<<endl;
	}

	e_topic_word_file.close();


	// Topic to Expression Distribution

	string topic_expr_filename = "topic-expression-distribution.txt";

	ofstream topic_expr_file;
	topic_expr_file.open((destpath+topic_expr_filename).c_str(), ofstream::out);

	for(int s = 0; s < T; ++s)
	{
		for(int k = 0; k < K; ++k)
		{
			double prob = ((nsk[s][k] + alpha_sk) / (nsksum[s] + K * alpha_sk));

			topic_expr_file<<prob<<" ";
		}

		topic_expr_file<<endl;
	}

	topic_expr_file.close();


	// Expression Topic Time Alphas and Betas

	string e_topic_time_alpha_filename = "e-topic-time-alpha.txt";
	string e_topic_time_beta_filename = "e-topic-time-beta.txt";

	ofstream e_topic_time_alpha_file, e_topic_time_beta_file;
	
	e_topic_time_alpha_file.open(((destpath+e_topic_time_alpha_filename)).c_str(), ofstream::out);
	e_topic_time_beta_file.open(((destpath+e_topic_time_beta_filename)).c_str(), ofstream::out);


	for(int s = 0; s < T; ++s)
	{
		e_topic_time_alpha_file<<alpha_st[s]<<endl;
		e_topic_time_beta_file<<beta_st[s]<<endl;
	}

	e_topic_time_alpha_file.close();
	e_topic_time_beta_file.close();


	// Top Content Topic Words

	string c_top_topic_words_filename = "top-c-topic-words.txt";
	ofstream c_top_topic_words_file;
	c_top_topic_words_file.open((destpath+c_top_topic_words_filename).c_str(), ofstream::out);

	
	map<int, string>::iterator it;

	if(num_top_words >= V)
	{
		num_top_words = V;
	}

	for (int k = 0; k < K; k++) {
		vector<pair<int, double> > words_probs;
		pair<int, double> word_prob;
		for (int w = 0; w < V; w++) {
			word_prob.first = w;
			word_prob.second = ((nkw[k][w] + alpha_w)/(nkwsum[k] + (V * alpha_w)));
			words_probs.push_back(word_prob);
		}

		// quick sort to sort word-topic probability
		quicksort(words_probs, 0, words_probs.size() - 1);

		c_top_topic_words_file<<"C-Topic "<<k<<":";

		for (int i = 0; i < num_top_words; i++) {
			it = c_wordMapRev.find(words_probs[i].first);
			if (it != c_wordMapRev.end()) {
				c_top_topic_words_file<<(it->second).c_str()<<"("<<words_probs[i].second<<")"<<"\t";
			}
		}

		c_top_topic_words_file<<endl;
	}
	c_top_topic_words_file.close();

	// Top Expression Topic Words

	string e_top_topic_words_filename = "top-e-topic-words.txt";
	ofstream e_top_topic_words_file;
	e_top_topic_words_file.open((destpath+e_top_topic_words_filename).c_str(), ofstream::out);

	if(num_top_words >= S)
	{
		num_top_words = S;
	}

	for (int s = 0; s < T; s++) {
		vector<pair<int, double> > words_probs;
		pair<int, double> word_prob;
		for (int w = 0; w < S; w++) {
			word_prob.first = w;
			word_prob.second = ((nsw[s][w] + alpha_s) / (nswsum[s] + (S * alpha_s)));
			words_probs.push_back(word_prob);
		}

		// quick sort to sort word-topic probability
		quicksort(words_probs, 0, words_probs.size() - 1);

		e_top_topic_words_file<<"E-Topic "<<s<<":";

		for (int i = 0; i < num_top_words; i++) {
			it = e_wordMapRev.find(words_probs[i].first);
			if (it != e_wordMapRev.end()) {
				e_top_topic_words_file<<(it->second).c_str()<<"("<<words_probs[i].second<<")"<<"\t";
			}
		}

		e_top_topic_words_file<<endl;
	}
	e_top_topic_words_file.close();

}


int main(int argc, char const *argv[])
{

	if(argc < 4)
	{
		cout<<"Usage "<<argv[0]<<" K T iter"<<endl;
		exit(1);
	}

	K = atoi(argv[1]);
	T = atoi(argv[2]);
	numIter = atoi(argv[3]);

	string basepath = "../../Dataset/CTP/";
	string destpath = "../../Dataset/CTP/output_"+to_string(K)+"_"+to_string(T)+"_"+to_string(numIter)+"/";

	make_dir(destpath);

	string corpus_path= basepath+"Corpus.txt";

	initialize(corpus_path.c_str());

	output_data_stats();

	run_topic_model();

	output_data_stats();

	cout<<"Writing Result Started"<<endl;

	output_result(destpath);

	cout<<"Writing Results Over"<<endl;

	return 0;
}
