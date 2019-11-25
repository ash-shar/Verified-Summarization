import codecs
import os
import math
import numpy as np 
import scipy.special
import operator


K = 30
T = 10
numItr = 1000

basepath = "../../Dataset/CTP/output_"+str(K)+"_"+str(T)+"_"+str(numItr)+"_old/"

PHI_K_W_PATH = basepath + 'c-topic-word-distribution.txt'
PHI_K_B_PATH = basepath + 'topic-behavior-distribution.txt'
PHI_S_W_PATH = basepath + 'e-topic-word-distribution.txt'
PHI_S_K_PATH = basepath + 'topic-expression-distribution.txt'
EXPRESSION_PRIOR_PATH = basepath + 'expression-priors.txt'
TOPIC_PRIOR_PATH = basepath + 'topic-priors.txt'
TOPIC_ALPHA_PATH = basepath + 'c-topic-time-alpha.txt'
TOPIC_BETA_PATH = basepath + 'c-topic-time-beta.txt'

C_VOCAB_MAP_PATH = basepath + 'c-vocab-mapping.txt'
E_VOCAB_MAP_PATH = basepath + 'e-vocab-mapping.txt'
BEHAV_MAP_PATH = basepath + 'behavior-mapping.txt'


CORPUS_PATH = '../../Dataset/CTP/' + 'Corpus.txt'

POSTERIOR_PATH = basepath + 'tweet-posteriors.txt'

def load_PHI_K_W():
	PHI_K_W = []

	file = codecs.open(PHI_K_W_PATH, 'r', 'utf-8')

	for row in file:
		s = row.strip().split(' ')
		curr_li = []
		for elem in s:
			prob = float(elem)
			curr_li.append(prob)

		PHI_K_W.append(curr_li)

	file.close()

	return PHI_K_W


def load_PHI_K_B():
	PHI_K_B = []

	file = codecs.open(PHI_K_B_PATH, 'r', 'utf-8')

	for row in file:
		s = row.strip().split(' ')
		curr_li = []
		for elem in s:
			prob = float(elem)
			curr_li.append(prob)

		PHI_K_B.append(curr_li)

	file.close()

	return PHI_K_B

def load_PHI_S_W():
	PHI_S_W = []

	file = codecs.open(PHI_S_W_PATH, 'r', 'utf-8')

	for row in file:
		s = row.strip().split(' ')
		curr_li = []
		for elem in s:
			prob = float(elem)
			curr_li.append(prob)

		PHI_S_W.append(curr_li)

	file.close()

	return PHI_S_W


def load_PHI_S_K():
	PHI_S_K = []

	file = codecs.open(PHI_S_K_PATH, 'r', 'utf-8')

	for row in file:
		s = row.strip().split(' ')
		curr_li = []
		for elem in s:
			prob = float(elem)
			curr_li.append(prob)

		PHI_S_K.append(curr_li)

	file.close()

	return PHI_S_K


def load_ALPHA_BETA_K():
	ALPHA_K = []

	alpha_file = codecs.open(TOPIC_ALPHA_PATH, 'r', 'utf-8')

	for row in alpha_file:
		ALPHA_K.append(float(row.strip()))

	alpha_file.close()


	BETA_K = []

	beta_file = codecs.open(TOPIC_BETA_PATH, 'r', 'utf-8')

	for row in beta_file:
		BETA_K.append(float(row.strip()))

	beta_file.close()

	return ALPHA_K, BETA_K


def load_TOPIC_PRIOR():
	TOPIC_PRIOR = []

	topic_prior_file = codecs.open(TOPIC_PRIOR_PATH, 'r', 'utf-8')

	for row in topic_prior_file:
		TOPIC_PRIOR.append(float(row.strip()))

	topic_prior_file.close()

	return TOPIC_PRIOR


def load_EXPRESSION_PRIOR():
	EXPRESSION_PRIOR = []

	expression_prior_file = codecs.open(EXPRESSION_PRIOR_PATH, 'r', 'utf-8')

	for row in expression_prior_file:
		EXPRESSION_PRIOR.append(float(row.strip()))

	expression_prior_file.close()

	return EXPRESSION_PRIOR


def load_C_VOCAB_MAP():
	C_VOCAB_MAP = {}

	file = codecs.open(C_VOCAB_MAP_PATH, 'r', 'utf-8')
	idx = 0
	for row in file:
		# idx+=1
		# print(idx)
		try:
			s = row.strip().split('\t')
			C_VOCAB_MAP[s[1]] = int(s[0])
		except:
			print("Error: ",row.strip())

	file.close()

	return C_VOCAB_MAP


def load_E_VOCAB_MAP():
	E_VOCAB_MAP = {}

	file = codecs.open(E_VOCAB_MAP_PATH, 'r', 'utf-8')
	idx = 0
	for row in file:
		# idx+=1
		# print(idx)
		try:
			s = row.strip().split('\t')
			E_VOCAB_MAP[s[1]] = int(s[0])
		except:
			print("Error: ",row.strip())

	file.close()

	return E_VOCAB_MAP


def load_BEHAV_MAP():
	BEHAV_MAP = {}

	file = codecs.open(BEHAV_MAP_PATH, 'r', 'utf-8')

	for row in file:
		s = row.strip().split('\t')
		BEHAV_MAP[s[1]] = int(s[0])

	file.close()

	return BEHAV_MAP


def compute_time_prob(alpha, beta, t):
	prob = (1.0*(math.pow(t, alpha - 1))*(math.pow(1 - t, beta - 1)))/(scipy.special.beta(alpha, beta))

	return prob


def compute_TWEET_POSTERIORS(BEHAV_MAP, C_VOCAB_MAP, E_VOCAB_MAP, ALPHA_K, BETA_K, TOPIC_PRIOR, EXPRESSION_PRIOR, PHI_S_K, PHI_S_W, PHI_K_W, PHI_K_B):

	DICT_TWEET_POSTERIORS = {}

	corpus_file = codecs.open(CORPUS_PATH, 'r', 'utf-8')

	idx = 0

	for row in corpus_file:
		# Tweet Id, Content Text, Expression Text, Behav, Timestamp
		li = row.strip().split('\t')

		tweet_id = li[0]

		c_text = li[1]
		e_text = li[2]
		b = li[3]
		ts = float(li[4].strip())

		b = BEHAV_MAP[b]

		c_text_li = c_text.strip().split()
		e_text_li = e_text.strip().split()

		curr_posterior_k = []
		curr_posterior_s = []


		for k in range(K):
			prob_k = 1.0

			prob_k *= TOPIC_PRIOR[k]
			prob_k *= compute_time_prob(ALPHA_K[k], BETA_K[k], ts)
			prob_k *= PHI_K_B[k][b]

			for word in c_text_li:
				if word == '':
					continue
				w = C_VOCAB_MAP[word]

				prob_k = prob_k * PHI_K_W[k][w]

			curr_posterior_k.append(prob_k)

		for s in range(T):
			prob_s = 1.0

			prob_s *= EXPRESSION_PRIOR[s]

			for word in e_text_li:
				if word == '':
					continue
				try:
					w = E_VOCAB_MAP[word]
					prob_s = prob_s * PHI_S_W[s][w]
				except:
					pass
					# print('Error:', word)

			curr_posterior_s.append(prob_s)


		sum_li = np.sum(curr_posterior_k)

		if sum_li == 0.0:
			sum_li = np.sum([1.0])
		
		curr_posterior_k = curr_posterior_k/sum_li
		curr_posterior_k = curr_posterior_k.tolist()


		sum_li = np.sum(curr_posterior_s)

		if sum_li == 0.0:
			sum_li = np.sum([1.0])
		
		curr_posterior_s = curr_posterior_s/sum_li
		curr_posterior_s = curr_posterior_s.tolist()

		curr_posterior = curr_posterior_k + curr_posterior_s

		DICT_TWEET_POSTERIORS[tweet_id] = li[3] + '\t' + str(curr_posterior)

		idx+=1

		if idx%1000==0:
			print(idx)

	corpus_file.close()

	return DICT_TWEET_POSTERIORS


def main():
	# Load Everything

	BEHAV_MAP = load_BEHAV_MAP()
	C_VOCAB_MAP = load_C_VOCAB_MAP()
	E_VOCAB_MAP = load_E_VOCAB_MAP()

	ALPHA_K, BETA_K = load_ALPHA_BETA_K()

	TOPIC_PRIOR = load_TOPIC_PRIOR()
	EXPRESSION_PRIOR = load_EXPRESSION_PRIOR()

	PHI_S_K = load_PHI_S_K()
	PHI_S_W = load_PHI_S_W()
	PHI_K_W = load_PHI_K_W()
	PHI_K_B = load_PHI_K_B()


	print('Loading Done')

	DICT_TWEET_POSTERIORS = compute_TWEET_POSTERIORS(BEHAV_MAP, C_VOCAB_MAP, E_VOCAB_MAP, ALPHA_K, BETA_K, TOPIC_PRIOR, EXPRESSION_PRIOR, PHI_S_K, PHI_S_W, PHI_K_W, PHI_K_B)


	print("Tweet Posterior Dict: ", len(DICT_TWEET_POSTERIORS))

	posterior_file = codecs.open(POSTERIOR_PATH, 'w', 'utf-8')

	for tweet_id in DICT_TWEET_POSTERIORS:
		print(str(tweet_id)+'\t'+str(DICT_TWEET_POSTERIORS[tweet_id]), file = posterior_file)

	posterior_file.close()


if __name__ == "__main__":main()