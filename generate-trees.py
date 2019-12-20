import codecs
import os
import json
import time
from datetime import datetime
from datetime import timezone
from collections import defaultdict
import copy
import pickle


datapath = "../../Dataset/pheme-rnr-dataset/"

feature_path = "../../Dataset/Features/tweet-posteriors.txt"

output_path = "../../Dataset/Parsed-Trees/"


import numpy
import operator


if not os.path.exists(output_path):
	os.makedirs(output_path)


CUTOFF = 20

class Graph: 
  
	# Constructor 
	def __init__(self, init_dict): 
  
		# default dictionary to store graph 
		self.graph = defaultdict(list)
		self.DICT_TREE = init_dict

		self.visited = {}  #[False] * (len(self.graph))
  
	# function to add an edge to graph 
	def addEdge(self, u, v): 
		self.graph[u].append(v) 
		self.visited[u] = False
		self.visited[v] = False
  
	# A function used by DFS 
	def DFSUtil(self, v, level): 
  
		# Mark the current node as visited  
		# and print it 
		self.visited[v] = True

		if level > CUTOFF:
			return

		# print(v, end = ' ')

		# Recur for all the vertices  
		# adjacent to this vertex 
		for i in self.graph[v]: 
			if self.visited[i] == False: 
				self.DFSUtil(i, level + 1)

				if 'cl' in self.DICT_TREE[i]:
					for elem in self.DICT_TREE[i]['cl']:
						if self.DICT_TREE[elem] == {}:
							continue
						self.DICT_TREE[i]['c'].append(self.DICT_TREE[elem])
				
					del self.DICT_TREE[i]['cl']

		if 'cl' in self.DICT_TREE[v]:
			for elem in self.DICT_TREE[v]['cl']:
				if self.DICT_TREE[elem] == {}:
					continue

				self.DICT_TREE[v]['c'].append(self.DICT_TREE[elem])	
		
			del self.DICT_TREE[v]['cl']

  
	# The function to do DFS traversal. It uses 
	# recursive DFSUtil() 
	def DFS(self, v): 
  
		# Call the recursive helper function  
		# to print DFS traversal 
		self.DFSUtil(v, 1)



def main():

	FEATURES = {}

	feature_file = codecs.open(feature_path, 'r', 'utf-8')

	for row in feature_file:
		s = row.strip().split('\t')

		FEATURES[s[0]] = eval(s[-1].strip())

	feature_file.close()

	print('Features Loaded:', len(FEATURES))




	datasets = os.listdir(datapath)

	for dataset in datasets:

		if dataset == '.DS_Store':
			continue

		print(dataset, 'started')

		err_cnt = 0

		dict_r = {}
		dict_label = {}
		dict_graph = {}

		LABELS = {}

		reactions = set()

		rumour_path = datapath+dataset+"/rumours/"
		non_rumour_path = datapath+dataset+"/non-rumours/"


		rumour_tweets = os.listdir(rumour_path)

		for tweet_id in rumour_tweets:
			if tweet_id == '.DS_Store':
				continue

			source_path = rumour_path+tweet_id+'/source-tweet/'+tweet_id+'.json'

			source_tweet_file = codecs.open(source_path, 'r', 'utf-8')

			tweet = source_tweet_file.read()
			d = json.loads(tweet)

			date_created = datetime.strptime(d["created_at"], '%a %b %d %H:%M:%S %z %Y')#.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S'))

			source_tweet_file.close()

			ts = time.mktime(date_created.timetuple())


			reactions_path = rumour_path+tweet_id+'/reactions/'

			reaction_tweets = os.listdir(reactions_path)

			if tweet_id not in dict_r:
				dict_r[tweet_id] = [(tweet_id, ts)]
				dict_graph[tweet_id] = []

			for r_tweet_id in reaction_tweets:

				if r_tweet_id == '.DS_Store':
					continue

				reaction_tweet_file = codecs.open(reactions_path+r_tweet_id, 'r', 'utf-8')

				r_tweet = reaction_tweet_file.read()
				d = json.loads(r_tweet)


				date_created = datetime.strptime(d["created_at"], '%a %b %d %H:%M:%S %z %Y')#.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S'))

				source_id = str(d['in_reply_to_status_id'])

				reaction_tweet_file.close()

				ts = time.mktime(date_created.timetuple())

				if source_id not in dict_graph:
					dict_graph[source_id] = []

				dict_r[tweet_id].append((r_tweet_id.split('.')[0], ts))

				dict_graph[source_id].append(r_tweet_id.split('.')[0])

				if r_tweet_id.split('.')[0] not in dict_graph:
					dict_graph[r_tweet_id.split('.')[0]] = []

				reactions.add(r_tweet_id)

				LABELS[r_tweet_id.split('.')[0]] = [0, 1]

			dict_label[tweet_id] = 'r'
			LABELS[tweet_id] = [0, 1]

			# print(dict_r[tweet_id])
			# exit(-1)

		# Non_Rumour

		non_rumour_tweets = os.listdir(non_rumour_path)

		for tweet_id in non_rumour_tweets:
			if tweet_id == '.DS_Store':
				continue
			source_path = non_rumour_path+tweet_id+'/source-tweet/'+tweet_id+'.json'

			source_tweet_file = codecs.open(source_path, 'r', 'utf-8')

			tweet = source_tweet_file.read()
			d = json.loads(tweet)

			date_created = datetime.strptime(d["created_at"], '%a %b %d %H:%M:%S %z %Y')#.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S'))

			source_tweet_file.close()

			ts = time.mktime(date_created.timetuple())		

			reactions_path = non_rumour_path+tweet_id+'/reactions/'

			reaction_tweets = os.listdir(reactions_path)

			if tweet_id not in dict_r:
				dict_r[tweet_id] = [(tweet_id, ts)]
				dict_graph[tweet_id] = []

			for r_tweet_id in reaction_tweets:

				if r_tweet_id == '.DS_Store':
					continue

				reaction_tweet_file = codecs.open(reactions_path+r_tweet_id, 'r', 'utf-8')

				r_tweet = reaction_tweet_file.read()
				d = json.loads(r_tweet)

				date_created = datetime.strptime(d["created_at"], '%a %b %d %H:%M:%S %z %Y')#.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S'))

				source_id = str(d['in_reply_to_status_id'])

				reaction_tweet_file.close()

				ts = time.mktime(date_created.timetuple())

				if source_id not in dict_graph:
					dict_graph[source_id] = []

				dict_r[tweet_id].append((r_tweet_id.split('.')[0], ts))

				dict_graph[source_id].append(r_tweet_id.split('.')[0])

				if r_tweet_id.split('.')[0] not in dict_graph:
					dict_graph[r_tweet_id.split('.')[0]] = []

				reactions.add(r_tweet_id)

				LABELS[r_tweet_id.split('.')[0]] = [1, 0]

			dict_label[tweet_id] = 'nr'
			LABELS[tweet_id] = [1, 0]

		curr_tree = {}

		IDs = set()

		for tweet_id in dict_graph:

			# if tweet_id not in dict_r:
			# 	continue

			
			try:
				curr_tree[tweet_id] = {}
				curr_tree[tweet_id]['f'] = FEATURES[tweet_id]
				curr_tree[tweet_id]['l'] = LABELS[tweet_id]
				curr_tree[tweet_id]['cl'] = copy.copy(dict_graph[tweet_id])

				curr_tree[tweet_id]['c'] = []

				IDs.add(tweet_id)
			except:
				# print('Error:', tweet_id)
				err_cnt += 1
	
		g = Graph(curr_tree)

		for tweet_id in IDs:
			curr_li = dict_graph[tweet_id]

			for curr_elem in curr_li:
				if curr_elem in IDs:
					g.addEdge(tweet_id, curr_elem)
		

		output_file = codecs.open(output_path+dataset+'.txt', 'w', 'utf-8')

		for tweet_id in dict_label:
			if tweet_id not in dict_r or tweet_id not in IDs:
				continue

			# print(tweet_id)
			
			g.DFS(tweet_id)
			# exit(-1)

			print(tweet_id + '\t' + str(g.DICT_TREE[tweet_id]), file = output_file)

		output_file.close()

		print(dataset, 'done', err_cnt)


if __name__ == "__main__":main()