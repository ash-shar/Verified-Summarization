import codecs
import os
import json
import time
from datetime import datetime
from datetime import timezone


datapath = "F:\\Acads\\MTP\\Dataset\\pheme-rnr-dataset\\"

output_path = "F:\\Acads\\MTP\\Dataset\\Parsed-Trees-New\\"

import numpy
import operator

def generate_tree(li, label):

	li.sort(key = operator.itemgetter(1))

	times = []

	for i in range(len(li)):
		times.append(li[i][1])

	times = numpy.asarray(times)

	min_time = times.min()
	max_time = times.max()

	times = (times - min_time)/ (max_time - min_time)


	prob = float("{0:.2f}".format(times[0]))

	if label == 'r':
		o = numpy.random.choice(numpy.arange(0, 2), p=[1-prob, prob])
	elif label == 'nr':
		o = numpy.random.choice(numpy.arange(0, 2), p=[prob, 1-prob])

	s = ' ('+str(o)+' '+li[0][0]+')'

	for i in range(1, len(li)):
		ts = times[i]
		prob = float("{0:.2f}".format(ts))

		if label == 'r':
			o = numpy.random.choice(numpy.arange(0, 2), p=[1-prob, prob])
		elif label == 'nr':
			o = numpy.random.choice(numpy.arange(0, 2), p=[prob, 1-prob])

		if i == len(li)-1:
			if label == 'r':
				curr_s = ' ('+str(1)+' '+li[i][0]+')'
			elif label == 'nr':
				curr_s = ' ('+str(0)+' '+li[i][0]+')'

		else:
			curr_s = ' ('+str(o)+' '+li[i][0]+')'

		s = ' ('+str(o)+s+curr_s+')'

	return s.strip()

def main():
	datasets = os.listdir(datapath)

	for dataset in datasets:

		print(dataset, 'started')

		dict_r = {}
		dict_label = {}
		dict_graph = {}

		reactions = set()

		rumour_path = datapath+dataset+"/rumours/"
		non_rumour_path = datapath+dataset+"/non-rumours/"


		rumour_tweets = os.listdir(rumour_path)

		for tweet_id in rumour_tweets:

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

				reaction_tweet_file = codecs.open(reactions_path+r_tweet_id, 'r', 'utf-8')

				r_tweet = reaction_tweet_file.read()
				d = json.loads(r_tweet)

				date_created = datetime.strptime(d["created_at"], '%a %b %d %H:%M:%S %z %Y')#.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S'))

				reaction_tweet_file.close()

				ts = time.mktime(date_created.timetuple())

				dict_r[tweet_id].append((r_tweet_id.split('.')[0], ts))

				dict_graph[tweet_id].append(r_tweet_id.split('.')[0])

				dict_graph[r_tweet_id.split('.')[0]] = []

				reactions.add(r_tweet_id)

			dict_label[tweet_id] = 'r'

			# print(dict_r[tweet_id])
			# exit(-1)

		# Non_Rumour

		non_rumour_tweets = os.listdir(non_rumour_path)

		for tweet_id in non_rumour_tweets:
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

				reaction_tweet_file = codecs.open(reactions_path+r_tweet_id, 'r', 'utf-8')

				r_tweet = reaction_tweet_file.read()
				d = json.loads(r_tweet)

				date_created = datetime.strptime(d["created_at"], '%a %b %d %H:%M:%S %z %Y')#.replace(tzinfo=timezone.utc).astimezone(tz=None).strftime('%Y-%m-%d %H:%M:%S'))

				reaction_tweet_file.close()

				ts = time.mktime(date_created.timetuple())

				dict_r[tweet_id].append((r_tweet_id.split('.')[0], ts))

				dict_graph[tweet_id].append(r_tweet_id.split('.')[0])

				dict_graph[r_tweet_id.split('.')[0]] = []

				reactions.add(r_tweet_id)

			dict_label[tweet_id] = 'nr'


		output_file = codecs.open(output_path+dataset+'.txt', 'w', 'utf-8')

		for tweet_id in dict_label:
			if tweet_id not in dict_r:
				continue

			if len(dict_r[tweet_id]) <= 1:
				continue

			s = generate_tree(dict_r[tweet_id], dict_label[tweet_id])

			print(s.strip(), file = output_file)


		# for tweet_id in dict_label:

		# 	if tweet_id not in dict_r:
		# 		continue

		# 	if len(dict_r[tweet_id]) <= 0:
		# 		continue

		# 	o = []
		# 	num = 0
		# 	s = ''

		# 	visited, stack = [], [tweet_id]

		# 	li_all = set()

		# 	while stack:
		# 		vertex = stack.pop()

		# 		li = [x for x in dict_graph[vertex] if x not in visited]

		# 		# print(o)

		# 		if len(li) == 0 and vertex not in li_all:
		# 			s = s+' '+'('+vertex+' '+vertex+')'
		# 			li_all.add(vertex)

		# 			if len(o) != 0:
		# 				o[-1] -= 1

		# 				while o[-1] <= 0:
		# 					o.pop()
		# 					s = s+' '+')'
		# 					num -= 1
		# 					if len(o) != 0:
		# 						o[-1] -= 1
		# 					else:
		# 						break

		# 		elif len(li) == 1 and vertex not in li_all and li[0] not in li_all:

		# 			li_nest = [x for x in dict_graph[li[0]] if x not in visited and x!=vertex]
					
		# 			if len(li_nest) == 0:
		# 				s = s+' '+'('+vertex+' '+li[0]+')'
		# 				li_all.add(li[0])

		# 				if len(o) != 0:
		# 					o[-1] -= 1

		# 					while o[-1] <= 0:
		# 						o.pop()
		# 						s = s+' '+')'
		# 						num -= 1
		# 						if len(o) != 0:
		# 							o[-1] -= 1
		# 						else:
		# 							break

		# 			else:
		# 				s = s+'  '+'('+vertex+''

		# 				num += 1

		# 				# if len(o) != 0:
		# 				#     o[-1] -= 1

		# 				o.append(len(li))

		# 			li_all.add(vertex)

		# 		elif len(li) >= 1 and vertex not in li_all:
		# 			s = s+' '+'('+vertex+''
		# 			li_all.add(vertex)
		# 			o.append(len(li))

		# 			num += 1

		# 		if vertex not in visited:
		# 			visited.append(vertex)
		# 			stack.extend([x for x in dict_graph[vertex] if x not in visited])

		# 	for i in range(num):
		# 		s = s+' '+')'

		# 	print(s.strip()+'\t'+dict_label[tweet_id], file = output_file)


		output_file.close()

		print(dataset, 'done')



if __name__ == "__main__":main()