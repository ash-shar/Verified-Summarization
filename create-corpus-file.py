import codecs
import os
import datetime
import time
import numpy as np
from nltk.corpus import stopwords

src_data_path = "../../Dataset/Parsed-Tweets-New-Source-Only/"
rep_data_path = "../../Dataset/Parsed-Tweets-Replies/"

src_pos_tag_path = "../../Dataset/Tweets-POS-Tags-Source/"
rep_pos_tag_path = "../../Dataset/Tweets-POS-Tags-Replies/"


tentative_path = '../../Dataset/LIWC/tentative.txt'
certain_path = '../../Dataset/LIWC/certain.txt'
negate_path = '../../Dataset/LIWC/negate.txt'
question_path = '../../Dataset/LIWC/question.txt'

corpus_path = '../../Dataset/CTP/Corpus.txt'

dataset_names = {'charlie', 'hebdo', 'charliehebdo', 'charlie-hebdo', 'germanwings', 'ottawashooting', 'ottawashootings', 'ottawa', 'sydney', 'lindt', 'sydneysiege', 'ferguson'}

def IsNumeral(s):
	try: 
		int(s)
		return True
	except ValueError:
		return False


def main():

	stop = set(stopwords.words('english'))

	EXPRESSION_WORDS = set()

	# Load Expression words

	tentative_file = codecs.open(tentative_path, 'r', 'utf-8')

	for row in tentative_file:
		EXPRESSION_WORDS.add(row.strip().lower())

	tentative_file.close()


	certain_file = codecs.open(certain_path, 'r', 'utf-8')

	for row in certain_file:
		EXPRESSION_WORDS.add(row.strip().lower())

	certain_file.close()


	negate_file = codecs.open(negate_path, 'r', 'utf-8')

	for row in negate_file:
		EXPRESSION_WORDS.add(row.strip().lower())

	negate_file.close()


	question_file = codecs.open(question_path, 'r', 'utf-8')

	for row in question_file:
		EXPRESSION_WORDS.add(row.strip().lower())

	question_file.close()


	QUESTION_MAP = {}
	EXCLAMATION_MAP = {}

	TIME_MAP = {}

	src_data_files = os.listdir(src_data_path)

	for filename in src_data_files:
		src_data_file = codecs.open(src_data_path+filename, 'r', 'utf-8')

		for row in src_data_file:
			s = row.strip().split('\t')

			if len(s) < 5:
				continue

			tweet_id = s[1]

			date = s[0]

			TIME_MAP[tweet_id] = date

			# if s[1] == '553556844665659392':
			# 	print(row, s)
			# 	exit(0)


			tweet = s[3].strip()

			QUESTION_MAP[tweet_id] = False
			EXCLAMATION_MAP[tweet_id] = False

			if '?' in tweet:
				QUESTION_MAP[tweet_id] = True

			if '!' in tweet:
				EXCLAMATION_MAP[tweet_id] = True

		src_data_file.close()

		print(src_data_path+filename)


	rep_data_files = os.listdir(rep_data_path)

	for filename in rep_data_files:
		rep_data_file = codecs.open(rep_data_path+filename, 'r', 'utf-8')

		for row in rep_data_file:
			s = row.strip().split('\t')
			
			if len(s) < 5:
				continue
			if s[1] in TIME_MAP:
				continue
			TIME_MAP[s[1]] = s[0]

			tweet_id = s[1]

			tweet = s[3].strip()

			QUESTION_MAP[tweet_id] = False
			EXCLAMATION_MAP[tweet_id] = False

			if '?' in tweet:
				QUESTION_MAP[tweet_id] = True

			if '!' in tweet:
				EXCLAMATION_MAP[tweet_id] = True


		rep_data_file.close()

		print(rep_data_path+filename)

	print('TIME_MAP Loaded', len(TIME_MAP))

	# Extract both Content & Expression words for each tweet

	SRC_C_WORDS = {}
	REP_C_WORDS = {}

	SRC_E_WORDS = {}
	REP_E_WORDS = {}

	TIMES = []

	src_pos_tag_files = os.listdir(src_pos_tag_path)

	src_pos_tag_files.sort()

	for filename in src_pos_tag_files:
		src_pos_tag_file = codecs.open(src_pos_tag_path+filename, 'r', 'utf-8')

		for row in src_pos_tag_file:
			words = row.strip().split()

			if len(words) == 1:
				continue

			tweet_id = words[0].strip().split('_')[0]

			SRC_C_WORDS[tweet_id] = set()
			SRC_E_WORDS[tweet_id] = set()

			if tweet_id=='553556844665659392':
				print(tweet_id,TIME_MAP[tweet_id])

			try:
				struct_time = datetime.datetime.strptime(TIME_MAP[tweet_id], '%Y-%m-%d %H:%M:%S+00:00')
			except:
				print('Error:',filename, tweet_id)

			ts = time.mktime(struct_time.timetuple())

			TIMES.append(ts)

			if QUESTION_MAP[tweet_id]:
				SRC_E_WORDS[tweet_id].add('?')

			if EXCLAMATION_MAP[tweet_id]:
				SRC_E_WORDS[tweet_id].add('!')

			words = words[1:]

			for w in words:
				word = w.strip().split('_')[0]
				tag = w.strip().split('_')[-1]

				word = word.lower()

				if word in dataset_names:
					continue

				f = 0
				for e in EXPRESSION_WORDS:
					if word.startswith(e):
						SRC_E_WORDS[tweet_id].add(e)
						f = 1
						break
				

				if f == 1:
					continue

				if word in stop:
					continue

				if IsNumeral(word):
					SRC_C_WORDS[tweet_id].add('NUMERAL')
					continue

				if tag == 'NNP':
					SRC_C_WORDS[tweet_id].add('PROPER_NOUN')
					continue

				if tag.startswith('N') or tag.startswith('V'):
					SRC_C_WORDS[tweet_id].add(word)

		src_pos_tag_file.close()

		print(src_pos_tag_path+filename)


	print('SRC Words Loaded', len(SRC_C_WORDS), len(SRC_E_WORDS))

	# REPLIES

	rep_pos_tag_files = os.listdir(rep_pos_tag_path)

	rep_pos_tag_files.sort()

	for filename in rep_pos_tag_files:
		rep_pos_tag_file = codecs.open(rep_pos_tag_path+filename, 'r', 'utf-8')

		for row in rep_pos_tag_file:
			words = row.strip().split()

			if len(words) == 1:
				continue

			tweet_id = words[0].strip().split('_')[0]

			REP_C_WORDS[tweet_id] = set()
			REP_E_WORDS[tweet_id] = set()

			try:
				struct_time = datetime.datetime.strptime(TIME_MAP[tweet_id], '%Y-%m-%d %H:%M:%S+00:00')
			except:
				print('Error:', filename, tweet_id)
				continue

			ts = time.mktime(struct_time.timetuple())

			TIMES.append(ts)

			if QUESTION_MAP[tweet_id]:
				REP_E_WORDS[tweet_id].add('?')

			if EXCLAMATION_MAP[tweet_id]:
				REP_E_WORDS[tweet_id].add('!')

			words = words[1:]

			for w in words:
				word = w.strip().split('_')[0]
				tag = w.strip().split('_')[-1]

				word = word.lower()

				if word in dataset_names:
					continue

				f = 0
				for e in EXPRESSION_WORDS:
					if word.startswith(e):
						REP_E_WORDS[tweet_id].add(e)
						f = 1
						break

				if f == 1:
					continue

				if word in stop:
					continue

				if IsNumeral(word):
					REP_C_WORDS[tweet_id].add('NUMERAL')
					continue

				if tag == 'NNP':
					REP_C_WORDS[tweet_id].add('PROPER_NOUN')
					continue

				if tag.startswith('N') or tag.startswith('V'):
					REP_C_WORDS[tweet_id].add(word)

		rep_pos_tag_file.close()

		print(rep_pos_tag_path+filename)

	print('REP Words Loaded', len(REP_C_WORDS), len(REP_E_WORDS))

	# Create Time Ranges

	TIMES = np.asarray(TIMES)

	min_time = TIMES.min()
	max_time = TIMES.max()

	TIMES = (TIMES - min_time)/ (max_time - min_time)


	# Create Corpus File

	idx = 0

	corpus_file = codecs.open(corpus_path, 'w', 'utf-8')

	# Tweet Id, Content Text, Expression Text, Behav, Timestamp

	src_pos_tag_files = os.listdir(src_pos_tag_path)

	src_pos_tag_files.sort()

	for filename in src_pos_tag_files:
		src_pos_tag_file = codecs.open(src_pos_tag_path+filename, 'r', 'utf-8')

		for row in src_pos_tag_file:
			words = row.strip().split()

			if len(words) == 1:
				continue

			tweet_id = words[0].strip().split('_')[0]

			c_text = ''

			for w in SRC_C_WORDS[tweet_id]:
				c_text = c_text + ' ' + w

			c_text = c_text.strip()


			e_text = ''

			for w in SRC_E_WORDS[tweet_id]:
				e_text = e_text + ' ' + w

			e_text = e_text.strip()

			behav = 'S'

			ts = TIMES[idx]

			ts = float("{0:.2f}".format(ts))

			if ts == 0.0:
				ts = 0.01

			if ts == 1.0:
				ts = 0.99

			ts_str = "{0:.2f}".format(ts)

			print(tweet_id+'\t'+c_text+'\t'+e_text+'\t'+behav+'\t'+ts_str, file = corpus_file)

			idx += 1

		src_pos_tag_file.close()

		print(src_pos_tag_path+filename)



	rep_pos_tag_files = os.listdir(rep_pos_tag_path)

	rep_pos_tag_files.sort()

	for filename in rep_pos_tag_files:
		rep_pos_tag_file = codecs.open(rep_pos_tag_path+filename, 'r', 'utf-8')

		for row in rep_pos_tag_file:
			words = row.strip().split()

			if len(words) == 1:
				continue

			tweet_id = words[0].strip().split('_')[0]

			try:
				struct_time = datetime.datetime.strptime(TIME_MAP[tweet_id], '%Y-%m-%d %H:%M:%S+00:00')
			except:
				print('Error:', filename, tweet_id)
				continue

			c_text = ''

			for w in REP_C_WORDS[tweet_id]:
				c_text = c_text + ' ' + w

			c_text = c_text.strip()


			e_text = ''

			for w in REP_E_WORDS[tweet_id]:
				e_text = e_text + ' ' + w

			e_text = e_text.strip()

			behav = 'R'

			ts = TIMES[idx]

			ts = float("{0:.2f}".format(ts))

			if ts == 0.0:
				ts = 0.01

			if ts == 1.0:
				ts = 0.99

			ts_str = "{0:.2f}".format(ts)

			print(tweet_id+'\t'+c_text+'\t'+e_text+'\t'+behav+'\t'+ts_str, file = corpus_file)

			idx += 1
		
		rep_pos_tag_file.close()

		print(rep_pos_tag_path+filename)

	corpus_file.close()





if __name__ == "__main__":main()