import codecs
import os
import operator
import datetime

user_info_path = '../../Dataset/User-Info/'

src_data_path = "../../Dataset/Parsed-Tweets-New-Source-Only/"

src_scrap_path = "../../Dataset/User-Source-Window-Tweets/"

user_reg_vector_path = '../../Dataset/User-Regularity/user_reg_vector.txt'

def compute_user_vectors(liwc_path):

	liwc_file = codecs.open(liwc_path, 'r', 'utf-8')

	liwc_set = set()

	for row in liwc_file:
		s = row.strip()
		liwc_set.add(s)

	liwc_file.close()

	print(liwc_path, 'Word Count:', len(liwc_set))


	USER_SCREENNAME_REV = {}

	user_info_files = os.listdir(user_info_path)

	for filename in user_info_files:
		user_info_file = codecs.open(user_info_path+filename, 'r', 'utf-8')

		err_cnt = 0
		for row in user_info_file:
			s = row.strip().split('\t')

			try:
				user_id = s[0]
				screen_name = s[1]

				USER_SCREENNAME_REV[screen_name] = user_id

			except:
				err_cnt += 1

		print(user_info_path+filename, err_cnt)

		user_info_file.close()

	print(liwc_path, 'User Screennames Loaded:', len(USER_SCREENNAME_REV))


	DICT_USER_VECTORS = {}


	scrap_files = os.listdir(src_scrap_path)

	for filename in scrap_files:
		scrap_file = codecs.open(src_scrap_path+filename, 'r', 'utf-8')

		print(src_scrap_path+filename)

		for row in scrap_file:
			s = row.strip().split('\t')

			date_tweet = datetime.datetime.strptime(s[0], '%Y-%m-%d %H:%M:%S+00:00')

			screen_name = USER_SCREENNAME_REV[s[2]]

			# if screen_name in words_set:
			# 	continue

			if screen_name not in DICT_USER_VECTORS:
				DICT_USER_VECTORS[screen_name] = [0]*60



			tweet_li = eval(s[3])

			filter_tweet_li = []

			for elem in tweet_li:
				tweet = elem[0]
				curr_date = datetime.datetime.strptime(elem[1],'%d %b %Y')

				if curr_date < date_tweet:

					filter_tweet_li.append((curr_date, tweet))


			for curr_date, tweet in filter_tweet_li:

				diff = (date_tweet - curr_date).days

				words = tweet.split(' ')
				f = 0
				for w in words:
					for l in liwc_set:
						if w.startswith(l):
							try:
								DICT_USER_VECTORS[screen_name][diff-1] = 1
							except:
								print('Error:',diff)
							f = 1
							break

					if f == 1:
						break

		scrap_file.close()

		print(src_scrap_path+filename)


	print(liwc_path, 'User_Vectors Loaded')

	return DICT_USER_VECTORS




def main():

	liwc_path = '../../Dataset/LIWC/tentative.txt'

	USER_REGULARITY = {}

	DICT_USER_VECTORS = compute_user_vectors(liwc_path)

	# Compute Regularity Scores

	for screen_name in DICT_USER_VECTORS:
		vec = DICT_USER_VECTORS[screen_name]

		USER_REGULARITY[screen_name] = []

		cnt = 0

		for elem in vec:
			if elem == 1:
				cnt += 1

		USER_REGULARITY[screen_name].append(int(cnt/6))

	liwc_path = '../../Dataset/LIWC/certain.txt'

	DICT_USER_VECTORS = compute_user_vectors(liwc_path)

	# Compute Regularity Scores

	for screen_name in DICT_USER_VECTORS:
		vec = DICT_USER_VECTORS[screen_name]

		cnt = 0

		for elem in vec:
			if elem == 1:
				cnt += 1

		USER_REGULARITY[screen_name].append(int(cnt/6))


	liwc_path = '../../Dataset/LIWC/negate.txt'

	DICT_USER_VECTORS = compute_user_vectors(liwc_path)

	# Compute Regularity Scores

	for screen_name in DICT_USER_VECTORS:
		vec = DICT_USER_VECTORS[screen_name]

		cnt = 0

		for elem in vec:
			if elem == 1:
				cnt += 1

		USER_REGULARITY[screen_name].append(int(cnt/6))


	liwc_path = '../../Dataset/LIWC/question.txt'

	DICT_USER_VECTORS = compute_user_vectors(liwc_path)

	# Compute Regularity Scores

	for screen_name in DICT_USER_VECTORS:
		vec = DICT_USER_VECTORS[screen_name]

		cnt = 0

		for elem in vec:
			if elem == 1:
				cnt += 1

		USER_REGULARITY[screen_name].append(int(cnt/6))

	print('USER Regularity Scored:', len(USER_REGULARITY))


	user_reg_vector_file = codecs.open(user_reg_vector_path, 'w', 'utf-8')

	for screen_name in USER_REGULARITY:
		print(screen_name+'\t'+str(USER_REGULARITY[screen_name]), file = user_reg_vector_file)


	user_reg_vector_file.close()





if __name__ == "__main__":main()
	