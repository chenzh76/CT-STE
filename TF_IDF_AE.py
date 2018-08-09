import json
import numpy as np

#Compute the text embeddings by TF-IDF weighted sum of word embeddings.
def sen2vec(words_in_lines):
  doc_line_embeddings = np.zeros((1, cols), dtype=float)
  w_total = 0.0
  for word in words_in_lines:
    idf = idf_dict[word]
    w_total += pow(idf, 1)
    doc_line_embeddings = doc_line_embeddings + pow(idf, 1) * embeddings[word]
  doc_line_embeddings = doc_line_embeddings / w_total
  return doc_line_embeddings


file = open('data/web_snippets.json')
lines = file.readlines()
doc_num = len(lines)

old_embeddings_file = open('data/used_embeddings.txt')
old_embeddings_lines = old_embeddings_file.readlines()
embeddings_file = open('data/topic_word_embeddings58.json')
embeddings_lines = embeddings_file.readlines()
test_item = json.loads(embeddings_lines[0])
rows, cols = len(embeddings_lines[10:]), len(test_item['embedding'].split(','))
embeddings = np.zeros((rows, cols), dtype=float)
embed_dict = dict()
words = []

old_words = []
for i in xrange(len(old_embeddings_lines)):
  tmp_list = old_embeddings_lines[i].split(' ')
  embed_dict[tmp_list[0]] = i
  old_words.append(tmp_list[0])

for i in xrange(rows):
  item = json.loads(embeddings_lines[i + 10])
  tmp_list = item['embedding'].split(',')
  words.append(item['word_or_topic'])
  index = embed_dict[item['word_or_topic']]
  for j in xrange(cols):
    embeddings[index, j] = tmp_list[j]
print embeddings

#Compute the IDF values for each word.
idf_dict = dict()
for i in xrange(len(lines)):
  item = json.loads(lines[i])
  for word in item['snippet_in_ids']:
    if word not in idf_dict:
      idf_dict[word] = []
    if i not in idf_dict[word]:
      idf_dict[word].append(i)

for key in idf_dict.keys():
  idf_dict[key] = np.log(doc_num * 1.0 / len(idf_dict[key]))

doc_embeddings = np.zeros((doc_num, cols), dtype=float)
for i in xrange(len(lines)):
  item = json.loads(lines[i])
  doc_embeddings[i] = sen2vec(item['snippet_in_ids'])

out_file = open('doc_embeddings_2.json', 'w')
for i in xrange(doc_num):
  item = json.loads(lines[i])
  new_item = dict()
  new_item['id'] = i
  embedding_str = '' + str(doc_embeddings[i, 0])
  for j in xrange(1, cols):
    embedding_str += ',' + str(doc_embeddings[i, j])
  new_item['embedding'] = embedding_str
  new_item['snippet'] = item['snippet']
  new_item['label'] = item['label']
  new_line = json.dumps(new_item) + '\n'
  out_file.write(new_line)
out_file.close()