import json
import numpy as np

def PCA(data_mat):
  rows, cols = data_mat.shape
  for i in xrange(rows):
    for j in xrange(cols):
      data_mat[i] = data_mat[i] - data_mat[i].mean()
  corr_mat = np.dot(data_mat, data_mat.transpose()) / cols

  evals, evects = np.linalg.eig(corr_mat)
  vals_vects = dict(zip(evals, evects.transpose()))
  keys = vals_vects.keys()
  keys.sort()

  Y = np.array([vals_vects[k].real for k in keys[:10]]).transpose()
  print Y.shape
  print Y
  return Y

def cosine_similarity(doc1_embed, doc2_embed):
  norm1 = np.sqrt(np.sum(np.square(doc1_embed)))
  norm2 = np.sqrt(np.sum(np.square(doc2_embed)))
  normalized_embeddings1 = doc1_embed / norm1
  normalized_embeddings2 = doc2_embed / norm2

  similarity = np.dot(normalized_embeddings1, normalized_embeddings2.transpose())
  return similarity

def tf_idf_filter_2vec(words_in_lines):
  filted_list = []
  tf_idf_list = []
  tf = dict()
  for word in words_in_lines:
    if word not in tf:
      tf[word] = words_in_lines.count(word) * 1.0 / len(words_in_lines)
    idf = idf_dict[word]
    tf_idf_list.append(tf[word] * idf)

  filted_list = []
  tf_idf_list.sort()
  median = tf_idf_list[int(len(tf_idf_list) / 1.5)]
  doc_line_embeddings = np.zeros((1, cols), dtype=float)
  w_total = 0.0
  for word in words_in_lines:
    tf_v = tf[word]
    idf = idf_dict[word]
    w_total += pow(tf_v * idf, 1)
    if tf_v * idf >= median:
      out_file2.write(words[word] + ' ')
      doc_line_embeddings = doc_line_embeddings + pow(tf_v * idf, 1) * embeddings[word]
  out_file2.write('\n')
  doc_line_embeddings = doc_line_embeddings / w_total
  return doc_line_embeddings


def sen2vec(words_in_lines):
  doc_line_embeddings = np.zeros((1, cols), dtype=float)
  w_total = 0.0
  for word in words_in_lines:
    # tf = words_in_lines.count(word) * 1.0 / len(words_in_lines)
    idf = idf_dict[word]
    w_total += pow(idf, 1)
    # print tf * idf
    doc_line_embeddings = doc_line_embeddings + pow(idf, 1) * embeddings[word]
  doc_line_embeddings = doc_line_embeddings / w_total

  # print 'min:', min(tf_idf_list), 'max:', max(tf_idf_list)
  # doc_line_embeddings2 = np.zeros((1, cols), dtype=float)
  # w_total2 = 0.0
  # for word in words_in_lines:
  #   tf = words_in_lines.count(word) * 1.0 / len(words_in_lines)
  #   idf = idf_dict[word]
  #   cos_weight = cosine_similarity(doc_line_embeddings, embeddings[word])
  #   w_total2 += tf * idf * pow(cos_weight, 1)
  #   doc_line_embeddings2 = doc_line_embeddings2 + pow(cos_weight, 1) * embeddings[word]
  #   w_total2 += pow(cos_weight, 1)
  # doc_line_embeddings2 = doc_line_embeddings2 / w_total2
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
out_file2 = open('filted_snippets', 'w')
for i in xrange(len(lines)):
  item = json.loads(lines[i])
  # doc_embeddings[i] = tf_idf_filter_2vec(item['snippet_in_ids'])
  doc_embeddings[i] = sen2vec(item['snippet_in_ids'])

# pca_label = PCA(doc_embeddings)

out_file = open('doc_embeddings_3.json', 'w')
# out_file2 = open('pca_label.txt', 'w')

# for i in xrange(doc_num):
#   label_str = str(pca_label[i, 0])
#   for j in xrange(1, 10):
#     label_str += ',' + str(pca_label[i, j])
#   out_file2.write(label_str + '\n')
# out_file2.close()


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

out_file2 = open('used_embeddings2.txt', 'w')
embed_rows, embed_cols = embeddings.shape
for i in xrange(embed_rows):
  embed_str = old_words[i]
  for j in xrange(embed_cols):
    embed_str += ' ' + str(embeddings[i, j])
  out_file2.write(embed_str + '\n')
out_file2.close()