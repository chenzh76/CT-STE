import numpy as np
import random
import json

class Document:
  def __init__(self, sentences_filename, word_to_index, index_to_word):
    self.sentences = []
    self.word_to_index = word_to_index
    self.index_to_word = index_to_word
    self.word_to_count = dict()
    file = open(sentences_filename)
    lines = file.readlines()

    for i in xrange(len(lines)):
      item = json.loads(lines[i])
      sentence = item['snippet_in_ids']
      # print sentence
      self.sentences.append(sentence)
      for word_id in item['snippet_in_ids']:
        if self.index_to_word[word_id] not in self.word_to_count:
          self.word_to_count[self.index_to_word[word_id]] = 0
        self.word_to_count[self.index_to_word[word_id]] += 1
    file.close()

class LDA_Model:
  def __init__(self, param_dict, document):
    self.alpha = param_dict['alpha']
    self.beta = param_dict['beta']
    self.iterations_num = param_dict['iterations_num']
    self.topics_num = param_dict['topics_num']
    self.save_step = param_dict['save_step']
    self.begin_save_iters = param_dict['begin_save_iters']

    #vocabulary size, topic number, document number
    self.document = document
    self.sentences_num = len(document.sentences)
    self.vocabulary_size = len(document.index_to_word)
    self.nmk = np.zeros((self.sentences_num, self.topics_num), dtype=float)
    self.nkt = np.zeros((self.topics_num, self.vocabulary_size), dtype=float)
    self.nmk_sum = np.zeros((self.sentences_num), dtype=float)
    self.nkt_sum = np.zeros((self.topics_num), dtype=float)
    self.phi = np.zeros((self.topics_num, self.vocabulary_size), dtype=float)
    self.theta = np.zeros((self.sentences_num, self.topics_num), dtype=float)

    self.docs = []
    for i in xrange(self.sentences_num):
      self.docs.append(document.sentences[i])

    self.z = []
    for m in xrange(self.sentences_num):
      self.z.append([])
      for n in xrange(len(self.docs[m])):
        topic = int(random.random() * self.topics_num)
        self.z[m].append(topic)
        self.nmk[m][topic] += 1
        self.nkt[topic][self.docs[m][n]] += 1
        self.nkt_sum[topic] += 1
      self.nmk_sum[m] = len(self.docs[m])

  def inference_Model(self):
    if self.iterations_num < self.save_step + self.begin_save_iters:
      print 'iterations num error.'

    for i in xrange(self.iterations_num):
      print 'Iteration', i
      self.update_estimated_parameters()

      if i % 100 == 0 and i > 0:
        self.print_topic_words()

      if i % 200 == 0 and i > 0:
        self.output_theta(i / 200)

      for i in xrange(self.sentences_num):
        for j in xrange(len(self.docs[i])):
          new_topic = self.sample_topic_Z(i, j)
          self.z[i][j] = new_topic

  def update_estimated_parameters(self):
    for i in xrange(self.topics_num):
      for j in xrange(self.vocabulary_size):
        self.phi[i, j] = (self.nkt[i][j] + self.beta) / (self.nkt_sum[i] + self.vocabulary_size * self.beta)

    for i in xrange(self.sentences_num):
      for j in xrange(self.topics_num):
        self.theta[i, j] = (self.nmk[i][j] + self.alpha) / (self.nmk_sum[i] + self.topics_num * self.alpha)

  def sample_topic_Z(self, m, n):
    old_topic = self.z[m][n]
    self.nmk[m][old_topic] -= 1
    self.nkt[old_topic][self.docs[m][n]] -= 1
    self.nmk_sum[m] -= 1
    self.nkt_sum[old_topic] -= 1

    p = [0 for i in xrange(self.topics_num)]
    for k in xrange(self.topics_num):
      p[k] = (self.nkt[k][self.docs[m][n]] + self.beta) / (self.nkt_sum[k] + self.vocabulary_size * self.beta) \
             * (self.nmk[m][k] + self.alpha) / (self.nmk_sum[m] + self.topics_num * self.alpha)
      # print p[k]
    for k in xrange(1, self.topics_num):
      p[k] += p[k - 1]

    u = random.random() * p[self.topics_num - 1]
    new_topic = 0
    while(1):
      # print p[new_topic]
      if u < p[new_topic]:
        break
      new_topic += 1
    
    if new_topic == self.topics_num:
      print 'new_topic error.'

    self.nmk[m][new_topic] += 1
    self.nkt[new_topic][self.docs[m][n]] += 1
    self.nmk_sum[m] += 1
    self.nkt_sum[new_topic] += 1
    return new_topic
  
  def print_topic_words(self):
    top_num = 15
    for k in xrange(self.topics_num):
      sorted_words_id = list((-self.phi[k]).argsort())
      words_str = 'topics ' + str(k) + ' : ' + self.document.index_to_word[sorted_words_id[0]]
      # print 'ids:', sorted_words_id
      # print 'phi:', self.phi[k][sorted_words_id]
      for i in xrange(1, top_num):
        words_str += ',' + self.document.index_to_word[sorted_words_id[i]]
      print words_str
    print '----------------------------'

  def output_theta(self, index):
    out_file = open('sentence_topics' + str(index) + '.txt', 'w')
    for m in xrange(self.sentences_num):
      topic_dist_str = str(self.theta[m][0])
      for k in xrange(1, self.topics_num):
        topic_dist_str += ',' + str(self.theta[m][k])
      item = dict()
      item['embedding'] = topic_dist_str
      new_line = json.dumps(item) + '\n'
      out_file.write(new_line)
    out_file.close()
    print 'succeed to save theta.'

    out_file2 = open('word_topics' + str(index) + '.txt', 'w')
    for m in xrange(self.vocabulary_size):
      topic_dist_str = str(self.phi[0][m])
      for k in xrange(1, self.topics_num):
        topic_dist_str += ',' + str(self.phi[k][m])
      item = dict()
      item['word'] = used_words[m]
      item['embedding'] = topic_dist_str
      new_line = json.dumps(item) + '\n'
      out_file2.write(new_line)
    out_file2.close()
    print 'succeed to save theta.'

if __name__ == '__main__':
  embeddings_file = open('used_embeddings.txt')
  embeddings_lines = embeddings_file.readlines()
  
  used_words = []
  word_to_index = dict()
  embed_rows, embed_cols = len(embeddings_lines), len(embeddings_lines[0].split(' ')) - 1
  print embed_rows, embed_cols
  embeddings = np.zeros((embed_rows, embed_cols), dtype=float)
  for i in xrange(len(embeddings_lines)):
    tmp_list = embeddings_lines[i].split(' ')
    used_words.append(tmp_list[0])
    word_to_index[tmp_list[0]] = i

    for j in xrange(embed_cols):
      embeddings[i, j] = tmp_list[j + 1]
  
  document = Document('web_snippets.json', word_to_index, used_words)
  print 'reading completed.'

  params = dict()
  params['alpha'] = 0.5
  params['beta'] = 0.1
  params['topics_num'] = 10
  params['iterations_num'] = 100001
  params['save_step'] = 40
  params['begin_save_iters'] = 80

  lda_model = LDA_Model(params, document)
  lda_model.inference_Model()
  print 'Done.'
