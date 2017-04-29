# This Python file uses the following encoding: utf-8
import helpers
import random

fname = 'pos.out'

with open(fname, 'r') as f:
	lines = f.readlines()
'''
with open('data2.out', 'r') as f:
	lines.extend(f.readlines())

random.shuffle(lines)
with open('data.txt', 'w') as f:
	for s in lines:
		f.write(s)
'''

q = []
r = []
print len(lines)
vocab = helpers.load_vocab()
for i in range(len(lines)):
	if i%2 == 0:
		q.append(helpers.sentence2vector(lines[i].decode('utf-8'), vocab=vocab))
	else:
		r.append(helpers.sentence2vector(lines[i].decode('utf-8'), vocab=vocab))

for i in range(len(q)):
	q[i] = " ".join(str(x) for x in q[i])
	r[i] = " ".join(str(x) for x in r[i])
#random.shuffle(r)

with open('pos.out', 'w') as f:
	for i in range(len(q)):
		f.write(q[i] + ',' + r[i] + ',1\n')

