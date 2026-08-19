import csv, sys

# read tsv files with the four essential columns: id, annotator, position, answer_coordinates
# output: dt: sid -> pos -> ansCord
def readTsv(fnTsv):
	dt = {}
	for row in csv.DictReader(open(fnTsv, 'r'), delimiter='\t'):
		sid = row['id'] + '\t' + row['annotator']   # sequence id
		pos = row['position']   # position
		ansCord = set(eval(row['answer_coordinates'])) # answer coordinates
		if not sid in dt:
			dt[sid] = {}
		dt[sid][pos] = ansCord
	return dt


def evaluate(fnGold, fnPred):

	dtGold = readTsv(fnGold)
	dtPred = readTsv(fnPred)

    # Calcuate both sequence-level accuracy and question-level accuracy
	seqCnt = seqCor = 0
	ansCnt = ansCor = 0
	for sid,qa in dtGold.items():
		seqCnt += 1
		ansCnt += len(qa)

		if sid not in dtPred: continue	# sequence does not exist in the prediction
		
		predQA = dtPred[sid]
		allQCorrect = True
		for q,a in qa.items():
			if q in predQA and a == predQA[q]: 
				ansCor += 1	# correctly answered question
			else:
				allQCorrect = False
		if allQCorrect: seqCor += 1
		
	print (f"Sequence Accuracy = %0.2f%% (%d/%d)", {(100.0 * seqCor/seqCnt)}, {seqCor}, {seqCnt})
	print (f"Answer Accuracy =   %0.2f%% (%d/%d)", {(100.0 * ansCor/ansCnt)}, {ansCor}, {ansCnt})

	return [seqCor, seqCnt, ansCor, ansCnt]


if __name__ == '__main__':
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: %s goldTsv predTsv\n" % sys.argv[0])
        sys.exit(-1)
    fnGold = sys.argv[1]
    fnPred = sys.argv[2]
    evaluate(fnGold, fnPred)
