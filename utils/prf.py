from __future__ import print_function
import numpy as np
from keras.callbacks import Callback


def calculate(predictions, test_label, prf_file, epoch):
	num = len(predictions)

	TP = len([1 for i in range(num) if
			  predictions[i][1] > predictions[i][0] and (test_label[i] == np.asarray([0, 1])).all()])
	FP = len([1 for i in range(num) if
			  predictions[i][1] > predictions[i][0] and (test_label[i] == np.asarray([1, 0])).all()])
	FN = len([1 for i in range(num) if
			  predictions[i][1] < predictions[i][0] and (test_label[i] == np.asarray([0, 1])).all()])
	TN = len([1 for i in range(num) if
			  predictions[i][1] < predictions[i][0] and (test_label[i] == np.asarray([1, 0])).all()])

	precision = recall = Fscore = 0, 0, 0
	try:
		precision = TP / (float)(TP + FP)  # ZeroDivisionError: float division by zero
		recall = TP / (float)(TP + FN)
		Fscore = (2 * precision * recall) / (precision + recall)
	except ZeroDivisionError as exc:
		print(exc.message)

	print(">> Report the result ...")
	print("-1 --> ", len([1 for i in range(num) if predictions[i][1] < predictions[i][0]]))
	print("+1 --> ", len([1 for i in range(num) if predictions[i][1] > predictions[i][0]]))
	print("TP=", TP, "  FP=", FP, " FN=", FN, " TN=", TN)
	print('\n')
	print("precision= ", precision)
	print("recall= ", recall)
	print("Fscore= ", Fscore)

	with open(prf_file, 'a') as pf:
		print('write prf...... ')
		pf.write("epoch= " + str(epoch+1) + '\n')
		pf.write("precision= " + str(precision) + '\n')
		pf.write("recall= " + str(recall) + '\n')
		pf.write("Fscore= " + str(Fscore) + '\n\n')




class computeF1(Callback):
    def __init__(self, x_test, y_test):
        # self.X_words_test = X_test
        self.x_test = x_test
        self.y_test = y_test

    def on_epoch_end(self, epoch, logs={}):
        # X_test = self.X_words_test
        predictions = self.model.predict_generator(self.x_test,
                                                   steps=len(self.y_test)//batch_size,
                                                   verbose=1)
        y_pred = predictions[0].argmax(axis=-1)  # Predict classes
        self.y_test = self.y_test.argmax(axis=-1)
        print(len(y_pred), len(self.y_test))

        iprf_file = 'results/prf.txt'
        target = test_file

        p, r, f, c = predictLabels1(target, y_pred)

        with open(prf_file, 'a') as pf:
            print('write prf...... ')
            pf.write("epoch= " + str(epoch + 1) + '\n')
            pf.write("precision= " + str(pre) + '\t' + str(p) + '\n')
            pf.write("recall= " + str(rec) + '\t' + str(r) + '\n')
            pf.write("Fscore= " + str(f1) + '\t' + str(f) + '\n')
            pf.write("processed %d tokens with %d phrases;\n" % (c.token_counter, c.found_correct))
            pf.write('found: %d phrases; correct: %d.\n\n' % (c.found_guessed, c.correct_chunk))


def predictLabels1(target, y_pred, flag=None):
    s = []
    sentences = []
    s_num = 0
    with open(target) as f:
        for line in f:
            if not line == '\n':
                s.append(line.strip('\n'))
                continue
            else:
                prediction = y_pred[s_num]
                s_num += 1
                for i in range(len(s)):
                    # if i >= maxlen_s: break
                    r = s[i] + '\t' + idx2label1[prediction[i]] + '\n'
                    sentences.append(r)
                sentences.append('\n')
                s = []
    with open('results/result.txt', 'w') as f:
        for line in sentences:
            f.write(str(line))

    p, r, f, c = conlleval.main((None, r'results/result.txt'))
    return round(Decimal(p), 2), round(Decimal(r), 2), round(Decimal(f), 2), c


