# encoding=utf-8
from keras.callbacks import Callback, ModelCheckpoint
import prf

# 该回调函数将在每个epoch后保存概率文件
class WritePRF(Callback):
	def __init__(self, probability_file, data, label):
		super(WritePRF, self).__init__()
		self.probability_file = probability_file
		self.data = data
		self.label = label

	def on_epoch_end(self, epoch, logs=None):
		resultFile = self.probability_file + str(epoch+1) + '.txt'
		predictions = self.model.predict(self.data)  # 测试
		prf.calculate(predictions, self.label, resultFile, epoch)


# 该回调函数将在每个迭代后保存的最好模型
class checkpoint():
	def __init__(self, best_model_file):
		self.model_file = best_model_file

	def check(self):
		checkpoint = ModelCheckpoint(filepath=self.model_file, monitor='val_loss',
									 verbose=1, save_best_only=True, mode='min')
		return checkpoint
