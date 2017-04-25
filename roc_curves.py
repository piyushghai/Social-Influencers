import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plotROCCuves(y_test, y_pred, filename, label):
	false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_pred)
	roc_auc = auc(false_positive_rate, true_positive_rate)
	roc_label = '{0} {1:0.5f}'.format(label, roc_auc)
	plt.plot(false_positive_rate, true_positive_rate, label=roc_label, linewidth=2)
	plt.title('Receiver Operating Characteristic (ROC)')
	plt.legend(loc='lower right')
	plt.plot([0, 1], [0, 1], 'k--')     # plot the diagonal
	plt.xlim([-0.1, 1.2])
	plt.ylim([-0.1, 1.2])
	plt.ylabel('True Positive Rate')
	plt.xlabel('False Positive Rate')
	plt.savefig(filename)
	plt.show()