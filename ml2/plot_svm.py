from patsy import dmatrices
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm

#Taken from http://nbviewer.jupyter.org/github/agconti/kaggle-titanic/blob/master/Titanic.ipynb

def plot_svm(df):
	# set plotting parameters
	plt.figure(figsize=(8,6))

        # # Create an acceptable formula for our machine learning algorithms
	formula_ml = 'Survived ~ C(Pclass) + C(Sex) + Age + SibSp + Parch + C(Embarked)'
	# create a regression friendly data frame
	y, x = dmatrices(formula_ml, data=df, return_type='matrix')

	# select which features we would like to analyze
	# try chaning the selection here for diffrent output.
	# Choose : [2,3] - pretty sweet DBs [3,1] --standard DBs [7,3] -very cool DBs,
	# [3,6] -- very long complex dbs, could take over an hour to calculate! 
	feature_1 = 2
	feature_2 = 3

	X = np.asarray(x)
	X = X[:,[feature_1, feature_2]]  


	y = np.asarray(y)
	# needs to be 1 dimensional so we flatten. it comes out of dmatrices with a shape. 
	y = y.flatten()      

	n_sample = len(X)

	np.random.seed(0)
	order = np.random.permutation(n_sample)

	X = X[order]
	y = y[order].astype(np.float)

	# do a cross validation
	nighty_precent_of_sample = int(.9 * n_sample)
	X_train = X[:nighty_precent_of_sample]
	y_train = y[:nighty_precent_of_sample]
	X_test = X[nighty_precent_of_sample:]
	y_test = y[nighty_precent_of_sample:]

	# create a list of the types of kerneks we will use for your analysis
	types_of_kernels = ['linear', 'rbf', 'poly']

	# specify our color map for plotting the results
	color_map = plt.cm.RdBu_r

	# fit the model
	for fig_num, kernel in enumerate(types_of_kernels):
    		clf = svm.SVC(kernel=kernel, gamma=3)
    		clf.fit(X_train, y_train)

    		plt.figure(fig_num)
    		plt.scatter(X[:, 0], X[:, 1], c=y, zorder=10, cmap=color_map)

    		# circle out the test data
    		plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none', zorder=10)
    
    		plt.axis('tight')
   	 	x_min = X[:, 0].min()
    		x_max = X[:, 0].max()
    		y_min = X[:, 1].min()
    		y_max = X[:, 1].max()

    		XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    		Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    		# put the result into a color plot
    		Z = Z.reshape(XX.shape)
    		plt.pcolormesh(XX, YY, Z > 0, cmap=color_map)
    		plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'],
               		levels=[-.5, 0, .5])

    		plt.title(kernel)
    		plt.show()
