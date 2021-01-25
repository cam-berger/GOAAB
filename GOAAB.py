import numpy as np
import pandas as pd
import subprocess
import random
import matplotlib.pyplot as plt
import argparse
import shutil
import os
import sys
import _pickle as cPickle
from scapy.all import *
from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import plot_confusion_matrix

# Cleans a processed numerical dataset
def clean(df):
    # Naieve solution: Drop infinite and NaN values from processed data
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"

    return df.replace([np.inf, -np.inf, np.nan], value=0)

    return df

# Split the data into numerical features for X and Y
def get_xy(df, train):
    data = df.drop(['Flow ID', 'Src IP', 'Dst IP', 'Timestamp'], axis=1)
    if not train:
        length = data.shape[0]
        for i in range(0, length):
            data.at[i, 'Label'] = 0
        data = data.astype({'Label': 'int32'})
    if train:
        #make sure there are labels on data
        if data['Label'].dtype == 'object':
            print("Invalid Format: No Labels")
            sys.exit(-1)
    data = clean(data)
    x_data = data.drop('Label', axis=1)
    x = x_data.to_numpy()
    y = data['Label'].to_numpy()
    data = None
    x_data = None

    return x, y

# Split x and y for training and testing
def test_train(x, y):
    return train_test_split(x, y, test_size=.33)

# Module: Support Vector Classifier
def runSVC(x_train, y_train, k, d, c):
    print("Training Support Vector Classifier:")

    kerns = ['linear', 'poly', 'rbf', 'sigmoid']
    clf = ''
    #if using poly as kernel
    if k==1:
        #degree between 3 and 20
        d = d%18+3
        print("\tkernel = "+kerns[k]+", with degree "+str(d))
        clf = SVC(kernel=kerns[k], C=c, degree=d).fit(x_train, y_train)
    else:
        print("\tkernel = "+kerns[k])
        clf = SVC(kernel=kerns[k], C=c).fit(x_train, y_train)
    return clf

# Module: Logistic Regression
def runLR(x_train, y_train, k, p, c):
    print("Training Logistic Regression Classifier:")
    k = k%5
    sol = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    pen = ['l1', 'l2', 'none']
    clf = ''
    # the following solvers only support l2 penalty
    if k == 0 or k == 1 or k == 3:
        p = 1
    elif k == 2:
        p = p%2
    elif k == 4:
        p = p%3

    print("\tsolver = "+sol[k]+", with penalty "+pen[p])
    clf = LogisticRegression(solver = sol[k], penalty=pen[p], C=c, max_iter=10000).fit(x_train, y_train)

    return clf

# Module: Gaussian Naive Bayes
def runGNB(x_train, y_train, v):
    print("Training Gaussian Naive Bayes Classifier:")
    v = -1*(v%11 + 1)
    v = 1*(10**v)
    print("\tvariable smoothing parameter = "+"{:1.0e}".format(v))
    clf = GaussianNB(var_smoothing = v).fit(x_train, y_train)
    return clf

# Module: Random Forest Classifier
def runRFC(x_train, y_train):
    print("Training Random Forest Classifier")
    clf = RandomForestClassifier(max_depth=2, random_state=0).fit(x_train, y_train)
    return clf

# Module: Standard Scalar
def runSS(x, train):
    print("Performing Standard Scalar")
    dump = 'dumped_ss.pkl'
    trans = ''

    if train:
        trans = StandardScaler()
        return trans.fit_transform(x)
        with open(dump, 'wb') as fid:
            cPickle.dump(trans, fid)
    else:
        with open(dump, 'rb') as fid:
            trans = cPickle.load(fid)

    return trans.fit_transform(x)

# Module: Max Absolute Scalar
def runMaxAbs(x, train):
    print("Performing Max Absolute Scalar")
    dump = 'dumped_mas.pkl'

    if train:
        trans = MaxAbsScaler()
        return trans.fit_transform(x)
        with open(dump, 'wb') as fid:
            cPickle.dump(trans, fid)
    else:
        with open(dump, 'rb') as fid:
            trans = cPickle.load(fid)

    return trans.fit_transform(x)

# Module: Normalization
def runNorm(x, train, n):
    print("Performing Normalization")

    dump = 'dumped_norm.pkl'
    if train:
        n = n%3
        norms = ['l1', 'l2','max']
        trans = Normalizer(norm = norms[n])
        with open(dump, 'wb') as fid:
            cPickle.dump(trans, fid)
    else:
        with open(dump, 'rb') as fid:
            trans = cPickle.load(fid)

    return trans.fit_transform(x)

# Module: Polynomial Features
def runPoly(x, train, d, inter):
    print("Performing Polynomial Features")

    dump = 'dumped_pf.pkl'
    if train:
        d = d%2 + 2
        inter = inter%2
        if inter == 0:
            inter = False
        else:
            inter = True
        trans = PolynomialFeatures(degree = d, interaction_only=inter)
        with open(dump, 'wb') as fid:
            cPickle.dump(trans, fid)
    else:
        with open(dump, 'rb') as fid:
            trans = cPickle.load(fid)

    return trans.fit_transform(x)

# Uses CICFlowMeter to extract features from input PCAP file
# returns the filename of the processed data in csv format
def convert_pcap(file_name):
    print("Opening {}...".format(file_name))

    # Move .pcap file from working directory to be converted
    cwd = os.getcwd() + '/'
    pcap_loc = cwd + "CICFlowMeter-4.0/bin/FromPCAP/"
    csv_loc = cwd + "CICFlowMeter-4.0/bin/ToCSV/"
    os.rename(cwd + file_name, pcap_loc + file_name)

    csv_fname = file_name + "_Flow.csv"
    # Perform converstion
    os.chdir('CICFlowMeter-4.0/bin')
    convertCommand = "./cfm FromPCAP ToCSV"
    subprocess.call(convertCommand.split())
    os.chdir('../../')

    # Move .pcap file back to working directory
    os.rename(pcap_loc + file_name, cwd + file_name)

    file_exist = os.path.isfile(csv_loc + csv_fname)
    if file_exist == False:
      return "DNE"

    # Move .csv converstion to working directory
    os.rename(csv_loc + csv_fname, cwd + csv_fname)
    print("Saved \'"+file_name+"\' as \'"+csv_fname+"\'")

    return csv_fname

# MAIN
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GOAAB')
    parser.add_argument('--pcap', metavar='<pcap file name>', help ='pcap file to be parsed', required=False)
    parser.add_argument('--csv', metavar='<csv file name>', help ='csv file for training or testing', required=False)
    parser.add_argument('--train', action="store_true", help = 'The \'--train\' option takes a classified dataset for training purposes')
    args = parser.parse_args()

    # are we in training mode?
    train = args.train

    # get filename of pcap (if it exists)
    file_name = args.pcap
    # otherwise get the csv filename
    if file_name==None:
        file_name = args.csv
        if file_name==None:
            print('No file name argument')
            sys.exit(-1)
    # validate file
    if not os.path.isfile(file_name):
        print('"{}" does not exist'.format(file_name))
        sys.exit(-1)

    # locations for dumped classifiers
    d_svd = 'dumped_svd.pkl'
    d_gnb = 'dumped_gnb.pkl'
    d_lgr = 'dumped_lgr.pkl'

    # if we are not training and we have a pcap file, process file to csv
    if args.pcap!=None:
        orig_name = file_name
        file_name = convert_pcap(file_name)
        if file_name == "DNE":
            print("\nFile: \'" + orig_name + "\' is improperly formatted")
            sys.exit(-1)
        if train:
            print("\nFile: \'" + file_name + "\' exported for labeling")
            sys.exit(0)

    # get dataframe of processed data and from this get x and y features
    print("Opening {}...".format(file_name))
    df = pd.read_csv(file_name)

    if df.shape[1] != 84:
        print("Invalid Format: Incorrect dimensions")
        sys.exit(-1)
    x, y = get_xy(df, train)

    # STEP 1: PRE-PROCESSING
    # Options:
    pp = 0
    if pp == 0:
        # Standard Scalar
        x = runSS(x, train)
        pre = "Standard Scaling"
    elif pp == 1:
        # Max Absolute Scalar
        x = runMaxAbs(x, train)
        pre = "Max Abs Scaling"
    elif pp == 2:
        # Normalization
        n = random.randint(0, 4)
        x = runNorm(x, train, n)
        pre = "Normalization"
    # elif pp == 3:
    #      # PolynomialFeatures
    #     d = random.randint(0, 2)
    #     inter = random.randint(0, 2)
    #     x = runPoly(x, train, 2, 1)
    #     pre = "Polynomial Features"

    # MORE TRANSFORMATION MODULES
    # i.e. PCA, SVD, add other feature sets, etc.

    # Split dataset after processing
    x_train, x_test, y_train, y_test = test_train(x, y)

    # Load classifier (clf is overwritten if in training mode)
    # NOTE: we might want to add functionality that keeps training our model with
    #       new input data
    clf = ''

    # STEP 3: CHOOSING CLASSIFIER
    with open(d_lgr, 'rb') as fid: # SVD: d_svd, LogReg: d_lgr, GaussianNB: d_gnb
        clf = cPickle.load(fid)

    if train:
        np.set_printoptions(precision=2)
        class_names = ['Malicious','Normal']
        use = 1
        if use == 0:
            k = random.randint(0, 12)
            d = random.randint(0, 16)
            c = 1
            clf = runSVC(x_train, y_train, k, d, c) # For Support Vector Classifier
            cl = "Support Vector Classifier"
        elif use == 1:
            k = random.randint(0, 12)
            p = random.randint(0, 12)
            c = 1
            clf = runLR(x_train, y_train, k, p, c) # For Logistic Regression
            cl = "Logistic Regression"
        elif use == 2:
            v = random.randint(0, 9)
            clf = runGNB(x_train, y_train, v) # For Gaussian Naieve Bayes
            cl = "Gaussian Naieve Bayes"
        elif use == 3:
            clf = runRFC(x_train, y_train)
            cl = "Random Forrest Classifier"

        print("Accuracy =",clf.score(x_test, y_test))

        title = cl + ' with ' + pre

        # Plot confusion matrix
        titles_options = [(title, None),
                          ("Normalized Confusion Matrix", 'true')]

        for title, normalize in titles_options:
            disp = plot_confusion_matrix(clf, x_test, y_test, display_labels=class_names,
                                         cmap=plt.cm.Blues,normalize=normalize)
            disp.ax_.set_title(title)

            print(title)
            print(disp.confusion_matrix)

        plt.show()
    # If not in training export results of prediction
    else:
        r = clf.predict(x)
        print("Results:")
        print(r)
        pd.DataFrame(r).to_csv("Results.csv")
        print("Results saved in \'Results.csv\'")

    # Save the classifier
    with open(d_lgr, 'wb') as fid:
        cPickle.dump(clf, fid)

    sys.exit(0)
