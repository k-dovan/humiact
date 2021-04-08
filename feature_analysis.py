
from os.path import join
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

categories = ["Boxing", "Facing", "Handholding", "Handshaking", "Hugging", "Kissing"]

def keypoint_feature_analysis():
    df_ds = pd.DataFrame()
    for cat in categories:
        df = pd.read_csv(join('dataset/train', cat, cat + "_keypoint_feature.csv"), index_col=0)
        labels = [cat for i in range(len(df.index))]
        df.insert(df.shape[1], "label", labels)
        df_ds = df_ds.append(df)

    # separate features and labels
    X = df_ds.iloc[:, :-1]
    y = df_ds.iloc[:, -1]

    # apply scaling to entire dataset
    trans = StandardScaler()
    X = trans.fit_transform(X)

    pca = PCA(n_components=2)
    # fit PCA on training set
    pca.fit(X)
    # apply mapping (transform) to both training and test set
    X = pca.transform(X)
    # labels = ["PC %d (var:%.2f)" % (i,var) for i, var in enumerate(pca.explained_variance_ratio_ * 100)]
    principalDf = pd.DataFrame(data=X
                               , columns=['principal component 1', 'principal component 2'])
    principalDf.reset_index(drop=True, inplace=True)
    y.reset_index(drop=True, inplace=True)
    finalDf = pd.concat([principalDf, y], axis=1)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    lbl1 = 'Principal Component 1 (var: %.2f)' % (pca.explained_variance_ratio_[0]*100)
    lbl2 = 'Principal Component 2 (var: %.2f)' % (pca.explained_variance_ratio_[1]*100)
    ax.set_xlabel(lbl1, fontsize=15)
    ax.set_ylabel(lbl2, fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    for lbl, color in zip(categories, colors):
        indicesToKeep = finalDf['label'] == lbl
        ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
                   , finalDf.loc[indicesToKeep, 'principal component 2']
                   , c=color
                   , s=50)
    ax.legend(categories)
    ax.grid()

    plt.show()

# run PCA analysis
keypoint_feature_analysis()

