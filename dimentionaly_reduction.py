
def reduction(data):
    from sklearn.decomposition import PCA
    pca=PCA(n_components=51)
    data=pca.fit_transform(data)
    return data


