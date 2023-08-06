import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import seaborn as sns
import warnings

from collections import namedtuple
from matplotlib.ticker import MaxNLocator

from fanalysis.mca import MCA
from fanalysis.pca import PCA as fa_PCA

from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.decomposition import PCA as sk_PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score, \
                            silhouette_samples

#
# Note au correcteur:
# Ceci est le code que j'utilise pour faire les exercices donnés en classes. Je me suis permis de 
# l'utiliser car je pense qu'il aide la correction (quelques points à valider pour trouver erreurs
# d'implémentation) et focus mon raisonement sur l'objectifs à atteindre et non comment l'atteindre.
# Si jamais, il y avait problème, je peux fournir un historique github montrant que j'ai dévelopé
# par moi-même et non copier.
#

def show_na(data):
    na_rows = data.isna().any(axis=1)
    na_ = data[na_rows]
    
    na_ratio = na_.shape[0] / data.shape[0]
    na_ratio = round(na_ratio * 100, 1)
    print(f"Valeur manquante {na_.shape[0]} ({na_ratio}%)")
    
    if na_.shape[0] > 0:
        display(na_)

    return na_.index

def show_types(dataframe):
    print("Types")
    types_ = dataframe.dtypes.to_frame()
    types_.columns = ["Type"]
    display(types_.T)

def show_distributions(data, num_cols=5, figsize=(12, 10)):
    num_rows = math.ceil(data.shape[1] / num_cols)
    
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    for col, ax in zip(data, axes.flatten()):
        sns.histplot(x=col, data=data, ax=ax)
    
    plt.tight_layout()
    plt.show()
    
def transform_boxcox(data, columns=None):
    """
    Retourne tuple (transforms, lambdas)
    """
    result = data.copy()
    
    if columns is None:
        columns = data.columns
    
    lambdas_ = []
    for c in columns:
        doi = data[c]
        doi_min = doi.min()
        if doi_min <= 0:
            doi = doi - doi_min + 1e-6
        
        result[c], lambda_ = sp.stats.boxcox(doi)
        lambdas_.append(lambda_)
        
    return result, lambdas_

def transform_log(data, columns=None):
    result = data.copy()
    
    if columns is None:
        columns = data.columns
    
    for c in columns:
        doi = data[c]
        doi_min = doi.min()
        if doi_min < 1:
            doi = doi - doi_min + 1e-6
        
        result[c] = np.log(doi + 1)
        
    return result

def transform_sqrt(data, columns=None):
    result = data.copy()
    
    if columns is None:
        columns = data.columns
    
    for c in columns:
        doi = data[c]
        doi_min = doi.min()
        if doi_min < 0:
            doi = doi - doi_min
        
        result[c] = np.sqrt(doi)
        
    return result

def transform_cbrt(data, columns=None):
    result = data.copy()
    
    if columns is None:
        columns = data.columns
    
    for c in columns:
        doi = data[c]
        result[c] = np.cbrt(doi)
        
    return result

def show_transforms(data, columns=None, figsize=(10, 10)):
    """
    Applique et afiche diverses transformes et les retourne 
    sous forme de namedtuple
    Attention, boxcox est un tuple (DataFrame, boxcox lambda)
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        transforms_names = ["log", 
                            "boxcox",
                            "cbrt"]
            
        Transforms = namedtuple("Transforms", transforms_names)
        transforms = Transforms(
            transform_log(data, columns=columns),
            transform_boxcox(data, columns=columns),
            transform_cbrt(data, columns=columns))

    if columns is None:
        columns = data.columns

    _, axes = plt.subplots(len(columns), len(transforms_names) + 1, figsize=figsize)

    for col_index, col_name in enumerate(columns):
        axes[col_index, 0].set_ylabel(col_name)

        # normal dist
        g = sns.histplot(data[col_name].to_numpy(), ax=axes[col_index, 0])
        g.set(xlabel=None)

        # log
        new_dist = transforms.log[col_name]
        g = sns.histplot(new_dist.to_numpy(), ax=axes[col_index, 1])
        g.set(xlabel=None, ylabel=None)

        # cox box 
        new_dist = transforms.boxcox[0][col_name]
        g = sns.histplot(new_dist, ax=axes[col_index, 2])
        g.set(xlabel=None, ylabel=None)

        # cubic root
        new_dist = transforms.cbrt[col_name]
        g = sns.histplot(new_dist, ax=axes[col_index, 3])
        g.set(xlabel=None, ylabel=None)

    axes[0, 0].set_title("Dist. Originale")
    axes[0, 1].set_title("Log")
    axes[0, 2].set_title("Boxcox")
    axes[0, 3].set_title("Cubic Root")

    plt.tight_layout()
    plt.show()

    return transforms

def show_outliers_iqr(data, eta=1.5, show_outliers_values=False, boxlists=None, figsize=(8, 6.5)):
    """
    threshold ~10-15%
    """
    q1 = data.quantile(0.25)
    q3 = data.quantile(0.75)
    delta = eta * (q3 - q1)
    
    low = data < (q1 - delta)
    low_rows = low.any(axis=1)

    high = data > (q3 + delta)
    high_rows = high.any(axis=1)
                
    outliers_rows = data[low_rows | high_rows]    
    outliers_ratio_rows = outliers_rows.shape[0] / data.shape[0]
    outliers_ratio_rows = round(outliers_ratio_rows * 100, 1)
    
    outliers_cols = (low | high).sum(axis=0)
    outliers_cols.name = "Count"
    outliers_cols = outliers_cols.to_frame()
    outliers_cols["%"] = round(outliers_cols.Count / data.shape[0] * 100, 1)
    
    print(f"IQR outliers par variable, eta: {eta}")
    display(outliers_cols.T)
    print()
    
    print(f"IQR outliers {outliers_rows.shape[0]} ({outliers_ratio_rows}%), eta: {eta}")    
    if outliers_rows.shape[0] > 0 and show_outliers_values:
        outliers_values = low | high
        outliers_values_rows = outliers_values.loc[outliers_rows.index]
        outliers_values_bg = np.where(outliers_values_rows, "background-color:lightsteelblue", "")
        outliers_values_styler = outliers_rows.style.apply(lambda _: outliers_values_bg, axis=None)
        display(outliers_values_styler)
    else:
        print()
        
    if not boxlists is None: 
        print(f"Outliers boxplots, eta: {eta}")
        n = len(boxlists)
        h_ratios = [len(boxes) for boxes in boxlists]
        fig, axes = plt.subplots(n, figsize=figsize, height_ratios=h_ratios)
        if n == 1:
            sns.boxplot(data[boxlists[0]], orient="h", whis=eta, ax=axes)
        else:
            for boxes, ax in zip(boxlists, axes.flatten()):
                sns.boxplot(data[boxes], orient="h", whis=eta, ax=ax)
        plt.tight_layout()
        plt.show()
        
    return outliers_rows.index

def show_correlation(data, 
                     method='pearson', 
                     corner=True, 
                     figsize=(6, 3), 
                     pairplot=False, 
                     pairplot_figsize=(8, 6)):
    plt.figure(figsize=figsize)
    sns.heatmap(data.corr(method=method).round(2), annot=True, linewidths=0.01, ax=plt.gca())
    plt.title(f"Corrélation - {method}")
    plt.show()
    
    # mac a quelques problemes avec le temps d'execution du pairplot
    # mettre cette affichage optionel
    if pairplot:
        g = sns.pairplot(data, corner=corner)
        g.fig.set_size_inches(*pairplot_figsize)
        plt.suptitle("Pair plot")
        plt.show()

def pca_init(std_data, n_components):
    # pour fin de comparaison, choisir au runtime entre fanalysis et scklearn
    if True:
        acp = fa_PCA(std_unit=False, 
                  n_components=n_components,
                  row_labels=std_data.index,
                  col_labels=std_data.columns)
        acp.fit(std_data.to_numpy())
    else:
        acp = sk_PCA(n_components=n_components, 
                     svd_solver="full")

        # fanalysis adapters
        acp.row_coord_ = acp.fit_transform(std_data)
        acp.row_labels_ = std_data.index
        acp.col_labels_ = std_data.columns
        acp.eig_ = [acp.explained_variance_,
                    acp.explained_variance_ratio_ * 100,
                    np.cumsum(acp.explained_variance_ratio_ * 100)]

    def show2d(coords, coords_labels, alpha):
        plt.scatter(coords[:, 0], coords[:, 1])

        for xy, text in zip(coords, coords_labels):
            text_ = plt.text(xy[0], xy[1], text)
            text_.set_alpha(alpha)
                
        plt.grid(True)
        
    def show(x, y, text_alpha=0.33, figsize=(5, 4)):
        plt.figure(figsize=figsize)
        show2d(acp.row_coord_[:, [x - 1, y - 1]], \
               acp.row_labels_, \
               text_alpha)
        plt.show()

    # override mapping_row
    acp.mapping_row = show
    
    return acp

def pca_analysis(std_data, figsize=(4, 2.5)):
    """
    Le threshold est ~60% sur cumul var. expliquee
    """
    acp = pca_init(std_data, None)
    
    saporta = 1 + 2 * math.sqrt((std_data.shape[1] - 1) / (std_data.shape[0] - 1))

    eig_vals = acp.eig_[0]
    eig_th0 = eig_vals[eig_vals > 1]
    eig_th1 = eig_vals[eig_vals > saporta]

    print("Valeurs propres:")
    print(acp.eig_[0].round(4))
    print()
    print("Valeurs propres > 1:")
    print(eig_th0.round(4))
    print()
    print(f"Valeurs propres > {round(saporta, 4)} (saporta):")
    print(eig_th1.round(4))
    print()
    print("Variance expliquee %:")
    print(acp.eig_[1].round(1))
    print()
    print("Variance expliquee cumul. %:")
    print(acp.eig_[2].round(1))
    print()

    num_eigval = len(acp.eig_[0])
    
    plt.figure(figsize=figsize)
    plt.plot(range(1, num_eigval + 1), acp.eig_[0], marker=".")
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("# axe factoriel")
    plt.ylabel("Valeur propre")
    plt.show()
    
def acm_init(data, n_components):
    acm = MCA(n_components=n_components,
              row_labels=data.index,
              var_labels=data.columns)
    acm.fit(data.to_numpy())
    return acm

def acm_analysis(data, figsize=(4, 2.5)):
    """
    Le threshold est ~60% sur cumul var. expliquee
    """
    acm = acm_init(data, None)
    
    threshold = 1 / data.shape[1]
    eig_vals = acm.eig_[0]
    eig_th = eig_vals[eig_vals > threshold]

    print("Valeurs propres:")
    print(acm.eig_[0].round(4))
    print()
    print(f"Valeurs propres > {round(threshold, 4)} (1 / p):")
    print(eig_th.round(4))
    print()
    print("Variance expliquee %:")
    print(acm.eig_[1].round(1))
    print()
    print("Variance expliquee cumul. %:")
    print(acm.eig_[2].round(1))
    print()

    num_eigval = len(acm.eig_[0])

    plt.figure(figsize=figsize)
    plt.plot(range(1, num_eigval + 1), acm.eig_[0], marker=".")
    plt.grid(True)
    plt.xlabel("# axe factoriel")
    plt.ylabel("Valeur propre")
    plt.show()

def kmeans_init(coords, n_clusters, n_init=20, max_iter=300):
    clstr = KMeans(n_clusters=n_clusters, n_init=n_init, max_iter=max_iter)
    clstr.fit(coords)
    return clstr
    
def kmeans_analysis(coords, clusters_range=range(2, 15), n_init=20, max_iter=300, figsize=(5, 3)):
    inertias = []
    sil_scores = []
    
    for k in clusters_range:
        clstr = kmeans_init(coords, k, n_init, max_iter)
        inertias.append(clstr.inertia_)
        sil_scores.append( silhouette_score(coords, clstr.labels_) )
    
    _, ax1 = plt.subplots(figsize=figsize)
    
    ax1.plot(clusters_range, inertias, label="wss", color="green", marker=".")
    ax1.set_xlabel("# clusters")
    ax1.set_ylabel("wss")
    ax1.grid(True)
    
    ax2 = plt.gca().twinx()
    ax2.plot(clusters_range, sil_scores, label="silhouette score", color="blue", marker=".")
    ax2.set_ylabel("silhouette score")

    g1, gl1 = ax1.get_legend_handles_labels()
    g2, gl2 = ax2.get_legend_handles_labels()
    plt.legend(g1 + g2, gl1 + gl2)

    plt.show()
    
def cah_init(coords, n_clusters):
    cah = AgglomerativeClustering(n_clusters=n_clusters, 
                                  metric="euclidean", 
                                  linkage='ward')
    cah.fit(coords)
    return cah

def cah_analysis(coords, figsize=(7, 7)):
    """
    Le threshold est a peu pres a la moitie de la hauteur
    """
    linkage_ = linkage(coords, method="ward", metric="euclidean")

    plt.figure(figsize=figsize)
    plt.subplot(211)
    dendrogram(linkage_)
    plt.title("Dendogramme")
        
    cluster_inertias = linkage_[-15:, 2]
    cluster_inertias = cluster_inertias[::-1]
    
    plt.subplot(212)
    plt.step(range(2, len(cluster_inertias) + 2), cluster_inertias)
    plt.xlabel("# clusters")
    plt.ylabel("Inertie")
    plt.grid()
    
    plt.show()
    
def dbscan_init(coords, eps, min_samples):
    dbs = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
    dbs.fit(coords)
    return dbs

def dbscan_eps_analysis(coords, figsize=(5, 3), ylim=None):
    nn = NearestNeighbors(n_neighbors=2)
    distances, _ = nn.fit(coords).kneighbors(coords)
    distances = np.sort(distances, axis=0)

    plt.figure(figsize=figsize)
    plt.plot(distances[:,1], marker=".")
    if not ylim is None:
        plt.ylim(ylim)
    plt.ylabel("Distance")
    plt.xlabel("Individu")
    plt.grid(True)
    plt.show()

def dbscan_parameters_analysis(coords, eps_range, min_samples_range):
    score_max = -1e9
    eps_ = 0
    min_samples_ = 0
    n_clusters_ = 0
    with_ouliers_ = False
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbs = dbscan_init(coords, eps, min_samples)

            clusters = set(dbs.labels_)
            with_ouliers = -1 in clusters
            n_clusters = len(clusters) - (1 if with_ouliers else 0)

            # si dbscan ne donne que 1 seul cluster (-1, outliers)
            # bail out            
            if n_clusters < 2:
                continue
            
            score = silhouette_score(coords, dbs.labels_)
            if score > score_max:
                score_max = score
                eps_ = eps
                min_samples_ = min_samples
                n_clusters_ = n_clusters
                with_ouliers_ = with_ouliers
                    
    print("DBSCAN optimal parameters")
    print("eps:", eps_)
    print("min_samples:", min_samples_)
    print("silhouette score:", round(score_max, 4))
    print("# clusters:", n_clusters_, "+ ouliers" if with_ouliers_ else "(no outliers)")
                
    return eps_, min_samples_

def dbscan_outliers_analysis(coords, eps_range, min_samples, figsize=(5, 3)):
    outliers_ratio = []
    for eps in eps_range:
        dbs = dbscan_init(coords, eps, min_samples)
        outliers = dbs.labels_[dbs.labels_ == -1]
        ratio = len(outliers) / len(dbs.labels_)
        outliers_ratio.append(ratio)
        
    plt.figure(figsize=figsize)
    plt.plot(eps_range, outliers_ratio, marker=".")
    plt.xlabel("dbscan epsilon")
    plt.ylabel("outliers ratio")
    plt.grid(True)
    plt.show()

def clusters_analysis(coords, labels, original_data=None):
    score = silhouette_score(coords, labels)
    print("Silhouette score:", round(score, 4))
    print()
    
    samples = silhouette_samples(coords, labels)
    samples_means = []
    clusters = set(labels)
    for k in clusters:
        labels_k = labels == k
        
        if labels_k.any() > 0:
            sample_mean = samples[labels_k].mean()
            sample_mean = round(sample_mean, 4)
            samples_means.append(sample_mean)
        else:
            samples_means.append(np.nan)
        
    print("Silhouette score par cluster")
    print(samples_means)
    print()
    
    if not original_data is None:
        tss = (original_data.mean() - original_data) ** 2
        tss = tss.sum(axis=0)

        groups = original_data.groupby(labels)

        bss = (original_data.mean() - groups.mean()) ** 2
        bss = bss.multiply(groups.size(), axis=0)
        bss = bss.sum(axis=0)

        r2 = bss / tss
        r2.name = "$R^2$"

        print("Variance expliquée par les clusters")
        display(r2.round(2).to_frame().T)
    
def show_clusters(coords_, coords_name_, labels, figsize=(5, 4), text_alpha=1, marker_size=None):
    clusters = set(labels)
    n_clusters = len(clusters)
    
    colors_steps = np.arange(0, 1, 1 / n_clusters)
    colors = plt.cm.nipy_spectral(colors_steps)
    
    plt.figure(figsize=figsize)
    for k in clusters:
        cluster = labels == k
        coords = coords_[cluster].to_numpy()
        coords_name = coords_name_[cluster]

        label = "Outliers" if k == -1 else f"Cluster_{k}"
        plt.scatter(coords[:, 0], coords[:, 1], label=label, color=colors[k], s=marker_size)
        
        for xy, text in zip(coords, coords_name_):
            text_ = plt.text(xy[0], xy[1], text)
            text_.set_alpha(text_alpha)

    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1.0))
    plt.show()