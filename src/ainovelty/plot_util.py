import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.lda import LDA
from ml_util import *
from ainovelty.cmds import cmdscale


__seed__ = 123456

__output_folder__ = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
__fig_dpi__ = 100


class DimReductionType:
    MDS, CMDS, Isomap, SpectralEmbedding, PCA, LDA  = range(0, 6)

def plot_scatter(self, inst_matrix, alg_matrix, inst_rank_matrix, alg_rank_matrix, main_title, hide_axis_labels):
    plt.figure(1)
    plt.suptitle(main_title, fontsize=20)

    #plt.subplots_adjust(bottom = 0.1)
    #plt.scatter( coords[:, 0], coords[:, 1], marker = 'o')
    plt.subplot(221)
    plt.scatter(inst_matrix[:, 0], inst_matrix[:, 1], marker = 'o', s=50, alpha=0.4)
    plt.title('Instances')
    plt.axis('equal')

    if hide_axis_labels == True:
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])

    #Vt = V.transpose()

    plt.subplot(222)
    plt.scatter(alg_matrix[:, 0], alg_matrix[:, 1], marker = 's', c = 'r', s=50, alpha=0.4)
    plt.title('Algorithms')
    plt.axis('equal')
    if hide_axis_labels == True:
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])


    plt.subplot(223)
    plt.scatter(inst_rank_matrix[:, 0], inst_rank_matrix[:, 1], marker = 'o', s=50, alpha=0.4)
    plt.title('Rank-Instances')
    plt.axis('equal')
    if hide_axis_labels == True:
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])


    plt.subplot(224)
    plt.scatter(alg_rank_matrix[:, 0], alg_rank_matrix[:, 1], marker = 's', s=50, alpha=0.4, color='red')
    plt.title('Rank-Algorithms')
    plt.axis('equal')
    if hide_axis_labels == True:
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])

    plt.interactive(False)

    ## to do: save .eps files under the root folder not src
    eps_to_save_path = os.path.join(__output_folder__, main_title+'.eps')
    print("eps_to_save_path: ", eps_to_save_path)
    plt.savefig(eps_to_save_path, format='eps', dpi=__fig_dpi__, bbox_inches='tight')
    ##plt.show()


def plot_scatter(twod_matrix, title, hide_axis_labels=True, plt_marker = 'o', plt_s = 30, plt_alpha = 0.4):
    '''
        Scatter plot

        :param twod_matrix: numpy 2D matrix - data points, each with represented in 2 dimensions (x,y)
        :param title: string - plot title
        :param hide_axis_labels: boolean - whether to hide plot's axis labels
        :param plt_marker: string - marker
        :param plt_s: int - marker size
        :param plt_alpha: float - marker transparency
        :return:
    '''
    plt.scatter(twod_matrix[:, 0], twod_matrix[:, 1], marker = plt_marker, s = plt_s, alpha = plt_alpha)
    plt.title(title)
    plt.axis('equal')

    if hide_axis_labels == True:
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])

    plt.show()
    plt.gcf().clear()


def create_subplot_scatter(twod_matrix, title, ax=None, hide_axis_labels=False, plt_marker = 'o', plt_s = 30, plt_alpha = 0.4):
    '''
        Create a scatter plot as a subplot, which can be combined with other plots in a figure

        :param twod_matrix: numpy 2D matrix - data points, each with represented in 2 dimensions (x,y)
        :param title: string - plot title
        :param ax: plot axis
        :param hide_axis_labels: boolean - whether to hide plot's axis labels
        :param plt_marker: string - marker
        :param plt_s: int - marker size
        :param plt_alpha: float - marker transparency
        :return scatter subplot:
    '''
    if ax is None:
        ax = plt.gca()
    ax.set_title(title)
    ax.axis('equal')

    if hide_axis_labels == True:
        ax.axes.get_xaxis().set_ticks([])
        ax.axes.get_yaxis().set_ticks([])

    scp = ax.scatter(twod_matrix[:, 0], twod_matrix[:, 1], marker = plt_marker, s = plt_s, alpha = plt_alpha)

    return scp


def plot_init(plt, main_title):
    '''
        Initialize plot as a figure

        :param plt: plot
        :param main_title: string - plot title
    '''
    fig = plt.figure(1)
    # use the next line to remove space between subplots
    #fig.subplots_adjust(wspace=0)
    plt.suptitle(main_title, fontsize=20)

def add_plot(plt, plt_data, plt_inx, title, hide_axis_labels=True, plt_marker = 'o', plt_c = 'b', plt_s = 30, plt_alpha = 0.4):
    '''
        Add a

        :param plt:
        :param plt_data:
        :param plt_inx:
        :param title:
        :param hide_axis_labels:
        :param plt_marker:
        :param plt_c:
        :param plt_s:
        :param plt_alpha:
        :return:
    '''

    plt.subplot(plt_inx)
    plt.scatter(plt_data[:, 0], plt_data[:, 1], marker = plt_marker, c = plt_c, s = plt_s, alpha = plt_alpha)
    plt.title(title)
    plt.axis('equal')

    if hide_axis_labels == True:
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])



def plot_inst_novelty_figure(dio, Ur, main_title, to_show):

    ft_rank_comb_matrix = np.concatenate((dio.i_ft_matrix, dio.ia_rank_matrix), axis=1)
    norm_ft_rank_comb_matrix = normalize_minmax(ft_rank_comb_matrix)

    ft_perf_comb_matrix = np.concatenate((dio.i_ft_matrix, dio.ia_perf_matrix), axis=1)
    norm_ft_perf_comb_matrix = normalize_minmax(ft_perf_comb_matrix)

    ft_latent_comb_matrix = np.concatenate((dio.i_ft_matrix, Ur), axis=1)
    norm_ft_latent_comb_matrix = normalize_minmax(ft_latent_comb_matrix)

    norm_ft_matrix = normalize_minmax(dio.i_ft_matrix)

    mds_norm_ft_matrix = apply_MDS_with_ddissimilarity(norm_ft_matrix, 2)
    mds_Ur = apply_MDS_with_ddissimilarity(Ur, 2)
    mds_norm_ft_rank_comb_matrix = apply_MDS_with_ddissimilarity(norm_ft_rank_comb_matrix, 2)
    mds_norm_ft_perf_comb_matrix = apply_MDS_with_ddissimilarity(norm_ft_perf_comb_matrix, 2)
    mds_norm_ft_latent_comb_matrix = apply_MDS_with_ddissimilarity(norm_ft_latent_comb_matrix, 2)

    plot_init(plt, main_title)
    add_plot(plt, mds_norm_ft_matrix, 231, 'descriptive features')
    add_plot(plt, mds_Ur, 232, 'latent features', plt_c='r')
    add_plot(plt, mds_norm_ft_rank_comb_matrix, 233, 'descriptive+rank features', plt_c='k')
    add_plot(plt, mds_norm_ft_perf_comb_matrix, 234, 'descriptive+runtime features', plt_c='y')
    add_plot(plt, mds_norm_ft_latent_comb_matrix, 235, 'descriptive+latent features', plt_c='m')

    eps_to_save_path = os.path.join(__output_folder__, main_title+'.eps')
    print("eps_to_save_path: ", eps_to_save_path)
    plt.savefig(eps_to_save_path, format='eps', dpi=__fig_dpi__, bbox_inches='tight')

    if to_show == True:
        plt.show()


def plot_alg_novelty_figure(dio, Vr, main_title, to_show):

    mds_Vr = apply_MDS_with_ddissimilarity(Vr, 2)

    plot_init(plt, main_title)
    add_plot(plt, mds_Vr, 111, 'latent features')

    eps_to_save_path = os.path.join(__output_folder__, main_title+'.eps')
    print("eps_to_save_path: ", eps_to_save_path)
    plt.savefig(eps_to_save_path, format='eps', dpi=__fig_dpi__, bbox_inches='tight')

    if to_show == True:
        plt.show()

    plt.gcf().clear()


def plot_cluster_scatter(data, cluster_labels, hide_axis_labels = True):
    plt.scatter(data[:, 0], data[:, 1], marker = 'o', s=100, alpha=0.4, c = cluster_labels.astype(np.float))
    plt.title('K-Means Clustering, k = '+str(len(np.unique(cluster_labels))))
    plt.axis('equal')
    if hide_axis_labels == True:
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])

    plt.show()
    plt.gcf().clear()

def plot_ft_importance_from_clsf(dataset_name, ft_importance_arr, top_ft_inx_arr, num_mapped_clusters, plot_title, output_folder=None, to_show = False):
    '''
        Plot a feature importance bar chart
    '''
    y_pos = np.arange(len(ft_importance_arr))
    bar_list = plt.barh(y_pos, ft_importance_arr, color = 'red', alpha=0.4)
    # set different colors to some bars (high outliers) in the chart
    # for inx in range(len(top_ft_inx_arr)):
    # bar_list[int(top_ft_inx_arr[inx])].set_color('b')
    for top_ft_inx in top_ft_inx_arr:
        bar_list[int(top_ft_inx)].set_color('b')

    # _ = plt.xlabel('Relative importance')
    # plt.bar(range(len(ft_importance_arr)), ft_importance_arr[indices], align="center")
#     plt.title(`len(ft_importance_arr)`+" ("+`len(top_ft_inx_arr)`+") instance features to "+`num_mapped_clusters`+" clusters/classes")
    plt.title(`len(ft_importance_arr)`+" ("+`len(top_ft_inx_arr)`+") instance features")
    plt.xlabel("Gini Importance")
    plt.ylabel("Features")

    if output_folder is None:
        output_folder = __output_folder__

    main_title = plot_title+'-'+dataset_name
    to_save_path = os.path.join(output_folder, main_title+'.pdf')
    plt.savefig(to_save_path, format='pdf', dpi=__fig_dpi__, bbox_inches='tight')
    if to_show == True:
        plt.show()

    plt.gcf().clear()
    
    
def plot_conf_importance_from_clsf(dataset_name, ft_importance_arr, top_ft_inx_arr, num_mapped_clusters, plot_title, output_folder=None, to_show = False):
    '''
        Plot a parameter importance bar chart
    '''
    y_pos = np.arange(len(ft_importance_arr))
    bar_list = plt.barh(y_pos, ft_importance_arr, color = 'red', alpha=0.4)
    # set different colors to some bars (high outliers) in the chart
    # for inx in range(len(top_ft_inx_arr)):
    # bar_list[int(top_ft_inx_arr[inx])].set_color('b')
    for top_ft_inx in top_ft_inx_arr:
        bar_list[int(top_ft_inx)].set_color('b')

    # _ = plt.xlabel('Relative importance')
    # plt.bar(range(len(ft_importance_arr)), ft_importance_arr[indices], align="center")
    plt.title(`len(ft_importance_arr)`+" ("+`len(top_ft_inx_arr)`+") configurations to "+`num_mapped_clusters`+" clusters/classes")
    plt.xlabel("Gini Importance")
    plt.ylabel("Parameters")

    plt.yticks(range(0, len(ft_importance_arr)+1))

    if output_folder is None:
        output_folder = __output_folder__

    main_title = plot_title+'-'+dataset_name
    to_save_path = os.path.join(output_folder, main_title+'.pdf')
    plt.savefig(to_save_path, format='pdf', dpi=__fig_dpi__, bbox_inches='tight')
    if to_show == True:
        plt.show()

    plt.gcf().clear()


def plot_2d_scatter_subset(dio, matrix, subset_list, plt_title, plt_annt_list = None, marker_size = 30, dim_reduction_type = DimReductionType.MDS, hide_axis_labels = True, output_folder=None, to_show = False):

    if len(matrix[0]) > 2:
        dim_matrix = apply_dim_reduction(matrix, 2, dim_reduction_type)
    else:
        dim_matrix = matrix

    labels_selected = np.zeros(shape=(len(matrix)))
    for inx in subset_list:
        labels_selected[inx] = 1
    labels = ['Not selected', 'Selected']

    plt.figure(1)

    plt.scatter(dim_matrix[:, 0], dim_matrix[:, 1], marker = 'o', s=(marker_size*((labels_selected*2)+1)), c = labels_selected.astype(np.float), label = labels, alpha=0.4)
    plt.title(plt_title)
    plt.axis('equal')
    if hide_axis_labels == True:
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])


    if plt_annt_list != None:
        for label, x, y in zip(plt_annt_list, dim_matrix[:, 0], dim_matrix[:, 1]):
            plt.annotate(
            label,
            xy = (x, y),
            xytext = (-10, 10),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            fontsize=7,
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    # plt.interactive(False)

    if output_folder is None:
        output_folder = __output_folder__

    main_title = plt_title+"-"+dio.dataset_name
    to_save_path = os.path.join(output_folder, main_title+'.pdf')
    plt.savefig(to_save_path, format='pdf', dpi=__fig_dpi__, bbox_inches='tight')
    if to_show == True:
        plt.show()

    plt.gcf().clear()


def plot_2d_scatter_subset_with_cluster(dataset_name, matrix, subset_list, cluster_labels, plt_title, plt_annt_list = None, marker_size = 30, dim_reduction_type = DimReductionType.MDS, hide_axis_labels = True, output_folder=None, to_show = False):
    '''
        Generate and save a scatter plot with clustering info and annotations
    '''
    if len(matrix[0]) > 2:
        dim_matrix = apply_dim_reduction(matrix, 2, dim_reduction_type)
    else:
        dim_matrix = matrix

    labels_selected = np.zeros(shape=(len(matrix)))
    for inx in subset_list:
        labels_selected[inx] = 1
    labels = ['Not selected', 'Selected']

    plt.figure(1)

    plt.scatter(dim_matrix[:, 0],
                dim_matrix[:, 1],
                marker = 'o',
                s=(marker_size*((labels_selected*3)+1)),
                c = cluster_labels,
                label = labels,
                alpha=0.4)
    plt.title(plt_title+" ("+str(len(np.unique(cluster_labels)))+" clusters)")
    plt.axis('equal')
    if hide_axis_labels == True:
        plt.gca().axes.get_xaxis().set_ticks([])
        plt.gca().axes.get_yaxis().set_ticks([])


    if plt_annt_list != None:
        for label, x, y in zip(plt_annt_list, dim_matrix[:, 0], dim_matrix[:, 1]):
            plt.annotate(
            label,
            xy = (x, y),
            xytext = (-10, 10),
            textcoords = 'offset points', ha = 'right', va = 'bottom',
            fontsize=2,
            bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.4),
            arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    # plt.interactive(False)

    if output_folder is None:
        output_folder = __output_folder__

    main_title = plt_title+"-"+dataset_name
    to_save_path = os.path.join(output_folder, main_title+'.pdf')
    plt.savefig(to_save_path, format='pdf', dpi=__fig_dpi__, bbox_inches='tight')
    if to_show == True:
        plt.show()

    plt.gcf().clear()


def apply_dim_reduction(norm_matrix, dim, type):
    dim_model = None
    dim_matrix = None
    if type == DimReductionType.MDS:
        dim_model = manifold.MDS(n_components=dim, dissimilarity="euclidean", random_state=__seed__)
    elif type == DimReductionType.Isomap:
        dim_model = manifold.Isomap(n_components=dim)
    elif type == DimReductionType.SpectralEmbedding:
        dim_model = manifold.SpectralEmbedding(n_components=dim)
    elif type == DimReductionType.PCA:
        dim_model = PCA(n_components=dim)
    elif type == DimReductionType.LDA: ##TODO: reconsider this
        dim_model = LDA(n_components=dim)
    elif type == DimReductionType.CMDS: ## classical MDS - TO change for dim reduction
        dist_matrix = euclidean_distances(norm_matrix)
        return cmdscale(dist_matrix)

    dim_matrix = dim_model.fit_transform(norm_matrix)
    # dim_matrix = dim_model.fit(norm_matrix).embedding_

    return dim_matrix


def apply_MDS(matrix, dim):

    # calculate similarity between rows of a given matrix
    inst_similarity = euclidean_distances(matrix)
    print(inst_similarity.shape)

    mds = manifold.MDS(n_components=dim, dissimilarity="precomputed", max_iter=100, n_init=1)
    mds_matrix = mds.fit_transform(inst_similarity)

    return mds_matrix

def apply_Isomap(matrix, dim):
    ismp = manifold.Isomap(n_components=dim)
    ismp_matrix = ismp.fit_transform(matrix)

    return ismp_matrix


def apply_MDS_with_ddissimilarity(matrix, plt_dim = 2, plt_dissimilarity="euclidean"):

    mds = manifold.MDS(n_components=plt_dim, dissimilarity=plt_dissimilarity, max_iter=100, n_init=1)
    mds_matrix = mds.fit_transform(matrix)

    return mds_matrix





# f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,5))
# data = np.random.random((10, 2))
# create_subplot_scatter(data, 'sc1_plot', ax1)
# data = np.random.random((10, 2))
# create_subplot_scatter(data, 'sc2_plot', ax2)
# plt.show()

# X = [[0, 1], [3, 6], [4, 5]]
# sim = euclidean_distances(X)
# print(sim)
