import numpy as np
import os
import pprint
import itertools
import sys
import operator


#from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_regression

from ml_util import *
from data_io import *
from _collections import defaultdict
from sklearn.cross_validation import ShuffleSplit
from sklearn.metrics.metrics import r2_score
from sklearn.feature_selection.rfe import RFE
from sklearn.linear_model.base import LinearRegression
from sklearn.linear_model.randomized_l1 import RandomizedLasso


__seed__ = 123456


class FeatureSelectionMethod:  
    VarianceThreshold, VarianceThresholdPercentile, SelectKBest, SelectPercentile, GiniPercentile, RegrGiniPercentile, RegrGiniPercentileMultiStep, RegrGiniPercentileMultiStepwithRedundancy, RegrMeanDecrAccuracy, RecursiveFtEliminationLR, RecursiveFtEliminationSVR, StabilitySelRLasso = range(0, 12)

class InstanceClusteringFTType:
    Descriptive, Latent, DescriptiveLatent, DescriptiveLatentExt, DescriptiveSubset, DescriptiveSubsetLatent, DescriptiveSubsetLatentExt = range(0, 7)


## TODO : change return for other feature selection methods
def choose_features(norm_ft_matrix, output_matrix, type, k_best_features=10, num_folds_g = 10, num_consc_wrs_steps_g = 5, ft_outlier_threshold_g = 95):
    '''
        Choose a subset of given features

        :param norm_ft_matrix: numpy 2D matrix - normalized feature matrix
        :param output_matrix: numpy 2D matrix - output matrix
        :param type: FeatureSelectionMethod - feature selection method
        :param k_best_features: int - number of features to be selected if applicable
        :param num_folds_g: int - number of folds to evaluate the quality of norm_ft_matrix -> output_matrix mapping (if applicable)
        :param num_consc_wrs_steps_g: number of consecutive worsening steps, used for clustering in GiniPercentile
        :param ft_outlier_threshold_g: percentile threshold level in [0, 100) used to determine number of features to be selected
        :return:
    '''
    num_features = len(norm_ft_matrix.T)

    # f_selector = None
    scores = None
    top_ft_inx_arr = None

#     if type == FeatureSelectionMethod.VarianceThreshold:
#         f_selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
#         f_selector.fit_transform(norm_ft_matrix)
#         scores = f_selector.variances_
# 
# 
#     elif type == FeatureSelectionMethod.VarianceThresholdPercentile:
#         f_selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
#         f_selector.fit_transform(norm_ft_matrix)
#         scores = f_selector.variances_
# 
#         scores, top_ft_inx_arr = percentile_based_outlier(scores, ft_outlier_threshold_g)
    ##elif
    if type == FeatureSelectionMethod.SelectKBest:
        f_selector = SelectKBest(f_regression, k=k_best_features)
        f_selector.fit_transform(norm_ft_matrix, output_matrix)

    elif type == FeatureSelectionMethod.SelectPercentile: ##TODO - correct this (Nan values for single output ??)
        k_perc = k_best_features * 100 / float(num_features)
        f_selector = SelectPercentile(f_regression, percentile=k_perc)
        f_selector.fit_transform(norm_ft_matrix, output_matrix[:, 0])
        scores = -np.log10(f_selector.pvalues_)
        scores /= scores.max()

    elif type == FeatureSelectionMethod.GiniPercentile:
        centroids, labels, num_clusters = gen_class_arr(output_matrix, ClusteringMethods.KMeansSLHadaptive, num_consc_wrs_steps = num_consc_wrs_steps_g, k_max=10) ## k_max=len(norm_ft_matrix)/4
        clsf = gen_clsf_model(norm_ft_matrix, labels, ClassificationMethods.RFC)
        # TODO: add cv scores to the log file to check the quality of the feature selection model
        # cv_scores = apply_CV(clsf, norm_ft_matrix, labels, num_folds_g)
        # print("CV scores: ", cv_scores)
        ft_importance_arr = get_ft_importance(clsf, FeatureImportanceMethods.Gini)
        scores, top_ft_inx_arr = percentile_based_outlier(ft_importance_arr, ft_outlier_threshold_g)
        sorted_top_ft_inx_arr = np.sort(top_ft_inx_arr)
        print("top_ft_inx_arr: ", top_ft_inx_arr)

        return scores, top_ft_inx_arr, ft_importance_arr, num_clusters
    
    elif type == FeatureSelectionMethod.RegrGiniPercentile: ##http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
        rf = RandomForestRegressor(random_state=__seed__)
        rf.fit(norm_ft_matrix, output_matrix)
        ft_importance_arr = rf.feature_importances_
        
        scores, top_ft_inx_arr = percentile_based_outlier(ft_importance_arr, ft_outlier_threshold_g)
        sorted_top_ft_inx_arr = np.sort(top_ft_inx_arr)
        print("top_ft_inx_arr: ", top_ft_inx_arr)
        
        print "ft_imp_total = ", np.sum(ft_importance_arr)
        
        cv_scrs = cross_validation.cross_val_score(RandomForestRegressor(random_state=__seed__), norm_ft_matrix, output_matrix, cv=num_folds)
        print "ft cv scores: ", cv_scrs
        
#         cv_scrs_after_fs = cross_validation.cross_val_score(RandomForestRegressor(random_state=__seed__), norm_ft_matrix[:, top_ft_inx_arr], output_matrix, cv=num_folds)
#         print "ft cv scores after feature selection: ", cv_scrs
        
        
        return scores, top_ft_inx_arr, ft_importance_arr, -1
 
 
    elif type == FeatureSelectionMethod.RegrGiniPercentileMultiStep: ##http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
        
        overall_ft_importance_arr = None
        
        for iter in range(1,10):
            rf = RandomForestRegressor(random_state= (__seed__ * iter))
            rf.fit(norm_ft_matrix, output_matrix)
            ft_importance_arr = rf.feature_importances_
            
            if overall_ft_importance_arr is None:
                overall_ft_importance_arr = ft_importance_arr
            else:
                overall_ft_importance_arr = np.add(overall_ft_importance_arr, ft_importance_arr)
                
        ft_importance_arr = overall_ft_importance_arr / float(10)       
        
        scores, top_ft_inx_arr = percentile_based_outlier(ft_importance_arr, ft_outlier_threshold_g)
        sorted_top_ft_inx_arr = np.sort(top_ft_inx_arr)
        print("top_ft_inx_arr: ", top_ft_inx_arr)
        
        print "ft_imp_total = ", np.sum(ft_importance_arr)
        
        cv_scrs = cross_validation.cross_val_score(RandomForestRegressor(random_state=__seed__), norm_ft_matrix, output_matrix, cv=num_folds)
        print "ft cv scores: ", cv_scrs
        
#         cv_scrs_after_fs = cross_validation.cross_val_score(RandomForestRegressor(random_state=__seed__), norm_ft_matrix[:, top_ft_inx_arr], output_matrix, cv=num_folds)
#         print "ft cv scores after feature selection: ", cv_scrs
        
        
        return scores, top_ft_inx_arr, ft_importance_arr, -1
 
 
    elif type == FeatureSelectionMethod.RegrGiniPercentileMultiStepwithRedundancy: ##http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
        
        overall_ft_importance_arr = None
        
        for iter in range(1,10):
            rf = RandomForestRegressor(random_state= (__seed__ * iter))
            rf.fit(norm_ft_matrix, output_matrix)
            ft_importance_arr = rf.feature_importances_
            
            if overall_ft_importance_arr is None:
                overall_ft_importance_arr = ft_importance_arr
            else:
                overall_ft_importance_arr = np.add(overall_ft_importance_arr, ft_importance_arr)
                
        ft_importance_arr = overall_ft_importance_arr / float(10)       
        
        scores, top_ft_inx_arr = percentile_based_outlier(ft_importance_arr, ft_outlier_threshold_g)
        
        ### generate linear correlation coefficient matrix
        sel_norm_ft_matrix = norm_ft_matrix[top_ft_inx_arr]
        num_sel_fts = len(top_ft_inx_arr)
        ###correlation_coeff_matrix = np.zeros((num_sel_fts, num_sel_fts))
        correlation_coff_dict = {}
        for ft_x in range(num_sel_fts):
            ft_x_inx = top_ft_inx_arr[ft_x]
            for ft_y in range(ft_x+1, num_sel_fts):
                ft_y_inx = top_ft_inx_arr[ft_y] 
                
                ft_x_avg = np.average(sel_norm_ft_matrix.T[ft_x_inx])
                ft_y_avg = np.average(sel_norm_ft_matrix.T[ft_y_inx])

                nom = 0
                denom_x = 0
                denom_y = 0
                
                num_insts = len(sel_norm_ft_matrix)
                for inst_inx in range(num_insts):
                    part_x = sel_norm_ft_matrix[inst_inx][ft_x_inx] - ft_x_avg
                    part_y = sel_norm_ft_matrix[inst_inx][ft_y_inx] - ft_y_avg
                
                    nom += part_x * part_y
                    denom_x += np.square(part_x)
                    denom_y += np.square(part_y)


                ## keep all as positive values - for simplicity
                ## -1,+1: completely correlated ++++  0: independent
                ##correlation_coeff_matrix[ft_x][ft_y] = correlation_coeff_matrix[ft_y][ft_x] = nom / ( np.sqrt(denom_x) * np.sqrt(denom_y) )
#                 correlation_coff_dict[str(ft_x_inx) + "_" + str(ft_y_inx)] = nom / ( np.sqrt(denom_x) * np.sqrt(denom_y) )
                correlation_coff_dict[str(ft_x_inx) + "_" + str(ft_y_inx)] = np.absolute(  nom / ( np.sqrt(denom_x) * np.sqrt(denom_y) )  )
                    
        ## sort the correlation coefficient dictionary by their values
        sorted_correlation_coff_list = sorted(correlation_coff_dict.items(), key=operator.itemgetter(1))
        
        print " RegrGiniPercentileMultiStepwithRedundancy >>>> BEFORE: ", scores, top_ft_inx_arr
        
        removed_ft_list = []
        
        ## http://stackoverflow.com/questions/529424/traverse-a-list-in-reverse-order-in-python
        for list_entry in reversed(sorted_correlation_coff_list):
            if list_entry[1] >= 0.95:
                ft_x_inx = int(list_entry[0][: list_entry[0].index("_")])
                ft_y_inx = int(list_entry[0][list_entry[0].index("_")+1:])
                
                ## do not check features which are already removed
                if ft_x_inx in removed_ft_list or ft_y_inx in removed_ft_list:
                    continue
                
                f_x = np.where(top_ft_inx_arr == ft_x_inx)[0]
                f_y = np.where(top_ft_inx_arr == ft_y_inx)[0]
                
                if scores[ f_x ] >= scores[ f_y ]:
                    top_ft_inx_arr = np.delete(top_ft_inx_arr, f_y, 0)
                    scores = np.delete(scores, f_y, 0)
                    removed_ft_list.append(ft_y_inx)
                    ##print "feature ", ft_y_inx, " is removed !"
                else:
                    top_ft_inx_arr = np.delete(top_ft_inx_arr, f_x, 0)
                    scores = np.delete(scores, f_x, 0)
                    removed_ft_list.append(ft_x_inx)
                    ##print "feature ", ft_x_inx, " is removed !"
                
                
        print " removed_ft_list: ", removed_ft_list
        print " RegrGiniPercentileMultiStepwithRedundancy >>>> AFTER: ", scores, top_ft_inx_arr        
                
                
        
        sorted_top_ft_inx_arr = np.sort(top_ft_inx_arr)
        print("top_ft_inx_arr: ", top_ft_inx_arr)
        
        print "ft_imp_total = ", np.sum(ft_importance_arr)
        
        cv_scrs = cross_validation.cross_val_score(RandomForestRegressor(random_state=__seed__), norm_ft_matrix, output_matrix, cv=num_folds)
        print "ft cv scores: ", cv_scrs
        
#         cv_scrs_after_fs = cross_validation.cross_val_score(RandomForestRegressor(random_state=__seed__), norm_ft_matrix[:, top_ft_inx_arr], output_matrix, cv=num_folds)
#         print "ft cv scores after feature selection: ", cv_scrs
        
        
        return scores, top_ft_inx_arr, ft_importance_arr, -1
    
        
    
    elif type == FeatureSelectionMethod.RegrMeanDecrAccuracy: ##http://blog.datadive.net/selecting-good-features-part-iii-random-forests/
        rf = RandomForestRegressor(random_state=__seed__)
        scores_dict = defaultdict(list)

        for train_idx, test_idx in ShuffleSplit(len(norm_ft_matrix), test_size = 0.1): ## default: test_size = 0.1
            X_train, X_test = norm_ft_matrix[train_idx], norm_ft_matrix[test_idx]
            Y_train, Y_test = output_matrix[train_idx], output_matrix[test_idx]
            r = rf.fit(X_train, Y_train)
            acc = r2_score(Y_test, rf.predict(X_test))
            for i in range(norm_ft_matrix.shape[1]):
                X_t = X_test.copy()
                np.random.shuffle(X_t[:, i])
                shuff_acc = r2_score(Y_test, rf.predict(X_t))
                scores_dict[i].append((acc-shuff_acc)/acc)
                
        print "Features sorted by their score:"
        print sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores_dict.items()], reverse=True)
        
        score_sorted_ft_list = sorted([(round(np.mean(score), 4), feat) for feat, score in scores_dict.items()], reverse=True)
        
        ft_importance_arr = np.zeros(len(norm_ft_matrix.T))
        for score_key_arr in score_sorted_ft_list:
            print "key = ", score_key_arr[1], " -- ft_importance_arr[key]", score_key_arr[0]
            ft_importance_arr[score_key_arr[1]] = score_key_arr[0]
        
        scores, top_ft_inx_arr = percentile_based_outlier(ft_importance_arr, ft_outlier_threshold_g)
        sorted_top_ft_inx_arr = np.sort(top_ft_inx_arr)
        print("top_ft_inx_arr: ", top_ft_inx_arr)
         
        print "ft_imp_total = ", np.sum(ft_importance_arr)

        return scores, top_ft_inx_arr, ft_importance_arr, -1
    
    
    elif type == FeatureSelectionMethod.RecursiveFtEliminationLR:
        rf = LinearRegression()
        rfe = RFE(rf, n_features_to_select=1)
        rfe.fit(norm_ft_matrix, output_matrix)
        
#         print "Features sorted by their rank:"
#         print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
        print "RFE LR feature ranking: ", rfe.ranking_
        #print "RFE LR feature support: ", rfe.support_
        
    elif type == FeatureSelectionMethod.RecursiveFtEliminationSVR:
        rf = SVR()
        rfe = RFE(rf, n_features_to_select=1)
        rfe.fit(norm_ft_matrix, output_matrix)
        
#         print "Features sorted by their rank:"
#         print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))
        print "RFE LR feature ranking: ", rfe.ranking_
        #print "RFE LR feature support: ", rfe.support_
            
    elif type == FeatureSelectionMethod.StabilitySelRLasso:
        rlasso = RandomizedLasso(random_state=__seed__)
        rlasso.fit(norm_ft_matrix, output_matrix[:,0])  ##TODO: NO MULTIVARIATE REGRESSION !!!! 
        
        print  "RLasso scores: ", rlasso.scores_

        ft_importance_arr = rlasso.scores_
        scores, top_ft_inx_arr = percentile_based_outlier(ft_importance_arr, ft_outlier_threshold_g)
        sorted_top_ft_inx_arr = np.sort(top_ft_inx_arr)
        print("top_ft_inx_arr: ", top_ft_inx_arr)
        
        return scores, top_ft_inx_arr, ft_importance_arr, -1
         
    return scores, top_ft_inx_arr, None, None



def ft_selection_post_process(ft_cost_matrix, ft_step_membership_matrix, selected_fts, selected_fts_imp, threshold_level = 0.8, threshold_step_level = 0.9):
    '''
        Post processing for features
    '''
    ft_cost_imp_dict = calc_total_ft_cost(ft_cost_matrix, ft_step_membership_matrix, selected_fts, selected_fts_imp)
    
    ###sel_ft_step_membership_matrix = ft_step_membership_matrix[selected_fts, :] ##TODO: dont do this twice !!

    
    ### sort feature sets wrt their costs / time to calculation
    sorted_ft_step_costs = np.argsort(ft_cost_imp_dict['ft_step_cost'])
    sorted_ft_step_costs = sorted_ft_step_costs[::-1]
    
    last_ft_total_cost = ft_cost_imp_dict['ft_total_cost']
    last_ft_total_imp = ft_cost_imp_dict['ft_total_imp']
    
    ## determine max outliers
    min_outlier_arr, min_outlier_inx_arr = percentile_based_min_outlier(ft_cost_imp_dict['ft_step_cost'], 0.0)
    
    print "min_outlier_arr: ", min_outlier_arr, " -- min_outlier_inx_arr: ", min_outlier_inx_arr
    min_outlier_arr
    
    new_selected_fts = curr_selected_fts = selected_fts
    new_selected_fts_imp = curr_selected_fts_imp = selected_fts_imp
    
    for ft_step_inx in sorted_ft_step_costs: ## iterate feature steps starting from worst
        
#         if ft_step_inx not in min_outlier_inx_arr:
#             continue
        
        ## determine the features to be removed
        tobe_removed_fts = np.where(ft_step_membership_matrix.T[ft_step_inx] == 1)[0]
        tobe_remoted_fts_inx = []
        ft_inx = 0
        for sel_ft in curr_selected_fts:
            if sel_ft in tobe_removed_fts:
                tobe_remoted_fts_inx.append(ft_inx)
            ft_inx += 1
    
        new_selected_fts = np.delete(curr_selected_fts, tobe_remoted_fts_inx)
        new_selected_fts_imp = np.delete(curr_selected_fts_imp, tobe_remoted_fts_inx)
        
        ## re-calculate total feature importance
        ft_cost_imp_dict = calc_total_ft_cost(ft_cost_matrix, ft_step_membership_matrix, new_selected_fts, new_selected_fts_imp)
        
        if ft_cost_imp_dict['ft_total_imp'] >= threshold_level: ## and ft_cost_imp_dict['ft_total_imp'] >= threshold_step_level*last_ft_total_imp:
            
            print "ft_step_inx:",ft_step_inx," features removed: ", tobe_removed_fts, " - cost gain: ", (last_ft_total_cost - ft_cost_imp_dict['ft_total_cost']), " - new ft imp: ", ft_cost_imp_dict['ft_total_imp']

            last_ft_total_cost = ft_cost_imp_dict['ft_total_cost']
            last_ft_total_imp = ft_cost_imp_dict['ft_total_imp']
            curr_selected_fts = new_selected_fts
            curr_selected_fts_imp = new_selected_fts_imp
            

    print "curr_selected_fts: ", curr_selected_fts

    return ft_cost_imp_dict, curr_selected_fts, curr_selected_fts_imp


