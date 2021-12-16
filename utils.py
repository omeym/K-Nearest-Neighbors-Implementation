import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY CODES ABOVE 
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)

    y_sum = 0
    y_pred_sum = 0
    avg = 0
    for y, y_pred in zip(real_labels, predicted_labels):
        avg = avg + y * y_pred
        y_sum = y_sum + y
        y_pred_sum = y_pred_sum + y_pred
    
    return 2 * (float(avg) / float(y_sum + y_pred_sum))


class Distances:
    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        p1 = np.asfarray(point1)
        p2 = np.asfarray(point2)

        mod_x = np.abs(np.subtract(p1,p2))
        mod_x = np.power(mod_x,3)

        mink_dist = np.cbrt(np.sum(mod_x))

        return mink_dist

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """

        p1 = np.asfarray(point1)
        p2 = np.asfarray(point2)
        
        mod_x = np.abs(np.subtract(p1,p2))
        mod_x = np.power(mod_x,2)

        eucl_dist = np.sqrt(np.sum(mod_x))

        return eucl_dist
        
    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        p1 = np.asfarray(point1)
        p2 = np.asfarray(point2)

        if((np.linalg.norm(p1) == 0) or (np.linalg.norm(p2) == 0)):
            return float(1)

        cosine_sim_dist = float(1.0 - ((np.dot(p1, p2) / (np.linalg.norm(p1) * np.linalg.norm(p2)))))

        return cosine_sim_dist



class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
   

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you need to try different distance functions you implemented in part 1.1 and different values of k (among 1, 3, 5, ... , 29), and find the best model with the highest f1-score on the given validation set.
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] training labels to train your KNN model
        :param x_val:  List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), and model (an instance of KNN) and assign them to self.best_k,
        self.best_distance_function, and self.best_model respectively.
        NOTE: self.best_scaler will be None.

        NOTE: When there is a tie, choose the model based on the following priorities:
        First check the distance function:  euclidean > Minkowski > cosine_dist 
		(this will also be the insertion order in "distance_funcs", to make things easier).
        For the same distance function, further break tie by prioritizing a smaller k.
        """
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None
        best_f1 = 0
        
        for func in distance_funcs:
            for k in range(1, min(31, len(x_train) + 1), 2):
                knnObj = KNN(k, distance_funcs[func])
                knnObj.train(x_train, y_train)
                predicted_labels = knnObj.predict(x_val)
                current_f1_score = f1_score(y_val, predicted_labels)
                
                if current_f1_score > best_f1:
                    self.best_k = k
                    self.best_distance_function = func
                    self.best_model = knnObj
                    best_f1 = current_f1_score

                    
       

  
    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is the same as "tuning_without_scaling", except that you also need to try two different scalers implemented in Part 1.3. More specifically, before passing the training and validation data to KNN model, apply the scalers in scaling_classes to both of them. 
		
        :param distance_funcs: dictionary of distance functions (key is the function name, value is the function) you need to try to calculate the distance. Make sure you loop over all distance functions for each k value.
        :param scaling_classes: dictionary of scalers (key is the scaler name, value is the scaler class) you need to try to normalize your data
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data
        :param y_val: List[int] validation labels

        Find the best k, distance_function (its name), scaler (its name), and model (an instance of KNN), and assign them to self.best_k, self.best_distance_function, best_scaler, and self.best_model respectively.
        
        NOTE: When there is a tie, choose the model based on the following priorities:
        First check scaler, prioritizing "min_max_scale" over "normalize" (which will also be the insertion order of scaling_classes). Then follow the same rule as in "tuning_without_scaling".
        """
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None
        best_f1 = 0
        
        for scaling_func in scaling_classes:
            
            CurrentScalarFunc = scaling_classes[scaling_func]()
            scaled_x_train = CurrentScalarFunc(x_train)
            scaled_x_val = CurrentScalarFunc(x_val)
            
            for dist_func in distance_funcs:
                for k in range(1, min(31, len(x_train) + 1), 2):
                    
                    knnObj = KNN(k, distance_funcs[dist_func])
                    knnObj.train(scaled_x_train, y_train)
                    y_val_pred = knnObj.predict(scaled_x_val)
                    current_f1_score = f1_score(y_val, y_val_pred)
                    
                    if current_f1_score > best_f1:
                        best_f1 = current_f1_score
                        self.best_k = k
                        self.best_distance_function = dist_func
                        self.best_scaler = scaling_func
                        self.best_model = knnObj
                        
                        

      
class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        
        features = np.array(features)
        feature_norms = np.linalg.norm(features, axis = 1)[:,np.newaxis]
        zero_indices = np.where(feature_norms==0)[0]
        feature_norms[zero_indices] = 1
        normalized_features = features/feature_norms

        return normalized_features


class MinMaxScaler:
    def __init__(self):
        pass

    # TODO: min-max normalize data
    def __call__(self, features):
        """
		For each feature, normalize it linearly so that its value is between 0 and 1 across all samples.
        For example, if the input features are [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]].
		This is because: take the first feature for example, which has values 2, -1, and 0 across the three samples.
		The minimum value of this feature is thus min=-1, while the maximum value is max=2.
		So the new feature value for each sample can be computed by: new_value = (old_value - min)/(max-min),
		leading to 1, 0, and 0.333333.
		If max happens to be same as min, set all new values to be zero for this feature.
		(For further reference, see https://en.wikipedia.org/wiki/Feature_scaling.)

        :param features: List[List[float]]
        :return: List[List[float]]
        """
      
        features = np.array(features, dtype = float)
        [r,c] = np.shape(features)

        for i in range(c):
            v = features[:, i]
            if(v.max() == v.min()):
                features[:,i] = 0
            else:
                features[:,i] = (v - v.min()) / (v.max() - v.min())


        return features