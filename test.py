from data import data_processing
from utils import Distances, HyperparameterTuner, NormalizationScaler, MinMaxScaler
from knn import *

def main():
    distance_funcs = {
        'euclidean': Distances.euclidean_distance,
        'minkowski': Distances.minkowski_distance,
        'cosine_dist': Distances.cosine_similarity_distance,
    }

    scaling_classes = {
        'min_max_scale': MinMaxScaler,
        'normalize': NormalizationScaler,
    }

    x_train, y_train, x_val, y_val, x_test, y_test = data_processing()

    print('x_train shape = ', x_train.shape)
    print('y_train shape = ', y_train.shape)

    knn_obj = KNN(5,distance_funcs['euclidean'])

    knn_obj.train(x_train, y_train)

    nn = knn_obj.get_k_neighbors(x_test[0:2])
    labels = knn_obj.predict(x_test[0:16])
    
    #tuner_without_scaling_obj = HyperparameterTuner()
    #tuner_without_scaling_obj.tuning_without_scaling(distance_funcs, x_train, y_train, x_val, y_val)

    #print("**Without Scaling**")
    #print("k =", tuner_without_scaling_obj.best_k)
    #print("distance function =", tuner_without_scaling_obj.best_distance_function)

    tuner_with_scaling_obj = HyperparameterTuner()
    tuner_with_scaling_obj.tuning_with_scaling(distance_funcs, scaling_classes, x_train, y_train, x_val, y_val)

    print("\n**With Scaling**")
    print("k =", tuner_with_scaling_obj.best_k)
    print("distance function =", tuner_with_scaling_obj.best_distance_function)
    print("scaler =", tuner_with_scaling_obj.best_scaler)


if __name__ == '__main__':
    main()


