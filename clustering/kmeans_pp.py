import numpy as np

## K-MEANS++
def euclid_dist(t1, t2):
    return np.sqrt(((t1-t2)**2).sum(axis = 1))
def calc_centroids(data, centroids):
    dist = np.zeros([data.shape[0], centroids.shape[0]])
    for idx, centroid in enumerate(centroids): 
        dist[:, idx] = euclid_dist(centroid, data)
    return np.array(dist)
def closest_centroids(data, centroids): 
    dist = calc_centroids(data, centroids) 
    return np.argmin(dist, axis = 1)
def move_centroids(data, closest, centroids):
    k = centroids.shape[0]
    new_centroids = np.array([data[closest == c].mean(axis = 0) for c in np.unique(closest)])
    if k - new_centroids.shape[0] > 0:
       print("adding {} centroid(s)".format(k - new_centroids.shape[0]))
       additional_centroids = data[np.random.randint(0, data.shape[0], k - new_centroids.shape[0])] 
       new_centroids = np.append(new_centroids, additional_centroids, axis = 0)
    return new_centroids
    
def init_centroids(data, num_clust):
    centroids = np.zeros([num_clust, data.shape[1]]) 
    centroids[0,:] = data[np.random.randint(0, data.shape[0], 1)]
    for i in range(1, num_clust):
        D2 = np.min([np.linalg.norm(data - c, axis = 1)**2 for c in centroids[0:i, :]], axis = 0) 
        probs = D2/D2.sum()
        cumprobs = probs.cumsum()
        ind = np.where(cumprobs >= np.random.random())[0][0]
        centroids[i, :] = np.expand_dims(data[ind], axis = 0)
    return centroids
    
def k_means(data, num_clust, num_iter): 
    centroids = init_centroids(data, num_clust)
    last_centroids = centroids
    for n in range(num_iter):
        closest = closest_centroids(data, centroids)
        centroids = move_centroids(data, closest, centroids)
        if not np.any(last_centroids != centroids):
            print("Early finish!")
            break
        last_centroids = centroids
    return centroids