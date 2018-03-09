from __future__ import print_function, division
import numpy as np
import pandas as pd
import math as m
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from math import pow


def safe_div(x,y):
    if y == 0:
        return 0
    else:
        return x/y

def preprocessing(x):
	"""
	tahap pra pemrosesan, normalisasi data

	args: 
		x: n-dimentional vector [a..b]
	returns:
		x: normalized x n-dimentional vector
	
	x_final = (d-d_min) / (d_max-d_min)
	"""
	dmax = max(x)
	dmin = min(x)
	x_normalized = [(x_-dmin)/(dmax-dmin) for x_ in x]
	return x_normalized

def kmeans(x,k,init_center=[]):

    if init_center == [] :
        init_center = x[:k]
    center = {}
    center_isnot_equal = True
    while center_isnot_equal:
        all_distance = []
        # print("center sekarang:",init_center)
        for c in init_center:
            distance = []
            for x_i in x:
                dist = m.sqrt((x_i-c)**2) # using euclidean distance
                distance.append(dist)
            all_distance.append(distance)
        tr_dist = np.array(all_distance).T # distance of all data to each initial center of cluster
        cluster = {}
        cluster = [[] for ic in range(k)] # init array of each cluster
        count = 0
        
        for i in tr_dist:
            cluster[i.argmin()].append(x[count])
            count += 1
        # print(cluster)
        
        for i in range(k):
            if len(cluster[i]) == 0:
                center[i] = 0
            else:
                center[i] = np.sum(cluster[i]) / len(cluster[i])
        # print(center)

        new_center = [center[ic] for ic in center]
        # print(new_center)
        is_equal = np.array_equal(new_center, init_center)
        if(is_equal):
            center_isnot_equal = False
            # print("center sudah sama!")
        else:
            init_center = new_center
            # print("center belum sama!")
    dmax = [np.max(data_dist) for data_dist in all_distance]
    prep_data = {'center':new_center, 'max_distance':dmax}
    return prep_data

def kmeans_new(x,k,init_center=[]):
    if init_center == [] :
        init_center = x[:k]
    kms = KMeans(n_clusters=k,init=np.array(init_center).reshape(-1,1))
    x = np.array(x).reshape(-1,1)
    kms.fit(x)
    center = kms.cluster_centers_
    dist = pd.DataFrame(euclidean_distances(x, center), columns=[0,1,2])
    max_dist = [dist[i].max() for i in range(k)]
    prep_data = {'center': center, 'max_distance': max_dist}
    return prep_data

def train(x, y, n_hidden = 5, input_node=5, ephocs=3000, lr=0.01, tanggal=[]):
		#iteration start
		#df = pd.read_csv("data/data-latih.csv", ';',thousands='.',decimal=',')
		#columns = ["PROUCTION", "PANEN"]
		X_raw = x
		y_raw = y
		x = preprocessing(x)
		y = preprocessing(y)
		weight_iter_log = []
		center_log = []
		output_log = []
		error_log = []
		weight = np.array([0 for i in range(n_hidden+1)])
		wbaru = np.array([0 for i in range(n_hidden+1)])
		center = []
		init_weight_out = np.array([0 for i in range(n_hidden+1)])
		#input_node = 5
		iterlagi = True
		count = 1
		while iterlagi:
		#for i in range(ephocs):
				start = 0
				end_d = start + input_node
				out_i = []
				err_i = []
				weight_log = []
				dt = y[start+input_node:] # data dipakai adalah 6-10
				for y_i in dt: # perhitungan diterapkan ke seluruh data train, dari n sampai akhir data (900 data)
						## phase 1: input layer to hidden layer with k-means
						# k-means function already iterating behind, only need dmax and final center, saved to center_dist
						x_input = np.array(x[start:end_d]) # x untuk input node (5 data sebelum n), menghitung output n
						#print("input node: data-",start," sampai data-",end_d)
						center_dist = kmeans(x_input, n_hidden, center) # return [dmax, center]
						# parsing center_distance to each variable
						#center_ = [i[1] for i in center_dist]
						center_ = center_dist['center']
						center = center_ # center value copied to global center
						center_log.append(center_) 
						# debugging purpose, log all final center for every iteration. turned out useless, cause it's all same
						# well, it's already FINAL. why would you iterate it again? doesn't it just need to go directly to phase 2 after first iter?
						distance = center_dist['max_distance']
						betas = [safe_div(distance[i],m.sqrt(center_[i])) for i in range(len(center_))] # standard deviation for each center
						# at this point, we have the center of rbf function and the radius (the beta or mu, or whatever)
						## phase 2: hidden layer to output layer with gaussian function
						gaussfunc = []
						for c in range(len(center_)):
								alldist = []
								for xi in x_input:
										dist = pow(xi - center_[c], 2)
										alldist.append(dist)
								#gaussini = np.exp(safe_div(np.sum(alldist),(2 * (betas[c]**2))))
								gaussini = np.exp(safe_div(np.sum(alldist),pow(betas[c],2)))
								gaussfunc.append(gaussini)

						gauss_biased = gaussfunc
						#if weight[n_hidden] != 0:
						#	gauss_biased.append(weight[n_hidden]) # tambah nilai bias ke gaussfunc, tambah 1 untuk bias
						#else:
						gauss_biased.append(1)
						
						out = np.dot(gauss_biased, weight)
						out_i.append(out)
						
						## phase 3: calculate error and weight update
						err = y_i - out # error, data target - output
						err_i.append(err)
						deltaw = np.dot(err, gauss_biased)
						wbaru = weight + lr * deltaw
						start += 1
						end_d += 1
				if np.allclose(weight,wbaru, rtol=1e-04, atol=1e-06) or (count == ephocs):
						#print("weight sudah sama")
						#print("iterasi ke-",count)
						iterlagi = False
				else:
						weight = wbaru
						#init_weight_out = wbaru
						weight_log.append(wbaru) # debugging purpose, log all weight for all x

				weight_iter_log.append(weight_log)
				output_log.append(out_i)
				error_log.append(err_i)
				count += 1
				#start += 1
				#end_d += 1
		#endtime = time() - stime
		#print("Waktu dibutuhkan:",endtime)
		# error percentage with MAPE
		sig_error = 0
		for t in range(len(out_i)):
			sig_error += safe_div( (y[t] - out_i[t]),y[t] )
		mape = (100/len(out_i)) * sig_error
		prep_data = {
			'weight': weight.tolist(), # weight terakhir
			'error_log' : err_i, # error terakhir
			'error' : mape,
			'output' : out_i, #output terakhir
			'X_raw' : X_raw[input_node:].tolist(),
			'y_raw' : y_raw[input_node:].tolist(),
			'X_norm': x[input_node:],
			'y_norm': y[input_node:],
			'tanggal':tanggal[input_node:].tolist(),
			'center' : center,
			'iteration' : count,
			'hidden_node' : n_hidden,
			'input_node' : input_node,
			'l_rate' : lr,
			'max_epochs' : ephocs
		}
		return prep_data

def test(x, y, weight, center, n_hidden = 3, input_node=5, ephocs=3000, lr=0.01, tanggal=[]):
		X_raw = x
		y_raw = y
		x = preprocessing(x)
		y = preprocessing(y)
		weight_iter_log = []
		center_log = []
		output_log = []
		error_log = []
		wbaru = np.array([0 for i in range(n_hidden+1)])
		init_weight_out = np.array([0 for i in range(n_hidden+1)])
		#input_node = 5
		iterlagi = True
		count = 1
		while iterlagi:
		#for i in range(ephocs):
				start = 0
				end_d = start + input_node
				out_i = []
				err_i = []
				weight_log = []
				dt = y[start+input_node:] 
				for y_i in dt: # perhitungan diterapkan ke seluruh data train, dari n sampai akhir data
						## phase 1: input layer to hidden layer with k-means
						# k-means function already iterating behind, only need dmax and final center, saved to center_dist
						x_input = np.array(x[start:end_d]) # x untuk input node (5 data sebelum n), menghitung output n
						#print("input node: data-",start," sampai data-",end_d)
						center_dist = kmeans(x_input, n_hidden, center) # return [dmax, center]
						# parsing center_distance to each variable
						#center_ = [i[1] for i in center_dist]
						center_ = center_dist['center']
						center = center_ # center value copied to global center
						center_log.append(center_) 
						# debugging purpose, log all final center for every iteration. turned out useless, cause it's all same
						# well, it's already FINAL. why would you iterate it again? doesn't it just need to go directly to phase 2 after first iter?
						distance = center_dist['max_distance']
						betas = [safe_div(distance[i],m.sqrt(center_[i])) for i in range(len(center_))] # standard deviation for each center
						# at this point, we have the center of rbf function and the radius (the beta or mu, or whatever)
						## phase 2: hidden layer to output layer with gaussian function
						gaussfunc = []
						for c in range(len(center_)):
								alldist = []
								for xi in x_input:
										dist = pow(xi - center_[c], 2)
										alldist.append(dist)
								#gaussini = np.exp(safe_div(np.sum(alldist),(2 * (betas[c]**2))))
								gaussini = np.exp(safe_div(np.sum(alldist),pow(betas[c],2)))
								gaussfunc.append(gaussini)

						gauss_biased = gaussfunc
						#if weight[n_hidden] != 0:
						#	gauss_biased.append(weight[n_hidden]) # tambah nilai bias ke gaussfunc, tambah 1 untuk bias
						#else:
						gauss_biased.append(1)
						
						out = np.dot(gauss_biased, weight)
						out_i.append(out)
						
						## phase 3: calculate error and weight update
						err = y_i - out # error, data target - output
						err_i.append(err)
						deltaw = np.dot(err, gauss_biased)
						wbaru = weight + lr * deltaw
						start += 1
						end_d += 1
				if np.allclose(weight,wbaru, rtol=1e-04, atol=1e-06) or (count == ephocs):
						#print("weight sudah sama")
						#print("iterasi ke-",count)
						iterlagi = False
				else:
						weight = wbaru
						#init_weight_out = wbaru
						weight_log.append(wbaru) # debugging purpose, log all weight for all x

				weight_iter_log.append(weight_log)
				output_log.append(out_i)
				error_log.append(err_i)
				count += 1
				#start += 1
				#end_d += 1
		#endtime = time() - stime
		#print("Waktu dibutuhkan:",endtime)
		# denormalization
		denorm = []
		for t in out_i:
			#denorm.append(t * (np.max(dt) - np.min(dt)) + np.min(dt))
			denorm.append(t * (np.max(y_raw) - np.min(y_raw)) + np.min(y_raw))
		# error percentage with MAPE
		sig_error = 0
		for t in range(len(out_i)):
			sig_error += safe_div( (dt[t] - out_i[t]),y[t] )
			#sig_error += safe_div( (y[t] - out_i[t]),y[t] )
		mape = (100/len(out_i)) * sig_error
		prep_data = {
			'weight': wbaru.tolist(), # weight terakhir
			'error_log' : err_i, # error terakhir
			'error' : mape,
			'output' : out_i, #output terakhir
			'X_raw' : X_raw[input_node:].tolist(),
			'y_raw' : y_raw[input_node:].tolist(),
			'X_norm': x[input_node:],
			'y_norm': y[input_node:],
			'tanggal':tanggal[input_node:].tolist(),
			'center' : center,
			'iteration' : count,
			'hidden_node' : n_hidden,
			'input_node' : input_node,
			'l_rate' : lr,
			'max_epochs' : ephocs,
			'denorm' : denorm
		}
		return prep_data

def predict(X, y, weight, center):
		X_raw = X
		y_raw = y
		X = preprocessing(X)
		y = preprocessing(y)
		n_hidden = len(center)
		center_dist = kmeans(X, n_hidden, center)
		center_ = center_dist['center']
		distance = center_dist['max_distance']
		betas = [safe_div(distance[i],m.sqrt(center_[i])) for i in range(len(center_))]
		gaussfunc = []
		for c in range(len(center_)):
				alldist = []
				for xi in X:
						dist = pow(xi - center_[c], 2)
						alldist.append(dist)
				gaussini = np.exp(safe_div(np.sum(alldist),pow(betas[c],2)))
				gaussfunc.append(gaussini)
		gauss_biased = gaussfunc
		gauss_biased.append(1)
		out_i = np.dot(gauss_biased, weight)
		denorm = (out_i * (np.max(y_raw) - np.min(y_raw)) + np.min(y_raw))
		return denorm