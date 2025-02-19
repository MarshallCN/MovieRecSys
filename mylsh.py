import numpy as np
import pandas as pd
import turicreate
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import pairwise_distances
import time
from copy import copy
import matplotlib.pyplot as plt
from collections import defaultdict 
import random
from tqdm import tqdm

'''compute norm of a sparse vector'''
def norm(x):
    sum_sq=x.dot(x.T)
    norm=np.sqrt(sum_sq)
    return(norm)

def cossim(u,v):
    norm = np.linalg.norm(u)*np.linalg.norm(v) # 向量范数之积 所有元素的平方和
    cosine = u@v/norm  #矩阵乘法
#     ang = np.arccos(cosine)
    return cosine

def sframe_to_scipy(tsf_u):
    tsf_u_stack = pd.DataFrame(tsf_u.stack()).reset_index()
    tsf_u_stack.columns = ['uid','col_id','value']
    
    
    # Map feature words to integers 
#     mapping = {attr:i for i, attr in enumerate(extracted_features.columns)}
    
    # Create numpy arrays that contain the data for the sparse matrix.
    row_id = np.array(tsf_u_stack['uid']) 
    col_id = np.array(tsf_u_stack['col_id']) # unique word id
    data = np.array(tsf_u_stack['value'])  # tfidf v

    width, height = tsf_u_stack['uid'].max()+1, tsf_u_stack['col_id'].max()+1
    
    
    # Create a sparse matrix.
    corpus = csr_matrix((data, (row_id, col_id)), shape=(width, height))
    
    return corpus

def generate_random_vectors(dim, n_vectors):
    return np.random.randn(dim, n_vectors)

def compare_bits(model, id_1, id_2):
    bits1 = model['bin_indices_bits'][id_1]
    bits2 = model['bin_indices_bits'][id_2]
    print('Number of agreed bits: ', np.sum(bits1 == bits2))
    return np.sum(bits1 == bits2)

def train_lsh(data, n_vectors, seed=0):    
    if seed is not None:
        random.seed(seed) 
        np.random.seed(seed)
        
    dim = data.shape[1]
    random_vectors = generate_random_vectors(dim, n_vectors)  

    # Partition data points into bins,
    bin_indices_bits = data.dot(random_vectors) >= 0
    # and encode bin index bits into integers
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
    bin_indices = bin_indices_bits.dot(powers_of_two)

    # Update `table` so that `table[i]` is the list of document ids with bin index equal to i
    table = defaultdict(list)
    for idx, bin_index in enumerate(bin_indices):
        #if bin_index not in table:
            #table[bin_index] = []
            
        # Fetch the list of document ids associated with the bin and add the document id to the end.
        table[bin_index].append(idx) # YOUR CODE HERE
  
    # Note that we're storing the bin_indices here
    # so we can do some ad-hoc checking with it,
    # this isn't actually required
    model = {'data': data, # 原来的sparse matrix
             'table': table, # bin:[id] dict hash table
             'random_vectors': random_vectors,  
             'bin_indices': bin_indices, # 10进制的binary hash code
             'bin_indices_bits': bin_indices_bits} #binary hash code
    return model

# inspect bins
def get_similarity_items(corpus, item_id, topn=5):
    """
    Get the top similar items for a given item id.
    The similarity measure here is based on cosine distance.
    """
    query = corpus[item_id]
    scores = corpus.dot(query.T).toarray().ravel()
    best = np.argpartition(scores, -topn)[-topn:]
    similar_items = sorted(zip(best, scores[best]), key=lambda x: -x[1])
    similar_item_ids = [similar_item for similar_item, _ in similar_items]

    return similar_item_ids

# Query the LSH model
from itertools import combinations

def search_nearby_bins(query_bin_bits, table, search_radius=2, initial_candidates=set()):
    """
    For a given query vector and trained LSH model, return all candidate neighbors for
    the query among all bins within the given search radius.
    
    Example usage
    -------------
    >>> model = train_lsh(corpus, num_vector=16, seed=143)
    >>> q = model['bin_index_bits'][0]  # vector for the first document
  
    >>> candidates = search_nearby_bins(q, model['table'])
    """
    num_vector = len(query_bin_bits)
    powers_of_two = 1 << np.arange(num_vector-1, -1, -1)
    
    # Allow the user to provide an initial set of candidates.
    candidate_set = copy(initial_candidates)
    
    for different_bits in combinations(range(num_vector), search_radius):       
        # Flip the bits (n_1,n_2,...,n_r) of the query bin to produce a new bit vector.
        ## Hint: you can iterate over a tuple like a list
        alternate_bits = copy(query_bin_bits)
        for i in different_bits:
            alternate_bits[i] = ~alternate_bits[i] 
        
        # Convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)
        
        # Fetch the list of documents belonging to the bin indexed by the new bit vector.
        # Then add those documents to candidate_set
        # Make sure that the bin exists in the table!
        # Hint: update() method for sets lets you add an entire list to the set
        if nearby_bin in table:
            more_docs = table[nearby_bin] # Get all document_ids of the bin
            candidate_set.update(more_docs)
            
    return candidate_set

def query(vec, model, k, max_search_radius):
  
    data = model['data']
    table = model['table']
    random_vectors = model['random_vectors']
    num_vector = random_vectors.shape[1]
    
    
    # Compute bin index for the query vector, in bit representation.
    bin_index_bits = (vec.dot(random_vectors) >= 0).flatten()
    
    # Search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in range(max_search_radius+1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, initial_candidates=candidate_set)
    
    # Sort candidates by their true distances from the query
    nearest_neighbors = turicreate.SFrame({'id':candidate_set})
    candidates = data[np.array(list(candidate_set)),:]
    nearest_neighbors['distance'] = pairwise_distances(candidates, vec, metric='cosine').flatten()
    
    return nearest_neighbors.topk('distance', k, reverse=True), len(candidate_set)