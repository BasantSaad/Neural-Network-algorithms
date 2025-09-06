import streamlit as st
import numpy as np

# ART1 algorithm implementation
def art1(input_vectors, bottom_up, top_down, rho):
    num_inputs, num_clusters = bottom_up.shape                            #1 (4,3) = 4 inputs and 3 clusters
    for input_vec in input_vectors:
        input_vec = np.array(input_vec)
        # Compute activations T_j                                         #5(net input to each cluster j = input_vec * bottom_up[:, j])
        activations = np.zeros(num_clusters)
        for j in range(num_clusters):
            activations[j] = np.sum(input_vec * bottom_up[:, j])
        
        # Sort clusters (descending)                                      #6(sort the clusters in descending order of their activations)
        cluster_order = np.argsort(-activations)
        found_cluster = False
        
        # Try each cluster until resonance or all clusters are inhibited  #7 back to F1(b)
        for j in cluster_order:                                           # for 8-  If test of reset is true, similarity < rho that mean reset and try again with the next cluster
            # Compute similarity
            t_j = top_down[j, :]
            intersection = np.sum(np.logical_and(input_vec, t_j))         #7.1 activation of winning cluster
            norm_input = np.sum(input_vec)                                #7.2 normalization of input vector
            similarity = intersection / norm_input if norm_input > 0 else 0
            
            # Check if similarity meets vigilance                         #8 test for reset (if similarity >= rho) is false to reset
            if similarity >= rho:                                         # and similarity >= rho then resonance is true and update weights of winning cluster
                # Resonance: Update weights
                # Top-down update: t_ji = I_i AND t_ji
                top_down[j, :] = np.logical_and(input_vec, t_j)            #9 update top-down weights (Tji=xi)
                
                # Bottom-up update: b_ij = (2 * t_ji) / (1 + ||t_j||)      #update bottom-up weights (Bij =L*Tji/(L-1+||Xj||) L=2)
                t_j_new = top_down[j, :]                                   #XJ activation of winning cluster
                norm_t_j = np.sum(t_j_new)                                 #||Xj|| 
                denominator = 1 + norm_t_j                                 # 1+||Xj|| المقام
                for i in range(num_inputs):
                    bottom_up[i, j] = (2 * t_j_new[i]) / denominator       # final bottom-up weights (Bij =L*Tji/(L-1+||Xj||) L=2)
                
                found_cluster = True
                break
        
        if not found_cluster:
            st.warning(f"No cluster found for input {input_vec} with vigilance {rho}")
    
    return bottom_up, top_down

# Streamlit UI
st.title("ART1 Neural Network 🧠")

# Input for vectors
st.header("Input Vectors 📥")
st.write("Enter input vectors as comma-separated values ")
input_vectors_str = st.text_input("Input Vectors", "1,1,0,0;0,0,0,1;1,0,0,0")

# Input for bottom-up matrix
st.header("Bottom-Up Weight Matrix ⬆️")
st.write("Enter the bottom-up matrix as semicolon-separated rows")
bottom_up_str = st.text_input("Bottom-Up Matrix", "1,0,0.2;0,0,0.2;0,0,0.2;0,1,0.2")

# Input for top-down matrix
st.header("Top-Down Weight Matrix ⬇️")
st.write("Enter the top-down matrix as semicolon-separated rows")
top_down_str = st.text_input("Top-Down Matrix", "1,0,0,0;0,0,0,1;1,1,1,1")

# Input for vigilance parameter
st.header("Vigilance Parameter ")
rho = st.slider("Vigilance Parameter (rho)", 0.0, 1.0, 0.7)

# Process inputs and run ART1
if st.button("Run ART1 🚀"):
    try:
        # Parse input vectors
        input_vectors = []
        for vec_str in input_vectors_str.split(";"):
            vec = [int(x) for x in vec_str.split(",")]
            input_vectors.append(vec)
        input_vectors = np.array(input_vectors)
        
        # Parse bottom-up matrix
        bottom_up = []
        for row_str in bottom_up_str.split(";"):
            row = [float(x) for x in row_str.split(",")]
            bottom_up.append(row)
        bottom_up = np.array(bottom_up)
        
        # Parse top-down matrix
        top_down = []
        for row_str in top_down_str.split(";"):
            row = [int(x) for x in row_str.split(",")]
            top_down.append(row)
        top_down = np.array(top_down)
        
        # Validate input dimensions
        num_inputs, num_clusters = bottom_up.shape
        if top_down.shape != (num_clusters, num_inputs):
            st.error("Dimensions mismatch: Top-down matrix must have shape (num_clusters, num_inputs).")
        elif input_vectors.shape[1] != num_inputs:
            st.error("Each input vector must have the same number of dimensions as the bottom-up matrix rows.")
        else:
            # Run ART1
            final_bottom_up, final_top_down = art1(input_vectors, bottom_up, top_down, rho)
            
            # Display results
            st.header("Results ")
            st.subheader("Final Bottom-Up Weight Matrix 📈")
            st.write(final_bottom_up)
            
            st.subheader("Final Top-Down Weight Matrix 📈")
            st.write(final_top_down)
    
    except Exception as e:
        st.error(f"Error processing inputs: {str(e)}")