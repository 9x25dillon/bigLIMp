# limps/src/entropy.jl
module Entropy

using Statistics
using ..Matrices: matrix_to_polynomials, optimize_matrix
using ..Polynomials: analyze_polynomials

export analyze_text_structure, process_entropy_matrix
export extract_text_features, calculate_text_entropy
export calculate_matrix_entropy, analyze_entropy_distribution

"""
Analyze text structure (length, vocab, entropy, etc.)
"""
function analyze_text_structure(text::String)
    words = split(text)
    
    analysis = Dict{String, Any}()
    analysis["text_length"]         = length(text)
    analysis["word_count"]          = length(words)
    analysis["unique_words"]        = length(unique(words))
    analysis["average_word_length"] = mean(length.(words))
    analysis["complexity_score"]    = analysis["text_length"] / 100.0
    
    word_freq = Dict{String, Int}()
    for word in words
        word_freq[word] = get(word_freq, word, 0) + 1
    end
    
    total_words = max(length(words), 1)
    entropy = 0.0
    for (_, freq) in word_freq
        p = freq / total_words
        entropy -= p * log(p)
    end
    
    analysis["text_entropy"]       = entropy
    analysis["vocabulary_richness"]= analysis["unique_words"] / max(analysis["word_count"], 1)
    analysis["word_frequency"]     = word_freq
    
    return analysis
end

"""
Extract numerical features from text
"""
function extract_text_features(text::String)
    words = split(text)
    total_words = max(length(words), 1)
    
    features = [
        Float64(length(text)),
        Float64(total_words),
        Float64(length(unique(words))),
        mean(Float64.(length.(words))),
        calculate_text_entropy(text)
    ]
    
    return reshape(features, 1, :)
end

"""
Calculate Shannon entropy of text
"""
function calculate_text_entropy(text::String)
    words = split(text)
    word_freq = Dict{String, Int}()
    
    for word in words
        word_freq[word] = get(word_freq, word, 0) + 1
    end
    
    total_words = max(length(words), 1)
    entropy = 0.0
    
    for (_, count) in word_freq
        p = count / total_words
        entropy -= p * log(p)
    end
    
    return entropy
end

"""
Calculate matrix entropy (coarse quantized)
"""
function calculate_matrix_entropy(matrix::Matrix{Float64})
    flat_values = vec(matrix)
    hist = Dict{Float64, Int}()
    
    for val in flat_values
        rounded_val = round(val, digits=3)
        hist[rounded_val] = get(hist, rounded_val, 0) + 1
    end
    
    total = max(length(flat_values), 1)
    entropy = 0.0
    
    for (_, count) in hist
        p = count / total
        entropy -= p * log(p)
    end
    
    return entropy
end

"""
Analyze entropy distribution in matrix (rows, cols, global)
"""
function analyze_entropy_distribution(matrix::Matrix{Float64})
    m, n = size(matrix)
    
    row_entropies = [calculate_matrix_entropy(reshape(matrix[i, :], 1, n)) for i in 1:m]
    col_entropies = [calculate_matrix_entropy(reshape(matrix[:, j], m, 1)) for j in 1:n]
    
    overall_entropy = calculate_matrix_entropy(matrix)
    
    return Dict(
        "overall_entropy"  => overall_entropy,
        "row_entropies"    => row_entropies,
        "col_entropies"    => col_entropies,
        "mean_row_entropy" => mean(row_entropies),
        "mean_col_entropy" => mean(col_entropies),
        "entropy_variance" => var(row_entropies)
    )
end

"""
Process entropy matrix through polynomial analysis
"""
function process_entropy_matrix(matrix::Matrix{Float64})
    poly_result     = matrix_to_polynomials(matrix)
    analysis_result = analyze_polynomials(poly_result)
    
    complexity = analysis_result["complexity_score"]
    method =
        complexity > 0.7 ? "rank" :
        complexity > 0.4 ? "structure" :
                           "sparsity"
    
    opt_result = optimize_matrix(matrix, method)
    
    return Dict(
        "polynomial_representation" => poly_result,
        "analysis"                  => analysis_result,
        "optimization"              => opt_result,
        "complexity_score"          => complexity,
        "optimization_method"       => method
    )
end

end # module Entropy
