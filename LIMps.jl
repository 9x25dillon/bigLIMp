# limps/src/LIMPS.jl
module LIMPS

using DynamicPolynomials
using MultivariatePolynomials
using LinearAlgebra
using JSON
using HTTP
using Sockets
using Statistics
using Logging
using Dates

# Include submodules
include("polynomials.jl")
include("matrices.jl")
include("entropy.jl")
include("api.jl")
include("config.jl")

# Submodules
using .Polynomials: create_polynomials, analyze_polynomials, optimize_polynomial
using .Matrices: optimize_matrix, matrix_to_polynomials, analyze_matrix_structure
using .Entropy: analyze_text_structure, process_entropy_matrix,
                extract_text_features, calculate_text_entropy
using .API: start_http_server, start_limps_server
using .Config: LIMPSConfig, configure_limps

export create_polynomials, analyze_polynomials, optimize_polynomial
export optimize_matrix, matrix_to_polynomials, analyze_matrix_structure
export analyze_text_structure, process_entropy_matrix
export start_http_server, start_limps_server
export LIMPSConfig, configure_limps
export process_limps_data, batch_process_limps, health_check

"""
Main LIMPS processing function
"""
function process_limps_data(data::Union{Matrix{Float64}, Vector{Float64}, String},
                           data_type::String = "matrix")
    try
        if data_type == "matrix"
            return process_matrix_data(Matrix{Float64}(data))
        elseif data_type == "vector"
            return process_vector_data(Vector{Float64}(data))
        elseif data_type == "text"
            return process_text_data(String(data))
        else
            return Dict("error" => "Unknown data type: $data_type")
        end
    catch e
        return Dict("error" => "Processing failed: $(e)")
    end
end

function process_matrix_data(matrix::Matrix{Float64})
    """Process matrix data through LIMPS pipeline"""
    results = Dict{String, Any}()
    
    # Convert to polynomial representation
    results["polynomial_representation"] = matrix_to_polynomials(matrix)
    
    # Analyze structure
    results["structure_analysis"] = analyze_matrix_structure(matrix)
    
    # Select optimization method based on structure
    complexity = results["structure_analysis"]["complexity_score"]
    method =
        complexity > 0.7 ? "rank" :
        complexity > 0.4 ? "structure" :
                           "sparsity"
    
    results["optimization"] = optimize_matrix(matrix, method)
    results["optimization_method"] = method
    
    return results
end

function process_vector_data(vector::Vector{Float64})
    """Process vector data through LIMPS pipeline"""
    matrix = reshape(vector, :, 1)
    return process_matrix_data(matrix)
end

function process_text_data(text::String)
    """Process text data through LIMPS pipeline"""
    results = Dict{String, Any}()
    
    # Analyze text structure
    results["text_analysis"] = analyze_text_structure(text)
    
    # Create feature vector via Entropy module helper
    features = extract_text_features(text)
    results["feature_vector"] = features
    
    # Convert features to polynomial representation
    variables = ["length", "words", "unique", "avg_len", "entropy"]
    results["polynomial_features"] = create_polynomials(features, variables)
    
    return results
end

"""
Batch processing for multiple datasets
"""
function batch_process_limps(data_list::Vector{Any},
                             data_types::Vector{String})
    """Process multiple datasets in batch"""
    results = Vector{Any}()
    
    for (data, data_type) in zip(data_list, data_types)
        try
            result = process_limps_data(data, data_type)
            push!(results, result)
        catch e
            push!(results, Dict("error" => "Batch processing failed: $(e)"))
        end
    end
    
    return results
end

"""
Health check function for microservice
"""
function health_check()
    """Return system health status"""
    return Dict(
        "status"   => "healthy",
        "timestamp"=> string(now()),
        "version"  => "1.0.0",
        "modules"  => ["Polynomials", "Matrices", "Entropy", "API"]
    )
end

"""
Main function for testing
"""
function main()
    println("=== LIMPS.jl Module Test ===")
    
    # Test matrix processing
    println("1. Testing matrix processing...")
    matrix = rand(5, 5)
    result1 = process_limps_data(matrix, "matrix")
    println("Matrix processing: ", haskey(result1, "error") ? "FAILED" : "SUCCESS")
    
    # Test text processing
    println("2. Testing text processing...")
    text = "Show monthly sales totals for electronics category"
    result2 = process_limps_data(text, "text")
    println("Text processing: ", haskey(result2, "error") ? "FAILED" : "SUCCESS")
    
    # Test batch processing
    println("3. Testing batch processing...")
    data_list = Any[matrix, text]
    data_types = ["matrix", "text"]
    result3 = batch_process_limps(data_list, data_types)
    println("Batch processing: ", length(result3), " items processed")
    
    # Test health check
    println("4. Testing health check...")
    health = health_check()
    println("Health status: ", health["status"])
    
    println("All tests completed!")
end

# Run main function if script is executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

end # module LIMPS
