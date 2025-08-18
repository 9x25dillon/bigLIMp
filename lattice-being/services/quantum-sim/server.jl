using HTTP, JSON3

function collapse(req)
    candidates = req["candidates"]
    priors = haskey(req, "priors") ? req["priors"] : fill(1.0, length(candidates))
    s = sum(priors)
    weights = [p/s for p in priors]
    winner = candidates[argmax(weights)]
    return Dict("winner_id" => winner["id"], "weights" => weights)
end

HTTP.serve() do http::HTTP.Request
    if http.method == "POST" && http.target == "/collapse"
        body = String(take!(http.body))
        req = JSON3.read(body)
        res = collapse(req)
        return HTTP.Response(200, JSON3.write(res))
    else
        return HTTP.Response(404, "Not Found")
    end
end
