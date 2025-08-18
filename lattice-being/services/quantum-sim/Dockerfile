FROM julia:1.10-bullseye
WORKDIR /app
COPY server.jl /app/server.jl
EXPOSE 8081
CMD ["julia", "/app/server.jl"]
