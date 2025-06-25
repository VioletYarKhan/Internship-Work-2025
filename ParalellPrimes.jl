using Base.Threads

function is_prime(n, primes)
    sq = sqrt(n)
    for p in primes
        if p > sq
            break
        end
        if n % p == 0
            return false
        end
    end
    return true
end

function calcPrimes(batchsize=10000)
    println("2")
    primes = [2]
    currNum = 3
    while true
        max = minimum([last(primes)^2, currNum + batchsize*2])
        candidates = currNum:2:max
        isprime_arr = Vector{Bool}(undef, length(candidates))

        @threads for i in eachindex(candidates)
            isprime_arr[i] = is_prime(candidates[i], primes)
        end

        for (i, isprime) in enumerate(isprime_arr)
            if isprime
                n = candidates[i]
                push!(primes, n)
                println(n)
            end
        end
        currNum = last(primes) + 2
    end
end
startTime = time()
calcPrimes()
print(time() - startTime)