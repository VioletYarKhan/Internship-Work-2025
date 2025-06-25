using Base.Threads

function calcPrimes()
    currNum = 3
    println("2")
    primes = [2]
    while true
        isPrime = true
        for num in primes
            if (num > sqrt(currNum))
                break
            end
            if (currNum%num == 0)
                isPrime = false
                break
            end
        end
        if isPrime
            append!(primes, currNum)
            println(currNum)
        end
        currNum += 2
    end
end

startTime = time()
calcPrimes()
print(time() - startTime)