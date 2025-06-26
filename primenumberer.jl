using Base.Threads

function calcPrimes()
    currNum = 3
    println("2")
    primes = [2]
    # 29.907999992370605
    while last(primes) < 5000000
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