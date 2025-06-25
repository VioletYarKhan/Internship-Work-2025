import math

currnum = 3
primes = []

print("2")

while True:
    isPrime = True
    for num in primes:
        if (num > math.sqrt(currnum)):
            break
        if (currnum%num == 0):
            isPrime = False
            break
    if (isPrime):
        primes.append(currnum)
        print(currnum)
    currnum += 2