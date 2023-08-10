import numpy as np
import argparse

parser = argparse.ArgumentParser(description="A program to obtain prime numbers between two numbers", 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--numbers", "-N", type=int, nargs='+')
parser.add_argument("--mode", "-M", choices=["Check", "Generate"], default="Answer")
parser.add_argument("--explain", "-E", action='store_true')


def check_prime(num:int) -> tuple:
    if num - int(num) > 0:
        raise ValueError(f"Input should be an integer, not a float with decimals")
    if num <= 0:
        return False, None
    if num == 1:
        return False, 1
    sqrt = np.sqrt(num)
    flag = True
    factors = []
    for i in range(2, int(sqrt)+1):
        if num % i == 0:
            flag = False
            factors.append(i)
    if flag:
        return True, None
    return False, factors


def get_primes(start:int, end:int):   
    primes = []
    for n in range(start, end+1):
        flag, _ = check_prime(n)
        if flag:
            primes.append(n)        
    return primes


if __name__ == "__main__":
    args = parser.parse_args()
    if args.mode == "Check":
        nums = args.numbers
        for num in nums:
            flag, factors = check_prime(num)
            if flag:
                print(f"{num} is a prime number.")
            else:
                if args.explain:
                    print(f"{num} is not a prime number.")
                    if factors:
                        print(f"{num}", end="")
                        for factor in factors:
                            print(f" = ({factor} * {int(num/factor)})", end="")
                        print()
                    else:
                        print(f"\t{num} can be divided by nothing.")
                else:
                    print(f"{num} is not a prime number.")
    
    else:
        nums = args.numbers
        if len(nums) > 2:
            raise ValueError(f"For the {args.mode} mode, you can only pass 2 inputs.")
        else:
            a, b = nums
        
        prime_list = get_primes(start=a, end=b)
        if len(prime_list) == 0:
            print(f"There are no prime numbers between {a} and {b}")
        else:
            print(f"Prime numbers between {a} and {b} are: ", end="")
            for item in prime_list:
                print(item, end=" ")
        print()
