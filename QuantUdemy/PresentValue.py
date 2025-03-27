from math import exp

# Here x will be present value
def future_discrete_value(x, r, n):
    return x*(1+r)**n

# Here x will be future value
def present_discrete_value(x, r, n):
    return x*(1+r)**-n

# Here x will be present value
def future_continuous_value(x, r, t):
    return x*exp(r*t)

# Here x will be future value
def present_continuous_value(x, r, t):
    return x*exp(-r*t)


if __name__ == '__main__':

    x = 100
    r = 0.05 # 5%
    n = 5

    print("Future value of x: %s" % future_discrete_value(x,r,n))
    print("Present value of x: %s" % present_continuous_value(x, r, n))
    print("Future value of x: %s" % future_continuous_value(x, r, n))
    print("Present value of x: %s" % present_continuous_value(x, r, n))