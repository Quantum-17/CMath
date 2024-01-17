# CMath
My own implementation of many commonly used math functions, including floating point and integer division, modulo for integers and floating points, trig functions, log2, exp2, sqrt, and pow. I ONLY use addition, multiplication, and subtraction operators implicitly. 


**Bit Manipulation**
As seen in the code, I make use of many bit manipulations to speed up computation. Commonly, I will use a void pointer to get the bits of a floating point number and alter them as needed for the function, and then I will revert the type back to floating point when returning. This can be done without the use of a void pointer, by casting directly from one pointer type to another, but I find that this is unintuitive.

**APPROXIMATION METHODS**
As of now, the approximation techniques are not very sophisticated. The comments in the code explain per function exactly what I am doing, but I will list the recurring ones below:

1. Taylor Series. I mostly make use of Taylor series when the function I am approximating has a very small domain that is also near 0 like sin(x) (as every point can be transformed to a value between 0 and pi/2), or 2^x (as the integral part can be computed with a bit shift directly).

2. Newton's Method. I make use of Newton's method to improve imprecise guesses which can be computed really quickly (especially in my sqrt and floating point division calculations)

3. Linear approximations. Useful for a quick approximation of log_2(x) or exp2 if precision is not needed, as seen in the code

4. As I become more mathematically mature, I am going to look into adding more sophisticated/faster approximation methods, specifically using Chebyshev interpolation as many real implementations do. However, as of now, while I have read the theory behind it, I don't feel comfortable enough with it to implement it here.
