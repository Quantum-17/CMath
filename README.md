# CMath
My own implementation of many commonly used math functions, including floating point and integer division, modulo for integers and floating points, trig functions, log2, exp2, sqrt, and pow. I ONLY use addition, multiplication, and subtraction operators implicitly. If any other more complex function is needed, I will have implemented it myself. Note that addition/subtraction/multiplication can be computed only with the basic logic operations, and I might implement multiplication using an FFT based algorithm in the future. However, as of now, I am taking these 3 to be fundamental operations, along with the bitwise operations. I will outline some ideas I used to speed up computation below.


**BIT MANIPULATIONS**

1. Bit Shifts. These speed up multiplication and division because they auto compute them for integers being multiplied/divided by powers of 2. This technique is used almost everywhere, but it's of greatest effect in the integer division method.

2. Bit Masking. Commonly it is useful to change certain bits in any variable (for example setting the sign bit to 0 for abs). It may also be necessary to get certain groups of bits (e.g. exponent bits in a floating point number). When performing these operations, it is helpful to take the bitwise AND of a number with a specific number (like 0x80...) to set every bit to 0 other than the ones we want. Additionally, it may be useful to use the bitwise XOR to concatenate/insert new sets of bits into a number (as seen in the floatShift/doubleShift method).

3. Void Pointers. Void pointers are very useful when performing bit manipulations on non integer typed variables (especially floating point numbers). This technique is used heavily throughout the code to get the different parts of floating point numbers. My motivation for using a void pointer is that inherently when we are performing bit operations we don't care about what those bits are meant to represent in the moment, we just want to change things around, and so we can just remove the typing by using a void pointer, and then cast that pointer to an int or long long or anything else. After performing our operations however, it is important that to return the ORIGINAL type, and not whatever it was casted to in the middle.

**APPROXIMATION METHODS**

As of now, the approximation techniques are not very sophisticated. The comments in the code explain per function exactly what I am doing, but I will list the recurring ones below:

1. Taylor Series. I mostly make use of Taylor series when the function I am approximating has a very small domain that is also near 0 like sin(x) (as every point can be transformed to a value between 0 and pi/2), or 2^x (as the integral part of an exponent can be computed with a bit shift directly). Because the domain we need to approximate is relatively small for these functions, the amount of operations needed in the taylor series is low (6 for sin(x) and 10 for 2^x). As a result, while there are definitely better approximation methods, taylor series loses very little time over them.

2. Newton's Method. I make use of Newton's method to improve imprecise guesses which can be computed really quickly (especially in my sqrt and floating point division calculations). At most only 4 iterations need to be computed to achieve 13 digits worth of accuracy, so my floating point division method for example happens very quickly.

3. Linear approximations. Useful for a quick approximation of log_2(x) or exp2 if precision is not needed, as seen in the code.

As I become more mathematically mature, I am going to look into adding more sophisticated/faster approximation methods, specifically using Chebyshev interpolation as many real implementations do. However, as of now, while I have read the theory behind it, I don't feel comfortable enough with it to implement it here.
