# CMath
My own implementation of many commonly used math functions, including floating point and integer division, modulo for integers and floating points, trig functions, log2, exp2, sqrt, and pow. I ONLY use addition, multiplication, and subtraction operators implicitly (division used in pade, but it could be called with divX instead). If any other more complex function is needed, I will have implemented it myself. Note that addition/subtraction/multiplication can be computed only with the basic logic operations, and I might implement multiplication using an FFT based algorithm in the future. However, as of now, I am taking these 3 to be fundamental operations, along with the bitwise operations. I will outline some ideas I used to speed up computation below.


**BIT MANIPULATIONS**

1. Bit Shifts. These speed up multiplication and division because they auto compute them for integers being multiplied/divided by powers of 2. This technique is used almost everywhere, but it's of greatest effect in the integer division method.

2. Bit Masking. Commonly it is useful to change certain bits in any variable (for example setting the sign bit to 0 for abs). It may also be necessary to get certain groups of bits (e.g. exponent bits in a floating point number). When performing these operations, it is helpful to take the bitwise AND of a number with a specific number (like 0x80...) to set every bit to 0 other than the ones we want. Additionally, it may be useful to use the bitwise XOR to concatenate/insert new sets of bits into a number (as seen in the floatShift/doubleShift method).

3. Void Pointers. Void pointers are very useful when performing bit manipulations on non integer typed variables (especially floating point numbers). This technique is used heavily throughout the code to get the different parts of floating point numbers. My motivation for using a void pointer is that inherently when we are performing bit operations we don't care about what those bits are meant to represent in the moment, we just want to change things around, and so we can just remove the typing by using a void pointer, and then cast that pointer to an int or long long or anything else. After performing our operations however, it is important that to return the ORIGINAL type, and not whatever it was casted to in the middle.

**APPROXIMATION METHODS**

This file has undergone a lot of changes, and I will be detailing the journey I took below.

1. Taylor Series: You need to start somewhere, and this is the most obvious starting point. They work, but they are quite slow, and only work over a very small area. The error is not great unless you use a high order polynomial, so I started looking for something better.

2. Pade Approximants: Basically, taylor series but better. In terms of precision, you really can't complain. An order 5-7 pade approximant should fulfill any precision requirement you have, but the limiting problem is that they are really slow. 2n multiplications atleast (probably a bit more), and a division cause this method to not be ideal. For a long time though, these were used in the high precision implementation of each function that needed to be approximated. Can we get faster?

3. Lagrange Polynomials: These are fast, and maybe the most well known polynomial approximation method. They will interpolate the function at the points you describe, but the problem is that they don't do a very good job of describing the functions behavior. They don't take into account any of the functions derivatives, and as a result the error term outside the interpolation points is relatively high. Is there something more precise?

4. Newton Polynomials: Now we are really getting somewhere! These are fast to evaluate, and really quite precise. Unfortunately, you can't really get arbitrary precision with these, but they get the job done. Additionally, if you have some kind of iterative refinement algorithm (Newton-Raphson for example), these can be extremely powerful. After some numerical testing, I did not manage to make these as good as the in built C functions though, so the search continues.

5. Spline interpolation: When trying to approximate a non-polynomial, we should assume that the function does not look like a polynomial on a large scale, obviously. However, on a small scale, a smooth function looks pretty similar to a polynomial. This is the intuition behind spline interpolation, which splits the interpolated function into different sections called "splines", and then fits a polynomial (usually cubic) to that small section. The final piecewise function is determined such that it is continuous (C0), the first derivatives of each spline are equal to each other at their intersection points (C1), and that the second derivatives are equal at each point (C2). Additionally, this library uses a clamped spline, because we know information about the derivative of each interpolated function. Splines are more general though, and if the interpolated function is unknown or not differentiable at the end points, we can use a natural spline instead. Spline interpolation is extremely powerful, however the runtime is not as good as you might expect. While you do only need to evaluate a cubic (3 multiplications), you also need to figure out which spline to use. This is a nontrivial task, and the best implementation I could find was a reduced binary search, as seen in the code (I split first on the exponent in the IEE754 format to avoid bsearch in 7 cases). Splines are precise, and they are fast, so are we done? Not quite! The C library still outperforms on every function, so there still must be something left to find.

6. Chebyshev Polynomials: To put it plainly, these are cracked. The chebyshev polynomials will find you the BEST interpolating polynomial for any function. If you want fast runtime, cut the order down to like 5-6. If you need high precision (that is implemented here in the code), go for order 12-13. If you need even higher precision, go for one iteration of Newton-Raphson (although on exp2 it turned out the original guess was actually better lol...). Talking more precisely about the precision, log2 with one iteration of newton raphson can go digit for digit with the C library no matter what the input number is. Exp2 performs slightly worse, owing to the fact that you can't newton-raphson it (not sure why this is), but it is still far better than any of the previous methods, barring some ungodly high order pade approximant (the tradeoff being that these are fast to compute). 

7. Newton-Raphson: this is not an approximation method but I have mentioned it previously. It is one of many iterative refinement algorithms, that can turn a pretty good guess into a really good one. There are other possible methods to do this (Steffensen, bisection, secant etc.) but this is the one I am using at the moment, because it avoid divisions entirely for the functions I am using it on. I will experiment more with these in the future.

**Generator/Helper Methods**
1. Gauss Elimination methods: Pretty self explanatory, they can output the solution for any nxn matrix, if it exists. They aren't super powerful and probably don't handle edge cases that well as I haven't built them for heavy use, but if you know that your system can be solved these can get the job done. Also, as written in the comments, you can perform matrix inversion with these

2. Spline maker: Generates a series of spline polynomials for any input function, if you want to use this method.

3. Chebyshev maker: Generates a chebyshev polynomial for any function of a specified order. Make sure you plot the function to see how good it is because in my experience higher order is not always better.



