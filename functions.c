#include <stdio.h>

double sqrt2 = 1.4142135623730950488016887242096980785696718753769480731766797379;
double ln2 = 0.6931471805599453094172321214581765680755001343602552541206800094;
double lnb2 = 1.4426950408889634073599246810018921374266459541529859341354494069;
double pi = 3.1415926535897932384626433832795028841971693993751058209749445923;
double e = 2.7182818284590452353602874713526624977572470936999595749669676277;

int abs(int a) {
    //0s the sign bit
    return a & 0x7FFFFFFF;
}


int intLog2(double a) {
    //get the bits of a using a void pointer (could cast directly but this is cleaner)
    void* bitPtr = &a;
    //treat those bits as an int to perform bit manipulations
    long long bits = *(long long*)bitPtr;
    //get the correct 11 exponent bits specified in the IEEE754 format and subtract the offset;
    return ((bits & 0x7ff0000000000000) >> 52)-1023;
}

int fIntLog2(float a) {
    void* bitPtr = &a;
    int bits = *(int*)bitPtr;
    return ((bits & 0x7f800000) >> 23)-1023;
}

//this function gives floating point numbers the ability to be "bit shifted" (multiplied/divided by powers of 2)
float floatShift(float a, int shift) {
    void* bitPtr = &a;
    int bits = *(int*)bitPtr;
    //get the 8 exponent bits, and then add the shift number to them
    int e = ((bits & 0x7f800000) >> 23)+shift;
    //set the 8 exponent bits of the original number to 0, and then XOR in the new exponent bits
    *(int*)bitPtr = (bits & 0x807FFFFF) ^ ((int) e << 23);
    return *(float*)bitPtr;
}

double doubleShift(double a, int shift) {
    void* bitPtr = &a;
    long long bits = *(long long*)bitPtr;
    int exp = ((bits & 0x7FF0000000000000) >> 52)+shift;
    *(long long*)bitPtr = (bits & 0x800FFFFFFFFFFFFF) ^ ((long long) exp << 52);
    return *(double*)bitPtr;
}

double doubleAbs (double a) {
    void* bitPtr = &a;
    *(long long*)bitPtr = *(long long*)bitPtr & 0x7FFFFFFFFFFFFFFF;
    return *(double*)bitPtr;
}

double log2approx(double a) {
    //note log2((1+frac)*2^exp)) = exp + log2(1+frac), and that log2(1+frac) ~ frac for frac in [0,1)
    int exp = intLog2(a);
    //note a = (1+frac)*2^exp, so frac = (a >> exp) - 1
    double frac = doubleShift(a, -exp)-1;
    return exp + frac;
    
}

float fLog2Approx(float a) {
    int exp = fIntLog2(a);
    float frac = floatShift(a, -exp);
    return frac+exp-1;
}

//implementation of division (1/x)
double divX(double a) {
    //note that log(1/x) = -log(x), and that 2^log2(1/x) = 1/x
    double log2a = log2approx(a);
    int logFloor = (int)(log2a);
    double r = log2a - logFloor;
    //compute 2^-log(x) as 2^-(floor(log2(x)) + frac(log2(x))
    //note that 2^-x ~ -x/2 + 1 for x in [0,1]
    double x0 = doubleShift(1.0, -logFloor) * (-0.5 * r + 1); // second thing is an approximation of 2^-x from 0 to 1
    for (int i = 0; i < 4; i++) {
        x0 = x0 + x0 * (1 - a * x0); //newtons method to improve our guess
    }
    return x0;
}

double exp2s(double a) {
    //computing a taylor series of 2^a where a is between -1 and 1;
    double sol = sqrt2;
    double cur = sqrt2;
    //same principle as the sine function division array, shown below
    float divs[10] = {1,0.5,0.3333333333333333333333333333333333,0.25,0.2,0.16666666666666666666666666666666,0.142857142857142857142857,0.125,0.111111111111111111,0.1};
    for (int i = 1; i < 10; i++) {
        cur = cur * ln2 * (a - 0.5) * divs[i-1];
        sol += cur;
    }
    return sol;
}

double exp2d(double a) {
    //note 2^a = 2^(floor(a) + (a - floor(a))), or 2 raised to the sum of the integral and decimal parts of a
    int shift = (int) a;
    //this form is better, because we can make use of a bit shift to save time computing
    return (1 << shift) * exp2s(a - shift);
}

double log2d(double a) {
    double sol = log2approx(a);
    //newtons method for log 2, computing successive derivatives of 2^x = a (notice the root x = log2(a))
    for (int i = 0; i < 3; i++) {
        sol = sol - lnb2*(1 - a*divX(exp2d(sol)));
    }
    return sol;
}

double nPow(double x, double n) {
    //note x^n = 2^(log_2(x^n)) = 2^(nlog_2(x)), and exp2/log2 are relatively fast operations
    return exp2d(n*log2d(x));
}

//specific optimization for sqrt, for nth root nPow will suffice
double sqrt(double x) {
    double expRaw = log2approx(x)*0.5;
    int expInt = (int)expRaw;
    double remainder = expRaw - expInt;
    double x0 = doubleShift(1.0,expInt)*(remainder+1);
    for(int i = 0; i < 4; i++) {
        x0 = 0.5*(x0 - x*divX(x0));
    }
    return x0;
}

//fast integer division
int intDiv(int dividend, int divisor) {
    int sign = 1;
    if ((dividend < 0) != (divisor < 0)) {
        sign = -1;
    }
    //normalize everything to unsigned
    unsigned int a = dividend & 0x7FFFFFFF;
    unsigned int b = divisor & 0x7FFFFFFF;
    //using unsigned fixes overflow problems
    unsigned int solMag = 0;
    //computing long division with powers of 2 in b
    //suppose a = b*q+r, and we want to compute q
    //let q = 2^s1 + 2^s2... + 2^sn
    //s_i can be computed by finding the difference in digits between a and b, and then a will be updated to be a-(b << i)
    //amount of digits is simply log2(a)
    float lenA = fLog2Approx((float) a);
    float lenB = fLog2Approx((float) b);
    int shift = (int) (lenA - lenB);
    while(a >= b) {
        a = a - (b << shift);
        solMag += 1 << shift;
        lenA = fLog2Approx((float) a);
        shift = (int) (lenA - lenB);
    }
    //fixes overflow
    if (solMag > 2147483647 && sign == 1) {
        return 2147483647;
    }
    int sol = solMag*sign;
    return sol;
}

int doubleIntDiv(double a, double b) {
    int sol = 0;
    a = doubleAbs(a);
    b = doubleAbs(b);
    double lenA = log2approx(a);
    double lenB = log2approx(b);
    int shift = (int) (lenA - lenB);
    while(a >= b) {
        a = a - doubleShift(b,shift);
        sol += 1 << shift;
        lenA = log2approx((double) a);
        shift = (int) (lenA - lenB);
    }
    return sol;
}

int mod(int x, int m) {
    return x - m*intDiv(x,m);
}

double sin(double a) {
    double sol;
    //normalization to put all numbers inside [0,pi/2]
    int multiple = doubleIntDiv(a, 0.5*pi);
    double pos = a - multiple*0.5*pi;
    //checks if a mod pi >= pi/2 and adjusts position accordingly
    if(a - (pi*(int)(0.5*multiple)) >= 0.5*pi) {
        pos = 0.5*pi - pos;
    }
    //checks if a mod 2pi >= pi and adjusts position accordingly
    if(a - (2*pi*(int)(0.25*multiple)) >= pi) {
        pos = -pos;
    }
    //taylor sine approx from [0,pi/2]
    double posSquare = pos*pos;
    sol = pos;
    double cur = pos;
    //contains [1/2*3, 1/4*5, 1/6*7, 1/8*9, 1/10*11, 1/12*13], which will give the taylor series denominators when multiplied in loop
    double divs[6] = {0.166666666666, 0.05, 0.0238095238095, 0.013888888888888, 0.00909090909090909, 0.00641025641025};
    for(int i = 3; i < 15; i += 2) {
        //array index = floor(i/2) - 1, which can be achieved with a right shift to avoid a division call
        cur = -cur*posSquare*divs[(i >> 1)-1];
        sol += cur;
        printf("%f\n",sol);
    }
    return sol;
}

double cos(double a) {
    return sin(0.5*pi - a);
}

double tan(double x) {
    double sine = sin(x);
    //make use of the fact that cos(x) = sqrt(1-sin^2x) to avoid an extra sine computation (sqrt faster than sine)
    return sine*divX(sqrt(1-sine*sine));
}
