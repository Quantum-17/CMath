//
//  main.c
//  CMath
//
//  Created by Quantum on 6/15/23.
//

#include <stdio.h>
#include <math.h> //for testing code
#include <stdlib.h>
#include <string.h>

double sqrt2 = 1.4142135623730950488016887242096980785696718753769480731766797379;
double ln2 = 0.6931471805599453094172321214581765680755001343602552541206800094;
double lnb2 = 1.4426950408889634073599246810018921374266459541529859341354494069;
double pi = 3.1415926535897932384626433832795028841971693993751058209749445923;
double e = 2.7182818284590452353602874713526624977572470936999595749669676277;


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
//this function is mostly academic, the division operator will be used onwards, but we could just use this
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

//high precision
double padeExp2 (double a) {
    int shift = (int) a;
    a = a-shift;
    double coeffs[5] = {0.346573590279972654708616, 0.05338366821313349162967805, 0.0046253423887351316627618, 0.00022900307399115421814, 0.0000052910945025509696};
    double curExp = a;
    double num = 1;
    double denom = 1;
    double curTerm;
    int denomSign = -1;
    for(int i = 0; i < 5; i++) {
        curTerm = curExp*coeffs[i];
        num += curTerm;
        denom += curTerm*denomSign;
        denomSign = -denomSign;
        curExp *= a;
    }
    return doubleShift(1, shift)*num/denom;
}
double chebyExp2 (double a) {
    void* bits = &a;
    int shift =  ((int)a) - (1 & (*(long long*)bits) >> 63);
    a = 2*(a-shift)-1;
    //13 tech better than this
    double coeffs[] = {1.41421356237309581161, 0.49012907173426339957, 0.08493289604575735008, 0.00981183290529221049, 0.00085013053941534642, 0.00005892655809482639, 0.00000340373106522296, 0.00000016852135602219, 0.00000000730117581854, 0.00000000028056511558, 0.00000000000948872566, 0.000000000000427451256750684265};
    double sol = 0;
    double cur = 1;
    for(int i = 0; i < 12; i++) {
        sol += cur*coeffs[i];
        cur *= a;
    }
    return sol*doubleShift(1, shift);
}
double newtonExp2 (double a) {
    int shift = (int) a;
    a = a - shift;
    double sol = 1+0.69314718054291183*a;
    double cur = a;
    double coeffs[] = {0.2402265086440898, 0.055504080403671, 0.00961831414247, 0.00133275024459, 0.00015512371038, 0.00001416649066, 0.0000018758212221392031390806};
    for(int i = 0; i < 7; i++) {
        cur *= a;
        sol += cur*coeffs[i];
    }
    return sol*doubleShift(1, shift);
}
double splineLog2(double a) {
    int exp = intLog2(a);
    a = doubleShift(a, -exp) - 1;
    double splines[21][4] = {{0.00000000000000000000,1.44267854246379756411,-0.71996650276555584913,0.44262717745356155996},{0.00000547272978000983,1.44233376048765671307,-0.71272608126660219607,0.39194422696090858160},{0.00004818606068698588,1.44098829056408672677,-0.69859864706911556365,0.34249820726969687490},{0.00016426292215239991,1.43855067647331313907,-0.68153534843370222607,0.30268384378706836291},{0.00040147784645161639,1.43481454141559994220,-0.66192063938070622164,0.26835810294432355105},{0.00079566589369595457,1.42984777202032065091,-0.64106020792052897939,0.23915349890006906697},{0.00138224127270787037,1.42368873054069489115,-0.61950356274183848715,0.21400407952492983554},{0.00218722806924928452,1.41644384937182477024,-0.59776891923523478578,0.19226943601833207387},{0.00323162668489883993,1.40821921027358487599,-0.57617924160235456377,0.17337846808956114408},{0.00453003710167684554,1.39913033735614011910,-0.55497187146165372074,0.15688384686901896137},{0.00609220813796908407,1.38928865982750093799,-0.53430434865151443802,0.14241658090192346742},{0.00792347567255090660,1.37880049122034864517,-0.51428148131058515347,0.12967475623042151733},{0.01002561844632247207,1.36776424165804710853,-0.49496804457655579901,0.11840858480223678817},{0.01239747582253735055,1.35626985591177628976,-0.47640019067873545788,0.10841050962648848821},{0.01503554341805588349,1.34439855173194477977,-0.45859323440899130153,0.09950703149161792271},{0.01793448166278411790,1.33222301110408669622,-0.44154747752999046195,0.09155234494808447976},{0.02108747722660454793,1.31980809107154550830,-0.42525289498728258142,0.08442346508565083674},{0.02448688403489590604,1.30721028937023220351,-0.40969090465036850368,0.07801558671162828196},{0.02812323030636628210,1.29448307742008661414,-0.39484249070853244534,0.07224120351202556467},{0.03199114267030548492,1.28165789431860721592,-0.38066728833321644698,0.06701876053164725067},{0.03606464141751075658,1.26882637326490854690,-0.36719419122683072398,0.06230317654441149683}};
    
    double bounds[] = {0,0.047619, 0.095238, 0.142857, 0.190476, 0.238095, 0.285714, 0.333333, 0.380952, 0.428571, 0.476190, 0.523810, 0.571429, 0.619048, 0.666667, 0.714286, 0.761905, 0.809524, 0.857143, 0.904762, 0.952381,1};
    int spline = 0;
    if(a >= 0.047619) {
        void* bitPtr = &a;
        long long bits = *(long long*)bitPtr;
        int expBits = (bits & 0xFFF0000000000000) >> 52;
        int lo;
        int hi;
        int mid;
        switch(expBits) {
            case 1018:
                //P_1
                spline = 1;
                break;
            case 1019:
                //P_2
                spline = 2;
                break;
            case 1020:
                //P_3/4/5
                if(a < 0.190476) {
                    spline = 3;
                } else {
                    if(a < 0.238095) {
                        spline = 4;
                    } else spline = 5;
                }
                break;
            case 1021:
                //P_6/7/8/9/10
                //bsearch
                lo = 6;
                hi = 10;
                mid = 8;
                while(lo < hi) {
                    mid = (lo+hi) >> 1;
                    if (a > bounds[mid]) {
                        lo = mid+1;
                    } else hi = mid-1;
                }
                spline = hi*(a >= bounds[hi]) + mid*(a < bounds[hi]);
                break;
            case 1022:
                //P_11/12/13/14/15/16/17/18/19/20
                //bsearch
                lo = 11;
                hi = 20;
                mid = 15;
                while(lo < hi) {
                    mid = (lo+hi) >> 1;
                    if (a > bounds[mid]) {
                        lo = mid+1;
                    } else hi = mid-1;
                }
                spline = hi*(a >= bounds[hi]) + mid*(a < bounds[hi]);
                break;
            case 1023:
                //P_21
                spline = 21;
                break;
        }
    }
    return exp + splines[spline][0] + splines[spline][1]*a + splines[spline][2]*a*a + splines[spline][3]*a*a*a;
}

double chebyLog2(double a) {
    int exp = intLog2(a);
    double b = 2*doubleShift(a, -exp)-3;
    double coeffs[13] = {0.58496250072115552054, 0.48089834665686792547, -0.08014972439874479271, 0.01781105842913566217, -0.00445276512434744469, 0.00118733558337381722, -0.00032981322405359326, 0.00009446689448088227, -0.00002755713214167699, 0.00000777435631165316, -0.00000232848489059378, 0.00000101805375163905, -0.000000312348874494690487745241};
    double cur = 1;
    double sol = exp;
    for(int i = 0; i < 13; i++) {
        sol += cur*coeffs[i];
        cur *= b;
    }
//    sol = sol - lnb2*(1 - a*(chebyExp2(-sol)));
    return sol;
}

double log2hp(double a) {
    double sol = chebyLog2(a);
    //newtons method for log 2, computing successive derivatives of 2^x = a (notice the root x = log2(a))
    return sol - lnb2*(1 - a*(chebyExp2(-sol)));
}

//high precision
double powHP(double x, double n) {
    //note x^n = 2^(log_2(x^n)) = 2^(nlog_2(x)), and exp2/log2 are relatively fast operations
    return chebyExp2(n*log2hp(x));
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

double legendreSIN(double arg) {
    double sol;
  //arg in [0,pi/2]
    int sign1 = -(arg < 0) + !(arg < 0);
    arg = arg*sign1;
    int div = (int)(M_2_PI*arg);
    double argNorm = arg-div*M_PI_2;
    div = div & 3;
    int sign2 = -(div >> 1) + !(div >> 1);
    argNorm = !(div & 1)*argNorm + (div & 1)*(M_PI_2-argNorm);
    if(argNorm > M_PI_4) {
        //cosine
        argNorm = M_PI/2 - argNorm;
        double arg2 = argNorm*argNorm;
        double arg4 = arg2*arg2;
        double arg8 = arg4*arg4;
        sol=0.99999999999999999899033913768707172-0.49999999999999986265716765799256*arg2+0.0416666666666636214084012688540*arg4-0.001388888888863299470388324357*arg2*arg4+0.000024801587196434694834646645*arg8-2.7557295838991412226509e-7*arg8*arg2+2.087388337742744569003e-9*arg8*arg4-1.1287128526390882751e-11*arg8*arg4*arg2;
    } else {
        //sine
        double arg2 = argNorm*argNorm;
        double arg4 = arg2*arg2;
        double arg8 = arg4*arg4;
        sol=argNorm*(0.9999999999999994995312825130903195-0.166666666666646809009809776993015*arg2+0.00833333333310686889627461111946*arg4-0.0001984126972797658544558330799*arg4*arg2+2.7557290247850414650824156e-6*arg8-2.50481517673677006748711e-8*arg8*arg2+1.5784385437304515164659e-10*arg8*arg4);
    }
    return sol*sign1*sign2;
}

long long choose(long long a, long long b) {
    long long limit;
    if (b > a-b) {
        limit = b;
    } else limit = a-b;
    long long num = 1;
    for(long long i = a; i > limit; i--) {
        num = num*i;
    }
    long long denom = 1;
    for(int i = 2; i <= (a-limit); i++) {
        denom = denom*i;
    }
    return num/denom;
}

//Linalg Methods to permute a matrix, turn a matrix into row echelon form, and reduced row echelon form
//Matrix inversion can be done trivially using RE
void permute(double** coef, double* data, int n) {
    int row = 0;
    double* tempAddr = NULL;
    double temp;
    while(row < n) {
        if(coef[row][row] == 0) {
            int searchRow = row+1;
            while(searchRow < n) {
                if(coef[searchRow][row] != 0) {
                    tempAddr = *(coef + searchRow);
                    *(coef + searchRow) = *(coef + row);
                    *(coef + row) = tempAddr;
                    temp = *(data + searchRow);
                    *(data+searchRow) = *(data + row);
                    *(data+row) = temp;
                    break;
                } else {
                    searchRow++;
                }
            }
        }
        row++;
    }
}

void RE(double** coef, double* data, int n) {
    for(int col = 0; col < n; col++) {
        for(int row = col+1; row < n; row++) {
            if(coef[row][col] != 0) {
                double scale = coef[col][col]/coef[row][col];
                for(int i = col; i < n; i++) {
                    coef[row][i] = scale*coef[row][i] - coef[col][i];
                }
                data[row] = scale*data[row] - data[col];
            }
        }
    }
}

void RRE(double** coef, double* data, int n) {
    for(int row = n-1; row >= 0; row--) {
        data[row] = data[row] / coef[row][row];
        coef[row][row] = 1;
        for(int i = row-1; i >= 0; i--) {
            data[i] = data[i] - coef[i][row]*data[row];
            coef[i][row] = 0;
        }
    }
}

double* solveSystem(double** coef, double* data, int n) {
    permute(coef, data, n);
    RE(coef, data, n);
    RRE(coef, data, n);
    return data;
}

//Generates splines for any function
double** splineClamped(double* x, double* y, int points) {
   //number of polynomials = points - 1, number of coeffs = 4*p
    //matrix ordered as: d_i + c_i x_i + b_i x_i^2 + a_i x_i^3
    double** pCoeff = malloc(4*(points-1)*sizeof(double*)); //polynomial coefficient matrix
    for(int i = 0; i < 4*(points-1); i++) {
        pCoeff[i] = calloc(4*(points-1), sizeof(double));
    }
    double* dataMatrix = calloc(4*(points-1), sizeof(double));
    int eqPtr = 0;
    //point equations
    for(int i = 0; i < points-1; i++) {
        //equation 1: P_i(x_i) = y_i
        pCoeff[eqPtr][4*i] = 1;
        pCoeff[eqPtr][4*i+1] = x[i];
        pCoeff[eqPtr][4*i+2] = x[i] * pCoeff[eqPtr][4*i+1];
        pCoeff[eqPtr][4*i+3] = x[i]*pCoeff[eqPtr][4*i+2];
        dataMatrix[eqPtr] = y[i];
        //equation 2: P_i(x_{i+1}) = y_{i+1}
        eqPtr++;
        pCoeff[eqPtr][4*i] = 1;
        pCoeff[eqPtr][4*i+1] = x[i+1];
        pCoeff[eqPtr][4*i+2] = x[i+1] * pCoeff[eqPtr][4*i+1];
        pCoeff[eqPtr][4*i+3] = x[i+1] * pCoeff[eqPtr][4*i+2];
        dataMatrix[eqPtr] = y[i+1];
        eqPtr++;
        //set data matrix
    }
    //first derivative equations
    for(int i = 0; i < points-2; i++) {
        //want P_i'(x_{i+1}) = P_{i+1}'(x_{i+1})
        //note that P_i'(x_i) = c_i + 2 b_i x_i + 3 a_i x_i^2
        //equation in the form c_i + 2 b_i x_i+1 + 3 a_i x_i+1^2 - c_i+1 - 2 b_i+1 x_i+1 - 3 a_i+1 x_i+1^2 = 0
        pCoeff[eqPtr][4*i+1] = 1;
        pCoeff[eqPtr][4*i+2] = 2*x[i+1];
        pCoeff[eqPtr][4*i+3] = 3*x[i+1]*x[i+1];
        pCoeff[eqPtr][4*i+5] = -1;
        pCoeff[eqPtr][4*i+6] = -2*x[i+1];
        pCoeff[eqPtr][4*i+7] = -3*x[i+1]*x[i+1];
        dataMatrix[eqPtr] = 0;
        eqPtr++;
    }
    //second derivative equations
    for(int i = 0; i < points-2; i++) {
        //want P_i''(x_{i+1}) = P_{i+1}''(x_{i+1})
        //note that P_i''(x_i) = 2 b_i + 6 a_i x_i
        //equation in the form 2 b_i + 6 a_i x_i+1 - 2 b_i+1 -6 a_i+1 x_i+1 = 0
        pCoeff[eqPtr][4*i+2] = 2;
        pCoeff[eqPtr][4*i+3] = 6*x[i+1];
        pCoeff[eqPtr][4*i+6] = -2;
        pCoeff[eqPtr][4*i+7] = -6*x[i+1];
        dataMatrix[eqPtr] = 0;
        eqPtr++;
    }
    //clamped equations (YOU NEED TO MODIFY THIS YOURSELF!!!)
    //General form P_0'(x_0) = f'(x_0), P_n'(x_n) = f'(x_n)
    
    //for exp2(x): P'(0) = log2, P'(1) = log4
    pCoeff[eqPtr][1] = 1; //P_0'(0) = c_0
    dataMatrix[eqPtr] = ln2;
    
    eqPtr++;
    //P_n'(1) = c_n + 2b_n + 3a_n
    pCoeff[eqPtr][4*points - 7] = 1;
    pCoeff[eqPtr][4*points - 6] = 2;
    pCoeff[eqPtr][4*points - 5] = 3;
    dataMatrix[eqPtr] = ln2*2;
    
    //solve for coefficients
    permute(pCoeff, dataMatrix, 4*(points-1));
    RE(pCoeff, dataMatrix, 4*(points-1));
    RRE(pCoeff, dataMatrix, 4*(points-1));
    
    double** p = malloc((points-1)*sizeof(double*)); //polynomial matrix
    for(int i = 0; i < 4*(points-1); i++) {
        p[i] = calloc(4, sizeof(double));
    }
    for(int i = 0; i < points-1; i++) {
        for(int j = 0; j < 4; j++) {
            p[i][j] = dataMatrix[4*i + j];
        }
    }
    //free everything else
    for(int i = 0; i < 4*(points-1); i++) {
        free(pCoeff[i]);
    }
    free(pCoeff);
    free(dataMatrix);
    return p;
}

//Generates an approximating chebyshev polynomial for any function of order n
void chebyshevPoly(int n) {
    //u = (2x-a-b)/(b-a), x = (b-a)/2 u + (a+b)/2
    //for exp2 in range [0,1]:
    //u = 2x-1, x = 1/2 u + 1/2
    double u[n][n]; //each entry is T_j(u_i)
    double y[n];
    double c[n];
    memset(&c, 0, n*sizeof(double));
    //init u,y
    u[0][0] = 1;
    for(int i = 1; i < n+1; i++) {
        u[i][0] = 1;
        u[n-i][1] = cos((2*i - 1)*M_PI/(2*n)); //chebyshev nodes
        y[n-i] = log2(1+(u[n-i][1]/2) + 0.5); //inner thing is just x_i as given by the above transform, pick whatever function you want
    }
    double cur;
    //c0
    for(int i = 0; i < n; i++) {
        c[0] += y[i];
    }
    c[0] /= n;
    
    //c1
    for(int i = 0; i < n; i++) {
        c[1] += u[i][1]*y[i];
    }
    c[1] = 2*c[1]/n;
    //c_i = 2*(T_i(u) dot y)/n
    for(int i = 2; i < n; i++) {
        //compute T_i(u_j)*y_j
        cur = 0;
        for(int j = 0; j < n; j++) {
            u[j][i] = 2*u[j][1]*u[j][i-1] - u[j][i-2];
            cur += y[j]*u[j][i];
        }
        c[i] = 2*cur/n;
    }
//Technically you are already done here, but for ease of use the below methods simplify the polynomial for you IN TERMS OF U NOT X
//generate T_k
    double T[n][n];
    memset(&T, 0, n*n*sizeof(double));
    T[0][0] = 1;
    T[1][1] = 1;
    for(int i = 2; i < n; i++) {
        T[i][0] = -T[i-2][0];
        for(int j = 1; j < n; j++) {
            T[i][j] = 2*T[i-1][j-1] - T[i-2][j];
        }
    }
    double p[n];
    memset(&p, 0, n*sizeof(double));
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            p[j] += c[i]*T[i][j];
        }
    }
//    Prints the equation of the polynomial to be plotted

//    for(int i = 0; i < n; i++) {
//        printf("%0.20f(2x-1)^%d + ", p[i], i);
//    }
    
    //prints array of coefficients, formatted so that you can just copy paste it
    printf("{");
    for(int i = 0; i < n-1; i++) {
        printf("%0.20f, ", p[i]);
    }
    printf("%0.30f}\n", p[n-1]);
}
