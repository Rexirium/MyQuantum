///////////////////////////////////////////////////////////////////////
//                                                                   //
//                      MBL - Weak Measurements                      //
//                                                                   //
//   Needs Eigen library to compile.                                 //
//                                                                   //
//   Arguments (all optional):                                       //
//     L         system size (default 16)                            //
//     Delta     weak measurement parameter (default 1.0)            //
//     lambda    weak measurement parameter (default 0.5)            //
//     p         probability of measurement (default 0.5)            //
//     steps     number of steps (default 10000)                     //
//     verbose   print all the steps (default 0)                     //
//                                                                   //
///////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <iostream>
#include <cmath>
#include <complex>
#include <random>
#include <chrono>
#include <iterator>
#include <algorithm>

#include "Eigen/Dense"
#include "Eigen/QR"
#include "Eigen/Eigenvalues"

using namespace std;
using namespace Eigen;

double const pi = acos(-1.);
default_random_engine generator;
normal_distribution<double> distribution(0.0,1.0);
uniform_real_distribution<double> uni_distribution(0.0,1.0);

// Integer exponentiation
int ipow(int base, int exp)
{
    int result = 1;
    for (;;)
    {
        if (exp & 1)
            result *= base;
        exp >>= 1;
        if (!exp)
            break;
        base *= base;
    }
    return result;
}

// Reads a spin
bool bit(int number, int position)
{
    return ((number&(1UL<<position))>>position);
}

// Reads two consecutive spins (pbc apply)
int twobits(int number, int position, int L)
{
    if(position<L-1)
        return ((number&(3UL<<position))>>position);
    else
        return (bit(number,0)<<1)|(bit(number,L-1));
}

// Sets spin to "up"
int setbit1(int number, int position)
{
    return (number|(1UL<<position));
}

// Sets spin to "down"
int setbit0(int number, int position)
{
    return (number&~(1UL<<position));
}

// Sets two consecutive spins (pbc apply)
int setbit(int number, int position, int whattoset, int L)
{
    int x = number;
    switch (whattoset)
    {
        case 0:
            x=setbit0(x,position%L);
            x=setbit0(x,(position+1)%L);
            break;

        case 1:
            x=setbit1(x,position%L);
            x=setbit0(x,(position+1)%L);
            break;

        case 2:
            x=setbit0(x,position%L);
            x=setbit1(x,(position+1)%L);
            break;

        case 3:
            x=setbit1(x,position%L);
            x=setbit1(x,(position+1)%L);
            break;

        default:
            break;
    }
    return x;
}

// Gaussian factor in front of measurement
double constantC(double x, double lambda, double Delta)
{
    return 1./sqrt(sqrt(pi))/sqrt(Delta)*exp(-(x+lambda)*(x+lambda)/2./Delta/Delta);
}

// Probability density of measuring x
double probMeasurement(complex<double>* psi, int L, double x, double lambda, double Delta, int position)
{
    double sumUp   = 0;
    double sumDown = 0;
    for(int i = 0; i < ipow(2,L); i++)
    {
        if(bit(i,position))
            sumUp   += norm(psi[i]);
        else
            sumDown += norm(psi[i]);
    }
    sumUp   *= constantC(x,-lambda, Delta)*constantC(x,-lambda, Delta);
    sumDown *= constantC(x, lambda, Delta)*constantC(x, lambda, Delta);
    return sumUp + sumDown;
}

// Generate a random number with PDF given by probMeasurement()
double numberMeasurement(complex<double>* psi, int L, double lambda, double Delta, int position)
{
    double sumUp   = 0;
    double sumDown = 0;
    for(int i = 0; i < ipow(2,L); i++)
    {
        if(bit(i,position))
            sumUp   += norm(psi[i]);
        else
            sumDown += norm(psi[i]);
    }
    double number = uni_distribution(generator);

    double xmin = -1000, xmax = 1000, xmid;
    int ii=0;
    while(xmax-xmin > 1.e-15 && ii < 100)
    {
        ii++;
        xmid = (xmax+xmin)/2.0;
        if(
            // CDF corresponding to PDF given by probMeasurement()
            (0.5*sumUp*(1.0 + erf((xmid - lambda) / Delta))) 
            + (0.5*sumDown*(1.0 + erf((xmid + lambda) / Delta))) < number
            )
            xmin = xmid;
        else
            xmax = xmid;
    }
    return xmax;
}

// Apply measurement to psi
void psiMeasurement(complex<double>* psi, int L, double x, double lambda, double Delta, int position)
{
    double c = 1./sqrt(probMeasurement(psi, L, x, lambda, Delta, position));
    for(int i = 0; i < ipow(2,L); i++)
    {
        if(bit(i,position))
            psi[i] *= c * constantC(x,-lambda, Delta);
        else
            psi[i] *= c * constantC(x, lambda, Delta);
    }
    return;
}

// Generate a random CUE matrix
Matrix<complex<double>,4,4> cue()
{
    double number = distribution(generator);
    Matrix<complex<double>,4,4> z;
    
    for(int i = 0; i < 4; i++)
        for(int j = 0; j < 4; j++)
            z(i,j) = complex<double>(distribution(generator),distribution(generator));

    HouseholderQR<Matrix<complex<double>,4,4> > qr(z.rows(),z.cols());
    qr.compute(z);
    Matrix<complex<double>,4,4> r = qr.matrixQR(); // .triangularView<Upper>() to get actual R
    Matrix<complex<double>,4,4> q = qr.householderQ();
    Matrix<complex<double>,4,4> rd;
    
    for(int i = 0; i < 4; i++)
        rd(i,i) = r(i,i)/abs(r(i,i));
    
    return (q * rd);
}

int main(int argc, char const *argv[])
{
    // System setup
    int    L       = 16;
    double Delta   = 1.0;
    double lambda  = 0.5;
    double p       = 0.5;
    int    steps   = 10000;

    bool   verbose = 0;

    if(argc==5)
    {
        L       = atoi(argv[1]);
        Delta   = atof(argv[2]);
        lambda  = atof(argv[3]);
        p       = atof(argv[4]);
    }
    if(argc==6)
    {
        L       = atoi(argv[1]);
        Delta   = atof(argv[2]);
        lambda  = atof(argv[3]);
        p       = atof(argv[4]);
        steps   = atoi(argv[5]);
    }
    if(argc==7)
    {
        L       = atoi(argv[1]);
        Delta   = atof(argv[2]);
        lambda  = atof(argv[3]);
        p       = atof(argv[4]);
        steps   = atoi(argv[5]);
        verbose = atoi(argv[6]);
    }
    generator.seed(chrono::system_clock::now().time_since_epoch().count());
    int sizeL = ipow(2,L);
    int sizeL2 = ipow(2,L/2);
    complex<double>* psi  = new complex<double>[sizeL];
    complex<double>* psi2 = new complex<double>[sizeL];
    SelfAdjointEigenSolver< MatrixXcd > eigensolver;
    MatrixXcd rhoA(sizeL2,sizeL2);
    Matrix<complex<double>,4,4> U [L];
    double entropy, entropy2, entropy3, eig, xprob, sum;
    complex<double> sumc;
    chrono::steady_clock::time_point begin;
    chrono::steady_clock::time_point end;
    cout << fixed;
    cout.precision(15);
    
    // Start from the Neel state
    psi[(sizeL-1)/3] = complex<double>(1.,0.);

    // Main loop
    for(int loop = 0; loop < steps; loop++)
    {
        // Generate the U matrices
        if(verbose) cout << "Generating U matrices." << endl;
        for(int i = 0; i < L; i++)
            U[i] = cue();

        // Apply U on even pairs
        if(verbose) cout << "Applying U even: " << flush;
        for(int pos = 0; pos < L; pos+=2)
        {
            fill(psi2,psi2+sizeL,0.);
            for(int j = 0; j < 4; j++)
            {
                for(int i = 0; i < sizeL; i++)                
                {
                    psi2[i] += U[pos](twobits(i,pos,L),j)*psi[setbit(i,pos,j,L)];
                }
            }
            copy(psi2,psi2+sizeL,psi);
            if(verbose) cout << "." << flush;
        }
        if(verbose) cout << endl;

        // Apply M
        if(verbose) cout << "Applying M:      " << flush;
        for(int pos = 0; pos < L; pos++)
        {
            if( uni_distribution(generator) < p )
            {
                double prob = numberMeasurement(psi, L, lambda, Delta, pos);
                psiMeasurement(psi, L, prob, lambda, Delta, pos);
            }
            if(verbose) cout << "." << flush;
        }
        if(verbose) cout << endl;

        // Apply U on odd pairs
        if(verbose) cout << "Applying U odd:  " << flush;
        for(int pos = 1; pos < L; pos+=2)
        {
            fill(psi2,psi2+sizeL,0.);
            for(int j = 0; j < 4; j++)
            {
                for(int i = 0; i < sizeL; i++)
                {
                    psi2[i] += U[pos](twobits(i,pos,L),j)*psi[setbit(i,pos,j,L)];
                }
            }
            copy(psi2,psi2+sizeL,psi);
            if(verbose) cout << "." << flush;
        }
        if(verbose) cout << endl;
        
        // Apply M
        if(verbose) cout << "Applying M:      " << flush;
        for(int pos = 0; pos < L; pos++)
        {
            if( uni_distribution(generator) < p )
            {
                xprob = numberMeasurement(psi, L, lambda, Delta, pos);
                psiMeasurement(psi, L, xprob, lambda, Delta, pos);
            }
            if(verbose) cout << "." << flush;
        }
        if(verbose) cout << endl;
        
        // Calculate reduced density matrix
        if(verbose) cout << defaultfloat;
        if(verbose) cout << "Calculating rhoA... " << flush;
        if(verbose) begin = chrono::steady_clock::now();
        rhoA.setZero();
        for(int b1 = 0; b1 < sizeL2; b1++)
        {
            for(int b2 = 0; b2 < sizeL2; b2++)
            {
                sumc = 0;
                for(int b3 = 0; b3 < sizeL2; b3++)
                {
                    sumc += conj(psi[b1*sizeL2 + b3]) * psi[b2*sizeL2 + b3];
                }
                rhoA(b1,b2) = sumc;
            }
        }
        if(verbose) end   = chrono::steady_clock::now();
        if(verbose) cout << "Completed in " << (double)(chrono::duration_cast<chrono::milliseconds>(end - begin).count())/1000. << "s." << endl;
        
        // Calculate eigenvalues of reduced density matrix
        if(verbose) cout << "Calculating rhoA eigenvalues... " << flush;
        if(verbose) begin = chrono::steady_clock::now();
        eigensolver.compute(rhoA,EigenvaluesOnly);
        if(verbose) end   = chrono::steady_clock::now();
        if(verbose) cout << "Completed in " << (double)(chrono::duration_cast<chrono::milliseconds>(end - begin).count())/1000. << "s." << endl;

        // Calculate entropy
        if(verbose) cout << fixed;
        if(verbose) cout << "Von Neumann entropy: " << flush;
        entropy  = 0;
        entropy2 = 0;
        entropy3 = 0;
        for(int j = 0; j < sizeL2; j++)
        {
            eig = real(eigensolver.eigenvalues()(j));
            if(eig>1.e-15)
            {
                entropy  -= eig * log(eig);
                entropy2 += eig*eig;
                entropy3 += eig*eig*eig;
            }
        }
        entropy2 = -log(entropy2);
        entropy3 = -0.5 * log(entropy3);
        if(!verbose) cout << loop << "\t" << flush;
        cout << entropy << "\t" << flush;
        if(verbose) cout << endl << "2-Rényi entropy:     " << flush;
        cout << entropy2 << "\t" << flush;
        if(verbose) cout << endl << "3-Rényi entropy:     " << flush;
        cout << entropy3 << endl;
        
        // Check the norm of the wavefunction
        if(verbose)
        {
            cout << "Norm check:          " << flush;
            sum = 0;
            for(int i = 0; i < sizeL; i++)
                sum += norm(psi[i]);
            cout << sum << endl;
        }
    }
    
    return 0;
}