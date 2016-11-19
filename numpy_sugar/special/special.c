#include "numpy_sugar/special.h"

#include <float.h>
#define _USE_MATH_DEFINES
#include <math.h>

#include "ncephes/cprob.h"
#include "ncephes/misc.h"


// Gaussian distribution

double nsugar_normal_pdf(double x)
{
    return exp(nsugar_normal_logpdf(x));
}

double nsugar_normal_cdf(double x)
{
    return ncephes_ndtr(x);
}

double nsugar_normal_icdf(double x)
{
    return ncephes_ndtri(x);
}

double nsugar_normal_sf(double x)
{
    return nsugar_normal_cdf(-x);
}

double nsugar_normal_isf(double x)
{
    return -nsugar_normal_icdf(x);
}

double nsugar_normal_logpdf(double x)
{
    static const double _norm_pdf_logC =
        0.9189385332046726695409688545623794198036193847656250;
    return -(x*x)/ 2.0 - _norm_pdf_logC;
}

/*
* double normal_logcdf(double a)
*
* For a > -20, use the existing ndtr technique and take a log.
* for a <= -20, we use the Taylor series approximation of erf to compute
* the log CDF directly. The Taylor series consists of two parts which we will name "left"
* and "right" accordingly.  The right part involves a summation which we compute until the
* difference in terms falls below the machine-specific EPSILON.
*
* \Phi(z) &=&
*   \frac{e^{-z^2/2}}{-z\sqrt{2\pi}}  * [1 +  \sum_{n=1}^{N-1}  (-1)^n \frac{(2n-1)!!}{(z^2)^n}]
*   + O(z^{-2N+2})
*   = [\mbox{LHS}] * [\mbox{RHS}] + \mbox{error}.
*
*/
double nsugar_normal_logcdf(double x)
{
    // we compute the left hand side of the approx (LHS) in one shot
    double log_LHS;
    // variable used to check for convergence
    double last_total = 0;
    // includes first term from the RHS summation
	double right_hand_side = 1;
    // numerator for RHS summand
	double numerator = 1;
    // use reciprocal for denominator to avoid division
	double denom_factor = 1;
    // the precomputed division we use to adjust the denominator
	double denom_cons = 1.0 / (x * x);
    long sign = 1, i = 0;

    if (x > 6) {
	       return -ncephes_ndtr(-x);     /* log(1+x) \approx x */
    }
    if (x > -20) {
	       return log(ncephes_ndtr(x));
    }
    log_LHS = -0.5 * x * x - log(-x) - 0.5 * log(2 * M_PI);

    while (fabs(last_total - right_hand_side) > DBL_EPSILON) {
    	i += 1;
    	last_total = right_hand_side;
    	sign = -sign;
    	denom_factor *= denom_cons;
    	numerator *= 2 * i - 1;
    	right_hand_side += sign * numerator * denom_factor;
    }
    return log_LHS + log(right_hand_side);
}

double nsugar_normal_logsf(double x)
{
    return nsugar_normal_logcdf(-x);
}

double nsugar_lgamma(double x)
{
  return lgamma(x);
}


double nsugar_chi2_sf(int k, double x)
{
  return ncephes_chdtrc(k, x);
}


double nsugar_beta_isf(double a, double b, double x)
{
    return ncephes_incbi(a, b, 1.0 - x);
}


double nsugar_logaddexp(double x, double y)
{
    double tmp = x - y;

    if (x == y)
        return x + M_LN2;

    if (tmp > 0)
        return x + log1p(exp(-tmp));
    else if (tmp <= 0)
        return y + log1p(exp(tmp));

    return tmp;
}

double nsugar_logaddexps(double x, double y, double sx, double sy)
{
    double tmp = x - y;

    double sxx = log(fabs(sx)) + x;
    double syy = log(fabs(sy)) + y;

    if (sxx == syy)
    {
        if (sx * sy > 0)
            return sxx + M_LN2;
        return -DBL_MAX;
    }

    if (sx > 0 && sy > 0)
    {
        if (tmp > 0)
            return sxx + log1p((sy/sx) * exp(-tmp));
        else if (tmp <= 0)
            return syy + log1p((sx/sy) * exp(tmp));
    }
    else if (sx > 0)
        return sxx + log1p((sy/sx) * exp(-tmp));
    else
        return syy + log1p((sx/sy) * exp(tmp));
    return tmp;
}

double nsugar_logaddexpss(double x, double y, double sx,
                         double sy, double* sign)
{
  // printf("sx sy: %.30f %.30f\n", sx, sy); fflush(stdout);
  double sxx = log(fabs(sx)) + x;
  double syy = log(fabs(sy)) + y;

  // printf("!!\n"); fflush(stdout);

  if (sxx == syy)
  {
    if (sx * sy > 0)
    {
      if (sx > 0)
        *sign = +1.0;
      else
        *sign = -1.0;
      return sxx + M_LN2;
    }
    else
    {
      *sign = 1.0;
      return -DBL_MAX;
    }
  }

  // printf("x-y: %.30f\n", x-y); fflush(stdout);
  // printf("sxx-syy: %.30f\n", sxx-syy); fflush(stdout);

  if (sxx > syy)
  {
    if (sx >= 0.0)
      *sign = +1.0;
    else
      *sign = -1.0;
  }
  else
  {
    if (sy >= 0.0)
      *sign = +1.0;
    else
      *sign = -1.0;
  }

  sx *= *sign;
  sy *= *sign;
  return nsugar_logaddexps(x, y, sx, sy);
}

double nsugar_logbinom(double N, double K)
{
    return -ncephes_lbeta(1 + N - K, 1 + K) - log1p(N);
}
