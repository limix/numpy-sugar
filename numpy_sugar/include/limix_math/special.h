#ifndef SPECIAL_H
#define SPECIAL_H

double lmath_chi2_sf(int k, double x);
double lmath_lgamma(double x);
double lmath_normal_pdf(double x);
double lmath_normal_cdf(double x);
double lmath_normal_icdf(double x);
double lmath_normal_sf(double x);
double lmath_normal_isf(double x);
double lmath_normal_logpdf(double x);
double lmath_normal_logcdf(double x);
double lmath_normal_logsf(double x);

double lmath_beta_isf(double a, double b, double x);

double lmath_logaddexp(double x, double y);
double lmath_logaddexps(double x, double y, double sx, double sy);
double lmath_logaddexpss(double x, double y, double sx, double sy,
                         double* sign);

double lmath_logbinom(double N, double K);

#endif
