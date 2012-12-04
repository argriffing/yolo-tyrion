
from StringIO import StringIO

import sympy


def sym_beta_expectation(concentration):
    return beta_expectation(concentration, concentration)

def sym_beta_variance(concentration):
    return beta_variance(concentration, concentration)

def sym_beta_covariance(concentration):
    return beta_covariance(concentration, concentration)

def sym_beta_expectation_of_product(concentration):
    return beta_expectation_of_product(concentration, concentration)


def beta_expectation(alpha, beta):
    """
    This is the expectation of X.
    It is from wikipedia.
    """
    return alpha / (alpha + beta)

def beta_variance(alpha, beta):
    """
    This is the variance of X.
    It is from wikipedia.
    """
    a = alpha * beta
    b = (alpha + beta)*(alpha + beta)*(alpha + beta + 1)
    return a / b

def beta_covariance(alpha, beta):
    """
    It is the covariance between X and 1-X.
    It is from wikipedia.
    """
    return -beta_variance(alpha, beta)

def beta_expectation_of_product(alpha, beta):
    """
    This is the expectation of X*(1-X).
    It is from wikipedia.
    """
    a = alpha * beta
    b = (alpha + beta) * (alpha + beta + 1)
    return a / b

def check_jeff(x, j):
    out = StringIO()
    print >> out, 'sympy value:', x
    print >> out, 'error:', sympy.simplify(x - j)
    return out.getvalue().rstrip()

def main():
    N = sympy.Symbol('N')
    mu = sympy.Symbol('mu')
    diag_concentration = 4*N*mu
    diag_expectation = sym_beta_expectation(diag_concentration)
    diag_variance = sym_beta_variance(diag_concentration)
    diag_diag_expectation = diag_variance + diag_expectation*diag_expectation
    diag_comp_expectation = sym_beta_expectation_of_product(diag_concentration)
    #
    x = diag_expectation
    j = sympy.Rational(1, 2)
    print 'E(X_1 + X_4)'
    print check_jeff(x, j)
    print
    #
    x = diag_variance
    j = 1 / (4 * (8*N*mu + 1) )
    print 'Var(X_1 + X_4)'
    print check_jeff(x, j)
    print
    #
    x = diag_diag_expectation
    j = (4*N*mu + 1) / (2 * (8*N*mu + 1) )
    print 'E((X_1 + X_4)^2)'
    print check_jeff(x, j)
    print
    #
    x = sym_beta_covariance(diag_concentration)
    j = -1/(4 * (8*N*mu + 1) )
    print 'Cov(X_1 + X_4, X_2 + X_3)'
    print check_jeff(x, j)
    print 
    #
    x = diag_comp_expectation
    j = (2*N*mu) / (8*N*mu + 1)
    print 'E((X_1 + X_4)(X_2 + X_3))'
    print check_jeff(x, j)
    print
    #
    side_concentration = 2*N*mu
    side_expectation = sym_beta_expectation(side_concentration)
    side_variance = sym_beta_variance(side_concentration)
    side_side_expectation = side_variance + side_expectation*side_expectation
    side_comp_expectation = sym_beta_expectation_of_product(side_concentration)
    #
    ex1x1 = sympy.Symbol('ex1x1')
    ex1x4 = sympy.Symbol('ex1x4')
    ex1x2 = sympy.Symbol('ex1x2')
    equations = [
            sympy.Eq(2*ex1x1 + 2*ex1x4, diag_diag_expectation),
            sympy.Eq(2*ex1x1 + 2*ex1x2, side_side_expectation),
            sympy.Eq(4*ex1x2, diag_comp_expectation),
            ]
    #print sympy.solve(equations, [N, mu])
    ex1x2_j = (N*mu) / (2*(8*N*mu  +1))
    ex1x1_j = (8*(N*mu)*(N*mu) + 8*N*mu + 1) / (4*(4*N*mu + 1)*(8*N*mu + 1))
    ex1x4_j = (2*(N*mu)*(N*mu)) / ((4*N*mu + 1)*(8*N*mu + 1))
    #
    print 'putative value of E(X_1 * X_2)'
    print ex1x2_j
    print 
    print 'putative value of E(X_1 * X_1)'
    print ex1x1_j
    print 
    print 'putative value of E(X_1 * X_4):'
    print ex1x4_j
    print 
    #
    lhs = 2*ex1x1_j + 2*ex1x4_j
    rhs = diag_diag_expectation
    print 'checking equation:'
    print '2 * E(X_1 * X_1) + 2*E(X_1 * X_4) = E((X_1 + X_4)^2)'
    print 'lhs:'
    print lhs
    print 'rhs:'
    print rhs
    print 'error:'
    print sympy.simplify(lhs - rhs)
    print
    #
    lhs = 2*ex1x1_j + 2*ex1x2_j
    rhs = side_side_expectation
    print 'checking equation:'
    print '2 * E(X_1 * X_1) + 2*E(X_1 * X_2) = E((X_1 + X_2)^2)'
    print 'lhs:'
    print lhs
    print 'rhs:'
    print rhs
    print 'error:'
    print sympy.simplify(lhs - rhs)
    print
    #
    lhs = 4 * ex1x2_j
    rhs = diag_comp_expectation
    print 'checking equation:'
    print '4 * E(X_1 * X_2) = E((X_1 + X_4)*(X_2 + X_3))'
    print 'lhs:'
    print lhs
    print 'rhs:'
    print rhs
    print 'error:'
    print sympy.simplify(lhs - rhs)
    print
    #
    cx1x2_j = -1/(16*(8*N*mu + 1))
    vx1_j = (20*N*mu + 3) / (16*(4*N*mu + 1)*(8*N*mu + 1))
    cx1x4_j = -(12*N*mu + 1) / (16*(4*N*mu + 1)*(8*N*mu + 1))
    ex1_j = sympy.Rational(1, 4)
    ex2_j = sympy.Rational(1, 4)
    ex4_j = sympy.Rational(1, 4)
    #
    # check some equations
    lhs = vx1_j
    rhs = ex1x1_j - ex1_j * ex1_j
    print 'checking equation:'
    print 'Var(X_1) = E(X_1 * X_1) - E(X_1) * E(X_1)'
    print 'lhs:'
    print lhs
    print 'rhs:'
    print rhs
    print 'error:'
    print sympy.simplify(lhs - rhs)
    print
    #
    lhs = cx1x2_j
    rhs = ex1x2_j - ex1_j * ex1_j
    print 'checking equation:'
    print 'Cov(X_1, X_2) = E(X_1 * X_2) - E(X_1) * E(X_2)'
    print 'lhs:'
    print lhs
    print 'rhs:'
    print rhs
    print 'error:'
    print sympy.simplify(lhs - rhs)
    print
    #
    lhs = cx1x4_j
    rhs = ex1x4_j - ex1_j * ex1_j
    print 'checking equation:'
    print 'Cov(X_1, X_4) = E(X_1 * X_4) - E(X_1) * E(X_4)'
    print 'lhs:'
    print lhs
    print 'rhs:'
    print rhs
    print 'error:'
    print sympy.simplify(lhs - rhs)
    print


if __name__ == '__main__':
    main()
