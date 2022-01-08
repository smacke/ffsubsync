"""Python program for golden section search (straight-up copied from Wikipedia).
   This implementation reuses function evaluations, saving 1/2 of the evaluations per
   iteration, and returns a bounding interval."""
import logging
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


invphi = (math.sqrt(5) - 1) / 2  # 1 / phi
invphi2 = (3 - math.sqrt(5)) / 2  # 1 / phi^2


def gss(f, a, b, tol=1e-4):
    """Golden-section search.

    Given a function f with a single local minimum in
    the interval [a,b], gss returns a subset interval
    [c,d] that contains the minimum with d-c <= tol.

    Example:
    >>> f = lambda x: (x-2)**2
    >>> a = 1
    >>> b = 5
    >>> tol = 1e-5
    >>> (c,d) = gss(f, a, b, tol)
    >>> print(c, d)
    1.9999959837979107 2.0000050911830893
    """

    (a, b) = (min(a, b), max(a, b))
    h = b - a
    if h <= tol:
        return a, b

    # Required steps to achieve tolerance
    n = int(math.ceil(math.log(tol / h) / math.log(invphi)))
    logger.info(
        "About to perform %d iterations of golden section search to find the best framerate",
        n,
    )

    def f_wrapped(x, is_last_iter):
        try:
            return f(x, is_last_iter)
        except TypeError:
            return f(x)

    c = a + invphi2 * h
    d = a + invphi * h
    yc = f_wrapped(c, n == 1)
    yd = f_wrapped(d, n == 1)

    for k in range(n - 1):
        if yc < yd:
            b = d
            d = c
            yd = yc
            h = invphi * h
            c = a + invphi2 * h
            yc = f_wrapped(c, k == n - 2)
        else:
            a = c
            c = d
            yc = yd
            h = invphi * h
            d = a + invphi * h
            yd = f(d, k == n - 2)

    if yc < yd:
        return a, d
    else:
        return c, b
