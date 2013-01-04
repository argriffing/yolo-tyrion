"""
Reformat text representations of matrices.

This is a one-off script intended for use as a filter within vim.
It tries to convert a mathematica matrix into a tex matrix.
"""

import argparse
import sys
import ast
import string

import numpy as np
import sympy

def main(args):

    # define a character translation table
    intab = '{}'
    outtab = '[]'
    deltab = '>'
    trantab = string.maketrans(intab, outtab)

    # get translated lines
    filtered_lines = []
    for line in sys.stdin:
        line = line.strip().translate(trantab, deltab)
        if not line:
            continue
        filtered_lines.append(line)

    # construct a single line of text
    # beginning with the first open bracket
    # and ending with the last close bracket
    s = ''.join(filtered_lines)
    s_trimmed = s[s.find('[') : s.rfind(']') + 1]

    # attempt to construct the matrix as a list of lists of numbers
    arr = ast.literal_eval(s_trimmed)

    # possibly use sympy to invert the matrix
    if args.sympy_inverse:
        nrows = len(arr)
        ncols = len(arr[0])
        flat_arr = [x for row in arr for x in row]
        M = sympy.Matrix(nrows, ncols, flat_arr)
        arr = M.inv().tolist()

    # define the scaling factor
    sf = 1
    try:
        sf = int(args.scale)
    except ValueError:
        sf = float(args.scale)

    # attempt to write the matrix in tex form
    for row in arr:
        print ' & '.join(str(x * sf) for x in row) + r' \\'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--sympy-inverse', action='store_true',
            help='use sympy to compute the inverse of the matrix')
    parser.add_argument(
            '--scale',
            help='scale the matrix by this value')
    main(parser.parse_args())

