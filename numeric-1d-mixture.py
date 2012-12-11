"""
Try to numerically determine a mixture model that meets certain criteria.

The mixture is a distribution over a 3-simplex.
But it is constrained so that all of the mass is at the corners and edges.
It is further constrained by some symmetries,
and some of the marginal distributions are known.
The idea of this script is to use the parameterization
to enforce the symmetries by variable and constraint elimination,
and to use least squares to search for the mixture which
has the desired marginal distributions.
It may be possible that this model is not general enough to simultaneously
meet all of the constraints.
"""

#NOTE: this is abandoned because everyone already believes
#NOTE: that the symmetries and the marginal beta distributions
#NOTE: are insufficient to define the joint density

def main():
    pass

if __name__ == '__main__':
    main()

