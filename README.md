# calculate read-based phase
There are 2 haplotypes that a child inherits and we can infer the pattern of
inheritance by finding the configuration of parental transmission that results
in the minimum total distance (ideally 0) with the 2 child haplotypes.

There are 8 pairwise distances between the 3 different pairs of haplotypes.
In addition, there are 8 possible configurations from Mendelian transmission,
e.g., no uniparental disomy.

Let C_0 = [c0, c1] be a vector of the child haplotypes and
M = [[m0, m0, m1, m1], [p0, p1, p0, p1]] be a matrix where columns represent
configurations of maternal and paternal haplotype transmission.

C_0 * M, where the hamming distance replaces the product between pairs of elements
in the matrix multiplication, gives a vector of distances for all possible
configurations where c0 is maternally inherited and c1 is paternally inherited.

Let C_1 = [c1, c0], then C_1 x M gives the vector of distances for all possible
configurations where c0 is paternally inherited and c1 is maternally inherited.

The parent of origin for the mutation then corresponds to the pattern of inheritance
of the haplotype onto which it was read-phased.