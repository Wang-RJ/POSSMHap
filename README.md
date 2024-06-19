# calculate read-based phase

There are 2 haplotypes that a child inherits and we can infer the pattern of
inheritance by finding the configuration of parental transmission that results
in the minimum total distance (ideally 0) with the 2 child haplotypes.

There are 8 pairwise distances between the 3 different pairs of haplotypes.
In addition, there are 8 possible configurations from Mendelian transmission,
e.g., no uniparental disomy.

Let C<sub>0</sub> = [c<sub>0</sub>, c<sub>1</sub>] be a vector of the child haplotypes and
M = [[m<sub>0</sub>, m<sub>0</sub>, m<sub>1</sub>, m<sub>1</sub>], [p<sub>0</sub>, p<sub>1</sub>, p<sub>0</sub>, p<sub>1</sub>]] be a matrix where columns represent
configurations of maternal and paternal haplotype transmission.

C<sub>0</sub> * M, where the hamming distance replaces the product between pairs of elements
in the matrix multiplication, gives a vector of distances for all possible
configurations where c<sub>0</sub> is maternally inherited and c<sub>1</sub> is paternally inherited.

Let C<sub>1</sub> = [c<sub>1</sub>, c<sub>0</sub>], then C<sub>1</sub> * M gives the vector of distances for all possible
configurations where c<sub>0</sub> is paternally inherited and c<sub>1</sub> is maternally inherited.

The parent of origin for the mutation then corresponds to the pattern of inheritance
of the haplotype onto which it was read-phased.
