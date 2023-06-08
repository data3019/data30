# [1] Add matrix using dot()

import numpy as np
A = np.array([[1, 2], [3, 4], [5, 6]])
A
B = np.array([[2, 5], [7, 4], [4, 3]])
B
# Add matrices A and B
C = A + B
print(C)

# 2
# Add matrix using add()
# Source Code:
import numpy as np
A = np.array([[1, 2], [3, 4], [5, 6]])
B = np.array([[2, 5], [7, 4], [4, 3]])
# Add matrices A and B
C = np.add(A, B)
print(C)

# 3
# Multiply using dot()
# Source Code:
# importing the module
import numpy as np
# creating two matrices
p = [[1, 2], [2, 3], [4, 5]]
q = [[4, 5, 1], [6, 7, 2]]
print("Matrix p :")
print(p)
print("Matrix q :")
print(q)
# computing product
result = np.dot(p, q)
# printing the result
print("The matrix multiplication is :")
print(result)

# [2a] Multiply using dot()
# Source Code:
import numpy as np
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(A)
B = np.array([[2, 7], [1, 2], [3, 6]])
print(B)
C = A.dot(B)
print(C)


# [3a] Linear Combination
# Source Code:
import numpy as np
x = np.array([[0, 1, 1],
 [1, 1, 0],
 [1, 0, 1]])
y = ([3.65, 1.55, 3.42])
scalars = np.linalg.solve(x, y)
print(scalars)

# [3b] Linear Combination
# Source Code:
import numpy as np
import matplotlib.pyplot as plt
def plotVectors(vecs, cols, alpha =1 ):
 plt.figure()
 plt.axvline(x=0, color = '#A9A9A9', zorder = 0)
 for i in range(len(vecs)):
 x = np.concatenate([[0,0],vecs[i]])
 plt.quiver([x[0]],
 [x[1]],
 [x[2]],
 [x[3]],
 angles='xy', scale_units='xy', scale=1,color=cols[i], alpha=alpha)
orange= '#FF9A13'
blue= '#1190FF'
plotVectors([[1,3],[2,1]],[orange,blue])
plt.xlim(0,5)
plt.ylim(0,5)
plt.show()


# [4a] Linear Equation
# Source Code:
import numpy as np
A = np.array([[20, 10], [17, 22]])
B = np.array([350, 500])
R = np.linalg.solve(A,B)
x, y = np.linalg.solve(A,B)
print(R)
print("x =", x)
print("y =", y)

# Linear Equation
# Source Code:
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(-10,10)
y = 2*x
y1 = -x + 3
plt.figure()
plt.plot(x,y)
plt.plot(x,y1)
plt.xlim(0,3)
plt.ylim(0,3)
plt.axvline(x=0, color='grey')
plt.show()

# [5a] Norm one-dimensional
# Source Code:
# import library
import numpy as np
# initialize vector
oned = np.arange(10)
# compute norm of vector
manh_norm = np.linalg.norm(oned)
print("Manhattan norm:")
print(manh_norm)

# [5b] Norm two-dimensional
# Source Code:
# import library
import numpy as np
# initialize matrix
twod = np.array([[ 1, 2, 3],
 [4, 5, 6]])
# compute norm of matrix
eucl_norm = np.linalg.norm(twod)
print("Euclidean norm:")
print(eucl_norm)


# [5c] Norm three-dimensional
# Source Code:
# import library
import numpy as np
# initialize matrix
threed = np.array([[[ 1, 2, 3],
 [4, 5, 6]],[[ 11, 12, 13],
 [14, 15, 16]]])
# compute norm of matrix
mink_norm = np.linalg.norm(threed)
print("Minkowski norm:")
print(mink_norm)

#
# [5] Norm
# Source Code
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
u = [0,0,1,6]
v = [0,0,4,2]
u_bis = [1,6,v[2],v[3]]
w = [0,0,5,8]
plt.quiver([u[0], u_bis[0], w[0]],
 [u[1], u_bis[1], w[1]],
 [u[2], u_bis[2], w[2]],
 [u[3], u_bis[3], w[3]],
 angles = 'xy', scale_units = 'xy', scale = 1, color = sns.color_palette())
plt.xlim(-2,6)
plt.ylim(-2,9)
plt.axvline(x=0, color='grey')
plt.axhline(y=0, color='grey')
plt.text(-1, 3.5, r'$||\vec{u}||$', color = sns.color_palette()[0], size =20)
plt.text(2.5, 7.5, r'$||\vec{v}||$', color = sns.color_palette()[1], size =20)
plt.text(2, 2, r'$||\vec{u}+\vec{v}||$', color = sns.color_palette()[2], size =20)
plt.show()


# [6a] Symmetric Matrix
import numpy as np

M = np.array([[2,3,4], [3,45,8], [4,8,78]])
print("---Matrix M---\n", M)
# Transposing the Matrix M
print('\n\nTranspose as M.T----\n', M.T)
if M.T.all() == M.all():
 print("--------> Transpose is equal to M!! -----> It is a Symmetric Matrix")
else:
 print("---------> Transpose is not equal o M!! ------> It is not a Symmetric Matrix")

# Symmetric Matrix
# Source Code:
import numpy as np
A = np.array([[2,4,-1],[4,-8,0],[-1,0,3]])
print(A)
print(A.T)

# [B] Aim: Performing matrix multiplication and finding eigen vectors and eigen values using
# TensorFlow.
# Source Code:
import tensorflow as tf
print("Matrix Multiplication Demo")
x=tf.constant([1,2,3,4,5,6],shape=[2,3])
print(x)
y=tf.constant([7,8,9,10,11,12],shape=[3,2])
print(y)
z=tf.matmul(x,y)
print("Product:",z)
e_matrix_A=tf.random.uniform([2,2],minval=3,maxval=10,dtype=tf.float32,name="matrixA
")
print("Matrix A:\n{}\n\n".format(e_matrix_A))
eigen_values_A,eigen_vectors_A=tf.linalg.eigh(e_matrix_A)
print("Eigen Vectors:\n{}\n\nEigen
Values:\n{}\n".format(eigen_vectors_A,eigen_values_A))