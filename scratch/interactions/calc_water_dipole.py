
doh = 0.1
dhh = 0.1633

q = -0.8476

# height in y
h = np.sqrt(doh**2 - (0.5*dhh)**2)

positions = np.array([[0,0,0],
                      [-0.5*dhh, h, 0],
                      [0.5*dhh, h, 0]])

v1 = positions[1] - positions[0]
v2 = positions[2] - positions[0]

dipole = (-0.5*q)*v1 + (-0.5*q)*v2