import math
import random
import matplotlib.pyplot as plt
import numpy as np

amax = 2. # accelération maximale
x = np.array([0])
y = np.array([0])
angle = 0.
vitesse = 1.
dt = 0.01
for i in range(1000):
    delta_x = vitesse*dt*math.cos(angle)
    delta_y = vitesse*dt*math.sin(angle)
    x = np.append(x,x[-1] + delta_x)
    y = np.append(y,y[-1] + delta_y)
    delta_vitesse = random.randint(-100,100)/100.*amax
    delta_angle = random.randint(-100,100)/100.
    vitesse += delta_vitesse
    angle += delta_angle

#plt.style.use('_mpl-gallery')
fig, ax = plt.subplots()

ax.plot(x, y)
ax.scatter(x[0], y[0],color='red')
ax.set(xlim=(x.min()-5, x.max()+5), ylim=(y.min()-5, y.max()+5))
plt.show()

