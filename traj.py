import math
import random
import matplotlib.pyplot as plt
import numpy as np

amax = 2. # accelération maximale
x = np.array([0])
y = np.array([0])
t = np.array([0])
v = np.array([10])
a = np.array([0])
delta_t = 0.01
for i in range(1,1000):
    delta_x = v[-1]*delta_t*math.cos(a[-1])
    delta_y = v[-1]*delta_t*math.sin(a[-1])
    delta_v = random.randint(-100,100)/1000.
    delta_a = random.randint(0,5)/1000.
    #delta_a= i/200*math.pi
    x = np.append(x,x[-1] + delta_x)
    y = np.append(y,y[-1] + delta_y)
    t = np.append(t,t[-1] + delta_t)
    v = np.append(v,max(v[-1] + delta_v,0))
    a = np.append(a,a[-1] + delta_a)

# Première dérivée
dx = np.gradient(x)
dy = np.gradient(y)

# Deuxième dérivée
ddx = np.gradient(dx)
ddy = np.gradient(dy)

# Courbure
courbure = np.divide(np.abs(dx * ddy - dy * ddx) , (dx**2 + dy**2)**(3/2))

#plt.style.use('_mpl-gallery')
fig, ax = plt.subplots(2,2)

ax[0,0].plot(x, y)
ax[0,0].scatter(x[0], y[0],color='red')
ax[0,0].set(xlim=(x.min()-5, x.max()+5), ylim=(y.min()-5, y.max()+5))
ax[0,0].set_title('trajectoire')


ax[0,1].plot(t, v)
#ax[0,1].set(xlim=(t.min()-5, t.max()+5), ylim=(v.min()-5, v.max()+5))
ax[0,1].set_title('vitesse')

ax[1,0].plot(t, a)
ax[1,0].set_title('angle')

ax[1,1].plot(t, courbure)
ax[1,1].set_title('courbure')

plt.show()
