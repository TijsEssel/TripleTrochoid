import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation

# ========================
# Instelbare parameters
# ========================
R1 = 6.0                 # straal
R2 = 8.0
R3 = 1.0
#parameter om de relatieve snelheid van R3 en R2 te bepalen
a=7/12

f1=1*np.pi
f0=0*np.pi

up=10
vp=0


# =================================================
# Tussenresultaten betreffende aantal omwentelingen
# =================================================

ggd=np.gcd(int(np.abs(R1)),int(np.abs(R2)))
#print("ggd =",ggd)
ggd2=np.gcd(int(np.abs(R2)),int(np.abs(R2)))
#print("ggd2 =",ggd2)
basisomwentelingen=int(np.abs(R1))/ggd
#print("basisomwentelingen = ",basisomwentelingen)
omwentelingen=7*np.abs(R1)*basisomwentelingen*int(np.abs(R3))*int(np.abs(R2))/ggd*int(np.abs(R3))

duur_ms = 5000*omwentelingen          # totale animatieduur in milliseconden
fps = 60                # frames per seconde


# Aantal frames en t-waarden
N = int(fps * (duur_ms / 1000))     # totaal aantal frames
t = np.linspace(0, omwentelingen*2*np.pi, N)      # t van 0 tot 2π
#onderstaande zorgt bij a=1 er voor dat cirkel 3 een cirkelbeweging maakt rondom cirkel 1
#de waarde van R2 heeft geen invloed op het aantal omwentelingen
#onderstaande zorgt bij a=0 er voor dat R3 geen invloed heeft op aantal omwentelingen omdat
#R3 niet rolt over R2

t1=-a*(R1/R2)*t

#deze parametervergelijking toont het middelpunt van cirkel 2 die rond cirkel 1 rolt
xmR2 = (R1 + R2) * np.cos(t+f0)+ np.cos((1+(R1 / R2)) * t)*0 - np.sin((1+(R1 / R2)) * t)*0 
ymR2 = (R1 + R2) * np.sin(t+f0)+ np.sin((1+(R1 / R2)) * t)*0 + np.cos((1+(R1 / R2)) * t)*0 

#deze parametervergelijking toon hoe het startpunt op cirkel 2 rond cirkel 1 rolt
xstraalR2=(R1 + R2) * np.cos(t+f0) + np.cos((1+(R1 / R2)) * t)*(R2) - np.sin((1+(R1 / R2)) * t)*0 
ystraalR2=(R1 + R2) * np.sin(t+f0) + np.sin((1+(R1 / R2)) * t)*(R2) + np.cos((1+(R1 / R2)) * t)*0 

#Center van R3
ucenter=(R2 + R3) * np.cos(t1+f1) + np.cos((1+(R2 / R3)) * t1)*0 - np.sin((1+(R2 / R3)) * t1)*0
vcenter=(R2 + R3) * np.sin(t1+f1) + np.sin((1+(R2 / R3)) * t1)*0 + np.cos((1+(R2 / R3)) * t1)*0

xmR3=(R1 + R2) * np.cos(t+f0) + np.cos((1+(R1 / R2)) * t)*ucenter - np.sin((1+(R1 / R2)) * t)*vcenter
ymR3=(R1 + R2) * np.sin(t+f0) + np.sin((1+(R1 / R2)) * t)*ucenter + np.cos((1+(R1 / R2)) * t)*vcenter

#straal R3
ustraal=(R2 + R3) * np.cos(t1+f1) + np.cos((1+(R2 / R3)) * t1)*(R2+R3) - np.sin((1+(R2 / R3)) * t1)*0
vstraal=(R2 + R3) * np.sin(t1+f1) + np.sin((1+(R2 / R3)) * t1)*(R2+R3) + np.cos((1+(R2 / R3)) * t1)*0

xstraalR3=(R1 + R2) * np.cos(t+f0) + np.cos((1+(R1 / R2)) * t)*ustraal - np.sin((1+(R1 / R2)) * t)*vstraal
ystraalR3=(R1 + R2) * np.sin(t+f0) + np.sin((1+(R1 / R2)) * t)*ustraal + np.cos((1+(R1 / R2)) * t)*vstraal

#punt voor het tracé up/vp
u=(R2 + R3) * np.cos(t1+f1) + np.cos((1+(R2 / R3)) * t1)*up - np.sin((1+(R2 / R3)) * t1)*vp
v=(R2 + R3) * np.sin(t1+f1) + np.sin((1+(R2 / R3)) * t1)*up + np.cos((1+(R2 / R3)) * t1)*vp

x=(R1 + R2) * np.cos(t+f0) + np.cos((1+(R1 / R2)) * t)*u - np.sin((1+(R1 / R2)) * t)*v
y=(R1 + R2) * np.sin(t+f0) + np.sin((1+(R1 / R2)) * t)*u + np.cos((1+(R1 / R2)) * t)*v


# ========================
# Figuur en assen
# ========================
fig, ax = plt.subplots(figsize=(10, 10))
#ax.set_aspect('equal', adjustable='box')

# Mooie marge rondom de cirkel
m = 1.05 *  (1*np.abs(R1)+2*np.abs(R2)+2*np.abs(R3)+2*np.abs(up))
ax.set_xlim( - m,  + m)
ax.set_ylim( - m,  + m)
circle = patches.Circle((0, 0), radius=R1, fill=False, edgecolor='black', linewidth=1)
ax.add_patch(circle)

circleR2 = patches.Circle((xmR2[0], ymR2[0]), R2, fill=False, ec='C2', lw=1, animated=True)
ax.add_patch(circleR2)

circleR3 = patches.Circle((xmR3[0], ymR3[0]), R3, fill=False, ec='C0', lw=1, animated=True)
ax.add_patch(circleR3)

s=t
s1=t1
ug=(R2 + R3) * np.cos(s1+f1) + np.cos((1+(R2 / R3)) * s1)*up - np.sin((1+(R2 / R3)) * s1)*vp
vg=(R2 + R3) * np.sin(s1+f1) + np.sin((1+(R2 / R3)) * s1)*up + np.cos((1+(R2 / R3)) * s1)*vp

xg=(R1 + R2) * np.cos(s+f0) + np.cos((1+(R1 / R2)) * s)*ug - np.sin((1+(R1 / R2)) * s)*vg
yg=(R1 + R2) * np.sin(s+f0) + np.sin((1+(R1 / R2)) * s)*ug + np.cos((1+(R1 / R2)) * s)*vg

plt.plot(xg, yg, color='lightgrey', linewidth='0.2')


#ax.set_xlabel('x')
#ax.set_ylabel('y')
ax.grid(True, ls='--', alpha=0.4)

# Statische referentie: volledige cirkel
#ax.plot(x, y, color='lightgray', lw=2, label='cirkel')

# Middelpunt markeren (optioneel)
#ax.plot(a, b, marker='+', color='k', ms=8)

# Dynamische objecten (worden per frame geüpdatet)
(trace,) = ax.plot([], [], color='C2', lw=1)
(trace2,) = ax.plot([], [], color='blue', lw=0.5)
(point,) = ax.plot([], [], 'o', color='C3', ms=4)
(middelpuntR2,)=ax.plot([],[],'o', color='C2',ms=4)
(middelpuntR3,)=ax.plot([],[],'o', color='C0',ms=4)
(straalR2,) = ax.plot([], [], color='C2', lw=2)
(straalR1,) = ax.plot([], [], color='black', lw=2)
(straalR3,) = ax.plot([], [], color='blue', lw=2)

#ax.legend(loc='lower center', ncol=3, frameon=False)

# ========================
# Animatiefuncties
# ========================
def init():
    """Zet alle dynamische objecten leeg aan het begin."""
    trace.set_data([], [])
    trace2.set_data([], [])
    point.set_data([], [])
    middelpuntR2.set_data([], [])
    middelpuntR3.set_data([], [])
    circleR2.center = (xmR2[0], ymR2[0])
    circleR3.center = (xmR3[0], ymR3[0])
    straalR2.set_data([], [])
    straalR1.set_data([], [])
    straalR3.set_data([], [])
    #return trace, trace2, point, middelpuntR2, circleR2, circleR3, straalR2, straalR1, straalR3
    return trace2, point, middelpuntR2, circleR2, circleR3, straalR2, straalR1, straalR3
def update(i):
    """Update de animatie voor frame i."""
    trace.set_data(xmR3[:i+1], ymR3[:i+1])
    trace2.set_data(x[:i+1], y[:i+1])
    # dit is hier i+1 omwille van gebruik van de slicing operator
    # Huidig punt
    point.set_data([xmR3[i]], [ymR3[i]])
    middelpuntR2.set_data([xmR2[i]], [ymR2[i]])
    middelpuntR3.set_data([xmR3[i]], [ymR3[i]])
    circleR2.center = (xmR2[i], ymR2[i])
    circleR3.center = (xmR3[i], ymR3[i])
    xi,yi,xmR2i,ymR2i,xmR3i,ymR3i,xstraalR2i,ystraalR2i=x[i],y[i],xmR2[i],ymR2[i],xmR3[i],ymR3[i],xstraalR2[i],ystraalR2[i]
    straalR1.set_data([0,xmR2i], [0,ymR2i])
    straalR2.set_data([xmR2i,xstraalR2i], [ymR2i,ystraalR2i])
    straalR3.set_data([xmR3i,xi], [ymR3i,yi])
    # Geef alle geüpdatete artists terug (vereist voor blit)
    #return trace, trace2, point, middelpuntR2, circleR2, circleR3, straalR2, straalR1, straalR3
    return trace2, point, middelpuntR2, circleR2, circleR3, straalR2, straalR1, straalR3
# ========================
# Start animatie
# ========================
ani = FuncAnimation(
    fig, update,
    frames=N,
    init_func=init,
    interval=1000/fps,  # milliseconden per frame
    blit=True,
    repeat=True
)

#plt.tight_layout()
plt.show()

# ========================
# Optioneel: opslaan
# ========================
# Als GIF (vereist: Pillow): pip install pillow
#ani.save('spirostar.gif', writer='pillow', fps=fps)

# Als MP4 (vereist: ffmpeg geïnstalleerd op het systeem)
#ani.save('cirkel.mp4', writer='ffmpeg', fps=fps)
#ani.save('cirkel.mp4', writer='pillow', fps=fps)
