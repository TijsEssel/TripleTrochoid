import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import streamlit as st

st.title("Interactieve Tripletrochoïde")

# Sliders voor parameters
R1 = st.slider("R1 (straal 1)", 1.0, 20.0, 6.0, 0.1)
R2 = st.slider("R2 (straal 2)", -20.0, 20.0, 8.0, 0.1)
R3 = st.slider("R3 (straal 3)", -10.0, 10.0, 1.0, 0.1)
a = st.slider("a (verhouding)", 0.0, 2.0, 7/12, 0.01)
d = st.slider("d (afstand)", 0.0, 50.0, 1.0, 0.01)

# Overige vaste parameters
f1 = 1 * np.pi
f0 = 0 * np.pi
up = d
vp = 0
fps = 60
omwentelingen = 7 * abs(R1)
N = int(fps * 500)
t = np.linspace(0, omwentelingen * 2 * np.pi, N)
t1 = -a * (R1 / R2) * t

# Hypotrochoïde berekeningen (zoals in je script)
u = (R2 + R3) * np.cos(t1 + f1) + np.cos((1 + (R2 / R3)) * t1) * up - np.sin((1 + (R2 / R3)) * t1) * vp
v = (R2 + R3) * np.sin(t1 + f1) + np.sin((1 + (R2 / R3)) * t1) * up + np.cos((1 + (R2 / R3)) * t1) * vp
x = (R1 + R2) * np.cos(t + f0) + np.cos((1 + (R1 / R2)) * t) * u - np.sin((1 + (R1 / R2)) * t) * v
y = (R1 + R2) * np.sin(t + f0) + np.sin((1 + (R1 / R2)) * t) * u + np.cos((1 + (R1 / R2)) * t) * v

# Plotten
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, color='maroon',linewidth='0.5')
ax.set_aspect('equal')
ax.grid(True, ls='--', alpha=0.4)
ax.set_title("Hypotrochoïde")
st.pyplot(fig)







