import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
import time
from datetime import timedelta
import sys

# Parameters - plot 영역 설정 관련
zoom_step = 500  # 한 번에 확대할 배율
x0, y0 = 0, 0  # (x0, y0): plot 영역 중심 좌표
eps = 5e0  # x0 좌우로 eps만큼 plot 함
eps_y = eps * (9 / 16)  # 16:9 비율에 맞추기 위해 y축 eps 계산
n = 1920  # 화소수 조절을 위한 parameter (3840:4K, 1920:Full HD)
nx, ny = n, int(n * (9 / 16))  # nx, ny: x, y축 화소수

# Parameters - tetration 계산 관련
max_iter = 1000  # 최대 몇 층까지 계산할 것인지를 정함
escape_radius = 1e+10  # 복소수 크기가 escape_radius를 벗어나면 발산한 것으로 처리함

clicked = False

def on_click(event):
    global clicked, x0, y0, eps, eps_y, zoom_step
    if event.button == 1 and event.inaxes:  # Only respond to left mouse clicks
        print(f'Mouse clicked at: x={event.xdata}, y={event.ydata}, Zooming in {zoom_step}x')
        prevData.append([x0, y0, eps, eps_y])
        x0, y0 = event.xdata, event.ydata
        eps /= zoom_step
        eps_y = eps * (9 / 16)
        clicked = True
    elif event.button == 3 and event.inaxes:
        print(f"Moving back to: x={prevData[-1][0]}, y={prevData[-1][1]}, Zooming out {zoom_step}x")
        x0, y0, eps, eps_y = prevData.pop()
        clicked = True

def on_close(event):
    print("Close button clicked. Exiting...")
    sys.exit()

def seconds_to_hms(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    fractional_seconds = td.total_seconds() - total_seconds
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60 + fractional_seconds
    return f"{hours:02} hours : {minutes:02} minutes : {seconds:09.6f} seconds"

prevData = []
prevData.append([x0, y0, eps, eps_y])
fig, ax = plt.subplots()
cid = fig.canvas.mpl_connect('button_press_event', on_click)
cid_close = fig.canvas.mpl_connect('close_event', on_close)

@jit(nopython=True, parallel=True, fastmath=True)
def compute_tetration_divergence(c, nx, ny, max_iter, escape_radius):
    divergence_map = np.zeros((nx, ny), dtype=np.bool_)
    for i in prange(nx):
        for j in prange(ny):
            c_val = c[i, j]
            z = c_val
            for k in range(max_iter):
                z = np.power(c_val, z)
                if np.abs(z) > escape_radius:
                    divergence_map[i, j] = True
                    break
    return divergence_map

while True:
    s = time.time()
    x = np.linspace(x0 - eps, x0 + eps, nx)
    y = np.linspace(y0 - eps_y, y0 + eps_y, ny)
    c = x[:, np.newaxis] + 1j * y[np.newaxis, :]
    divergence_map = compute_tetration_divergence(c, nx, ny, max_iter, escape_radius)

    plt.clf()
    plt.imshow(divergence_map.T, extent=[x0 - eps, x0 + eps, y0 - eps_y, y0 + eps_y], origin='lower', cmap='gray')
    plt.axis('off')
    filename = f"mytetration_x_{x0}_y_{y0}_eps_{eps}.png"
    plt.savefig(filename, dpi=600, bbox_inches='tight', pad_inches=0)

    e = time.time()
    print("Delta time>> " + seconds_to_hms(e - s))

    while not clicked:
        plt.pause(0.1)

    clicked = False
