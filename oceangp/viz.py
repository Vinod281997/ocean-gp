"""Plotting and animation helpers for the ocean flow project."""

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from oceangp.data import load_data, compute_speed, grid_size, magma_dark

DARK_BG = "#0b0b14"
LAND_COLOR = "#4a4036"


def _style_dark(fig, ax, cbar, bg=DARK_BG):
    """Apply dark-background styling to a figure, axis, and colorbar."""

    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.tick_params(colors="#cccccc")
    ax.xaxis.label.set_color("#dddddd")
    ax.yaxis.label.set_color("#dddddd")
    ax.title.set_color("#eeeeee")

    for spine in ax.spines.values():
        spine.set_color("#555555")

    cbar.ax.yaxis.set_tick_params(color="#cccccc")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#cccccc")
    cbar.outline.set_edgecolor("#555555")
    cbar.set_label("speed (km/h)", color="#dddddd")


def _draw_land(ax, land_mask, ny, nx, color=LAND_COLOR):
    """Overlay land cells as a solid muted patch above the flow plot."""

    if land_mask is None:
        return
    
    land_img = np.where(land_mask, 1.0, np.nan)  
    ax.imshow(land_img, origin="lower",
              extent=[0, nx * grid_size, 0, ny * grid_size],
              cmap=ListedColormap([color]), zorder=5)

def animate_speed(speed, fps=24, cmap="plasma", land_mask=None, dark=True, out_path=None):
    """Animate the speed heatmap over time.
    
    Args:
        speed: np.ndarray
            Flow velcity magnitudes, shape (n_frames, ny, nx).

        fps: int, optional
            Frames per second for the animation. Sets the
            inter-frame interval to 1000 / fps milliseconds. Default 24.

        cmap: str, optional
            Colormap for the heatmap. Default "plasma".

        out_path: str or Path, optional
            If provided, save the animation to this path.

    Returns:
        matplotlib.animation.FuncAnimation
        The animation object.

    """

    n_frames, ny, nx = speed.shape
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(speed[0], origin="lower", cmap=cmap, 
                   extent=[0, nx*grid_size, 0, ny*grid_size],
                   vmin=speed.min(), vmax=speed.max())
    ax.set(xlabel="x (km)", ylabel="y (km)")
    cbar =plt.colorbar(im, ax=ax, label="Speed (km/h)")
    fig.suptitle("Philippine Archipelago", color="#eeeeee")
    title = ax.set_title("Ocean Current Speed -> t = 0 hrs")
    _draw_land(ax, land_mask, ny, nx)                        
    if dark:
        _style_dark(fig, ax, cbar)

    def update(frame):
        im.set_data(speed[frame])
        title.set_text(f"Ocean Current Speed -> t = {frame*3} hrs")
        return im, title
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    if out_path:
        anim.save(out_path, writer=PillowWriter(fps=fps),
                  savefig_kwargs={"facecolor": fig.get_facecolor()})
    
    plt.close(fig)

    return anim

def animate_quiver(u, v, fps=24, stride=4, scale=None, land_mask=None, dark=True, out_path=None):
    """Animate the velocity vector field over time as color-coded arrows.
    
    Args:
        u, v: np.ndarray
            Horizontal and vertical velocity components, each of shape (n_frames, ny, nx).

        fps: int, optional
            Frames per second for the animation. Sets the
            inter-frame interval to 1000 / fps milliseconds. Default 24.

        stride: int, optional
            Subsampling factor for the quiver plot to avoid overcrowding. Default 4.

        scale: float, optional
            Scale for the quiver plot. If None, the scale is determined automatically. Default None.

        out_path: str or Path, optional
            If provided, save the animation to this path.
            
    Returns:
        matplotlib.animation.FuncAnimation
        The animation object.

    """

    n_frames, ny, nx = u.shape
    cmap = magma_dark() if dark else plt.cm.magma
    ix, iy = np.arange(0, nx, stride), np.arange(0, ny, stride)
    X, Y = np.meshgrid(ix*grid_size, iy*grid_size)
    us = u[:, ::stride, ::stride]
    vs = v[:, ::stride, ::stride]
    speed_sub = compute_speed(us, vs)

    fig, ax = plt.subplots(figsize=(10,8))
    q = ax.quiver(X, Y, u[0, ::stride, ::stride], v[0, ::stride, ::stride],
                  speed_sub[0], cmap=cmap, scale=scale, pivot="mid",
                  clim=(0, np.sqrt(u**2 + v**2).max()))
    ax.set(xlabel="x (km)", ylabel="y (km)", aspect="equal")
    cbar = fig.colorbar(q, ax=ax, label="Speed (km/h)")
    fig.suptitle("Philippine Archipelago", color="#eeeeee")
    title = ax.set_title("Ocean Current Direction -> t = 0 hrs")
    _draw_land(ax, land_mask, ny, nx)
    if dark:
        _style_dark(fig, ax, cbar)
    else:
        cbar.set_label("speed (km/h)")

    def update(frame):
        q.set_UVC(u[frame, ::stride, ::stride], v[frame, ::stride, ::stride], speed_sub[frame])
        title.set_text(f"Ocean Current Direction -> t = {frame*3} hrs")
        return q, title
    
    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000/fps, blit=False)

    if out_path:
        anim.save(out_path, writer=PillowWriter(fps=fps),
                  savefig_kwargs={"facecolor": fig.get_facecolor()})

    plt.close(fig)

    return anim

def animate_streamlines(u, v, fps=24, density=1.2, land_mask=None,
                        dark=True, out_path=None):
    """Animate the velocity vector field over time as streamlines.
    
    Args:
        u, v: np.ndarray
            Horizontal and vertical velocity components, each of shape (n_frames, ny, nx).

        fps: int, optional
            Frames per second for the animation. Sets the
            inter-frame interval to 1000 / fps milliseconds. Default 24.

        density: float, optional
            Controls the closeness of streamlines. Higher values result in more streamlines. Default 1.2.

        cmap: str, optional
            Colormap for the streamlines. Default "plasma".

        out_path: str or Path, optional
            If provided, save the animation to this path.
            
        Returns:
            matplotlib.animation.FuncAnimation
            The animation object.
        """
    
    n_frames, ny, nx = u.shape
    cmap = magma_dark() if dark else plt.cm.magma
    x = np.arange(nx) * grid_size
    y = np.arange(ny) * grid_size
    speed = compute_speed(u, v)
    norm = Normalize(vmin=speed.min(), vmax=speed.max())

    fig, ax = plt.subplots(figsize=(6,5))
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.suptitle("Philippine Archipelago", color="#eeeeee")
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("speed (km/h)", color="#dddddd" if dark else "black")
    if dark:
        fig.patch.set_facecolor(DARK_BG)
        cbar.ax.yaxis.set_tick_params(color="#cccccc")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#cccccc")
        cbar.outline.set_edgecolor("#555555")

    def update(frame):
        ax.clear()
        ax.streamplot(x, y, u[frame], v[frame], color=speed[frame],
                      cmap=cmap, norm=norm, density=density)
        _draw_land(ax, land_mask, ny, nx)
        ax.set(xlabel="x (km)", ylabel="y (km)", aspect="equal",
               title=f"Ocean Current Streamlines -> t = {frame * 3} hr")
        
        if dark:
            ax.set_facecolor(DARK_BG)
            ax.tick_params(colors="#cccccc")
            ax.xaxis.label.set_color("#dddddd")
            ax.yaxis.label.set_color("#dddddd")
            ax.title.set_color("#eeeeee")
            for spine in ax.spines.values():
                spine.set_color("#555555")

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps, blit=False)

    if out_path:
        anim.save(out_path, writer=PillowWriter(fps=fps),
                  savefig_kwargs={"facecolor": fig.get_facecolor()})

    plt.close(fig)
    
    return anim
