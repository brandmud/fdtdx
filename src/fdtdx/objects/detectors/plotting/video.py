import multiprocessing as mp
import tempfile
from functools import partial
from typing import Callable

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import moviepy as mpy
import numpy as np
import cv2
from rich.progress import Progress

from fdtdx.objects.detectors.plotting.plot2d import plot_2d_from_slices

import imageio.v2 as iio

mp.set_start_method("spawn", force=True)

def plot_from_slices(
    slice_tuple: tuple[np.ndarray, np.ndarray, np.ndarray],
    resolutions: tuple[float, float, float],
    minvals: tuple[float, float, float],
    maxvals: tuple[float, float, float],
    plot_dpi: int | None,
    plot_interpolation: str,
):
    """Creates a figure from 2D slices at a specific timestep using shared memory arrays.

    Args:
        slice_tuple: tuple of array slices in order xy, xz, yz
        resolutions: Tuple of (dx, dy, dz) spatial resolutions in meters
        minvals: Tuple of minimum values for colormap scaling
        maxvals: Tuple of maximum values for colormap scaling
        plot_dpi: DPI resolution for the figure. None uses default.
        plot_interpolation: Interpolation method for imshow ('gaussian', 'nearest', etc)

    Returns:
        numpy.ndarray: RGB image data of the rendered figure
    """
    # xy_slice = sa.attach("shm://xy")[t, :, :]
    # xz_slice = sa.attach("shm://xz")[t, :, :]
    # yz_slice = sa.attach("shm://yz")[t, :, :]
    xy_slice, xz_slice, yz_slice = slice_tuple

    fig = plot_2d_from_slices(
        xy_slice=xy_slice,
        xz_slice=xz_slice,
        yz_slice=yz_slice,
        resolutions=resolutions,
        minvals=minvals,
        maxvals=maxvals,
        plot_dpi=plot_dpi,
        plot_interpolation=plot_interpolation,
    )
    # Convert matplotlib figure to a numpy array
    fig.canvas.draw()
    # Get the canvas dimensions, accounting for device pixel ratio
    width, height = fig.canvas.get_width_height()
    device_pixel_ratio = fig.canvas.device_pixel_ratio
    if device_pixel_ratio != 1:
        width = int(width * device_pixel_ratio)
        height = int(height * device_pixel_ratio)

    try:
        data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore
        data = data.reshape(height, width, 4)
        data = data[:, :, :3]  # remove alpha channel
    except AttributeError:
        # Fall back to tostring_argb method
        data = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)  # type: ignore
        data = data.reshape((height, width, 4))
        data = data[:, :, 1:]  # Remove alpha channel
    plt.close(fig)
    return data


def _make_animation_frame(t: float | int, precomputed_figs, fps):
    """Creates a single frame for the video animation.

    Args:
        t: Time point in seconds
        precomputed_figs: List of precomputed figure arrays
        fps: Frames per second of the video

    Returns:
        numpy.ndarray: RGB image data for the frame at time t
    """
    t = int(t * fps)
    fig = precomputed_figs[t]
    return fig

def _normalize_and_colorize(arr, vmin, vmax):
    norm = np.clip((arr - vmin) / (vmax - vmin), 0, 1)
    norm_uint8 = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def _fast_render_frame(
    slice_tuple: tuple[np.ndarray, np.ndarray, np.ndarray],
    vmins: tuple[float, float, float],
    vmaxs: tuple[float, float, float],
):
    xy, xz, yz = slice_tuple

    xy_img = _normalize_and_colorize(xy, vmins[0], vmaxs[0])
    xz_img = _normalize_and_colorize(xz, vmins[1], vmaxs[1])
    yz_img = _normalize_and_colorize(yz, vmins[2], vmaxs[2])

    h = max(xy_img.shape[0], xz_img.shape[0], yz_img.shape[0])
    xy_img = cv2.resize(xy_img, (xy_img.shape[1], h))
    xz_img = cv2.resize(xz_img, (xz_img.shape[1], h))
    yz_img = cv2.resize(yz_img, (yz_img.shape[1], h))

    combined = np.concatenate([xy_img, xz_img, yz_img], axis=1)

    scale = 2  # or 3 for 3x resolution
    combined = cv2.resize(combined, (combined.shape[1]*scale, combined.shape[0]*scale), interpolation=cv2.INTER_CUBIC)
    return combined


def generate_video_from_slices(
    xy_slice: np.ndarray,
    xz_slice: np.ndarray,
    yz_slice: np.ndarray,
    plt_fn,  # Unused but kept for compatibility
    resolutions: tuple[float, float, float],
    num_worker: int | None,
    plot_interpolation: str,
    plot_dpi: int | None,
    fps: int = 60,
    progress: Progress | None = None,
    minvals: tuple[float | None, float | None, float | None] = (None, None, None),
    maxvals: tuple[float | None, float | None, float | None] = (None, None, None),
):
    # Ensure JAX arrays are converted to NumPy (host) arrays
    if hasattr(xy_slice, "block_until_ready"):
        xy = np.asarray(xy_slice.block_until_ready())
        xz = np.asarray(xz_slice.block_until_ready())
        yz = np.asarray(yz_slice.block_until_ready())
    else:
        xy = np.asarray(xy_slice)
        xz = np.asarray(xz_slice)
        yz = np.asarray(yz_slice)

    T = xy.shape[0]
    slices = [xy, xz, yz]

    # Min/max scaling setup
    minvals = tuple(s.min() if m is None else m for s, m in zip(slices, minvals))
    maxvals = tuple(s.max() if m is None else m for s, m in zip(slices, maxvals))

    slice_arr_list = [(xy[t], xz[t], yz[t]) for t in range(T)]
    render_fn = partial(_fast_render_frame, vmins=minvals, vmaxs=maxvals)

    if progress is None:
        progress = Progress()
    task = progress.add_task("Generating video", total=T)

    if num_worker:
        with mp.get_context("spawn").Pool(num_worker) as pool:
            frames = []
            for frame in pool.imap(render_fn, slice_arr_list):
                frames.append(frame)
                progress.update(task, advance=1)
    else:
        frames = []
        for s in slice_arr_list:
            frames.append(render_fn(s))
            progress.update(task, advance=1)

    progress.update(task, visible=False)

    # Write video
    _, path = tempfile.mkstemp(suffix=".mp4")
    with iio.get_writer(path, fps=fps, codec='libx264') as writer:
        for frame in frames:
            writer.append_data(frame)

    return path