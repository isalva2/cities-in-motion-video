import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation
from matplotlib.patches import Polygon


def plot_image(filename, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    img = mpimg.imread(filename)
    ax.imshow(img)
    ax.axis('off')
    return fig, ax


def get_image_shape(filename):
    "Returns WIDTH THEN HEIGHT of image"
    img = mpimg.imread(filename)
    return img.shape[1], img.shape[0]


def add_text(ax, text, x, y, **kwargs):
    ax.text(float(x), float(y), text,
        color='black', bbox=dict(facecolor='white', edgecolor='none', pad=3),
        **kwargs)


def add_polygon(ax, vertices, **kwargs):
    polygon = Polygon(vertices,
        **kwargs)
    ax.add_patch(polygon)

def export_video(timestamps_ns, file_paths, hashed_jsonl_data, output_path="output.mp4", fps=5, figsize=(10, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')

    def update(frame):
        timestamp_ns = timestamps_ns[frame]
        file_name = file_paths[frame]
        hr_timestamp = hashed_jsonl_data.get(timestamp_ns).get("timestamp")

        ax.clear()
        ax.axis('off')
        img = mpimg.imread(file_name)
        ax.imshow(img)
        add_text(ax, hr_timestamp, 100, 100)

    ani = animation.FuncAnimation(fig, update, frames=len(timestamps_ns), repeat=False)
    ani.save(output_path, writer=animation.FFMpegWriter(fps=fps))
    plt.close(fig)