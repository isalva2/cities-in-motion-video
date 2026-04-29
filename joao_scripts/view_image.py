import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image_path = r"C:\Users\joaob\Dropbox\Documents\hackaton_UIC\hackaton_project\data\W06E_12-02-2025\images\1764693301177844608-snapshot.jpg"

# --- verify image loads before anything else ---
try:
    img = mpimg.imread(image_path)
    print(f"✓ Image loaded: {img.shape}")
except Exception as e:
    print(f"✗ Failed to load image: {e}")
    raise

fig, ax = plt.subplots(figsize=(14, 8))
ax.imshow(img)
ax.set_title("Click on GCP features — crosswalk corners, lane markings, stop lines\n"
             "Right-click to undo last point | Close window when done",
             fontsize=11, color="white")
fig.patch.set_facecolor("#1a1a1a")
ax.set_facecolor("#1a1a1a")

clicked_points = []
markers        = []
texts          = []

def onclick(event):
    if event.inaxes != ax:
        return
    if event.button == 3:
        if clicked_points:
            clicked_points.pop()
            markers[-1].remove(); markers.pop()
            texts[-1].remove();   texts.pop()
            fig.canvas.draw()
            print(f"  ↩ Undone — {len(clicked_points)} point(s) remaining")
        return

    x, y = int(event.xdata), int(event.ydata)
    clicked_points.append([x, y])
    n = len(clicked_points)

    mk = ax.plot(x, y, 'ro', markersize=8, markeredgecolor='yellow',
                 markeredgewidth=1.5)[0]
    tx = ax.text(x + 10, y - 10, f"GCP {n}\n({x}, {y})",
                 color="yellow", fontsize=8,
                 bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=2))
    markers.append(mk)
    texts.append(tx)
    fig.canvas.draw()
    print(f"  GCP {n}: pixel ({x}, {y})")

def onclose(event):
    print("\n" + "="*50)
    print("IMAGE_POINTS = np.float32([")
    for i, (x, y) in enumerate(clicked_points):
        print(f"    [{x}, {y}],   # GCP {i+1}")
    print("])")
    print("="*50)
    print("\nPaste this into your reprojection script!")

fig.canvas.mpl_connect("button_press_event", onclick)
fig.canvas.mpl_connect("close_event", onclose)

plt.tight_layout()
plt.show()