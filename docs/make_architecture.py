"""Render docs/architecture.png (run: python docs/make_architecture.py)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

BOXES = {  # name: (x, y, w, colour)
    "UCI dataset\n(download script)": (0.2, 6.2, "#DDEBF7"),
    "EDA + cleaning\nsrc/data.py, src/eda.py": (3.0, 6.2, "#DDEBF7"),
    "Training + tuning\nsrc/train.py": (5.8, 6.2, "#DDEBF7"),
    "MLflow tracking\n(params, metrics, plots)": (8.6, 6.2, "#FFF2CC"),
    "Model artifact\nmodels/model.joblib": (5.8, 4.2, "#E2EFDA"),
    "GitHub\nrepository": (0.2, 4.2, "#EDEDED"),
    "GitHub Actions CI\nlint > test > train > docker": (3.0, 4.2, "#FCE4D6"),
    "Docker image\nFastAPI /predict /metrics": (5.8, 2.2, "#E2EFDA"),
    "Kubernetes\nDeployment x2 + LoadBalancer": (3.0, 2.2, "#E2EFDA"),
    "Clients\n(curl / apps)": (0.2, 2.2, "#EDEDED"),
    "Prometheus\n(scrapes /metrics)": (8.6, 2.2, "#FFF2CC"),
    "Grafana dashboard\n+ request logs": (8.6, 0.4, "#FFF2CC"),
}
EDGES = [(0, 1), (1, 2), (2, 3), (2, 4), (5, 6), (6, 4), (4, 7), (7, 8), (9, 8), (7, 10), (10, 11)]

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.set_xlim(0, 11.6)
ax.set_ylim(0, 7.6)
ax.axis("off")
W, H = 2.6, 1.0
names = list(BOXES)
for n, (x, y, c) in BOXES.items():
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.05", fc=c, ec="#444"))
    ax.text(x + W / 2, y + H / 2, n, ha="center", va="center", fontsize=9)
for a, b in EDGES:
    (x1, y1, _), (x2, y2, _) = BOXES[names[a]], BOXES[names[b]]
    p1 = (x1 + W / 2, y1 + H / 2)
    p2 = (x2 + W / 2, y2 + H / 2)
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    if abs(dx) >= abs(dy):  # horizontal-ish: leave from side
        s = 1 if dx > 0 else -1
        p1, p2 = (p1[0] + s * W / 2, p1[1]), (p2[0] - s * W / 2, p2[1])
    else:
        s = 1 if dy > 0 else -1
        p1, p2 = (p1[0], p1[1] + s * H / 2), (p2[0], p2[1] - s * H / 2)
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="->", color="#333", lw=1.4))
ax.set_title("Heart Disease MLOps - architecture", fontsize=13, weight="bold")
fig.savefig("docs/architecture.png", dpi=160, bbox_inches="tight")
