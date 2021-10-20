import plot_helper as ph
import matplotlib
import matplotlib.pyplot as plt
import colors as cc

h5f_50_70 = ph.ah.prepare_file("./dat/inhibition_sweep_rate_160/dyn/stim=off_k=5_jA=50.0_jG=50.0_jM=15.0_rate=70.0_rep=000.hdf5")
h5f_50_90 = ph.ah.prepare_file("./dat/inhibition_sweep_rate_160/dyn/stim=off_k=5_jA=50.0_jG=50.0_jM=15.0_rate=90.0_rep=000.hdf5")
h5f_50_110 = ph.ah.prepare_file("./dat/inhibition_sweep_rate_160/dyn/stim=off_k=5_jA=50.0_jG=50.0_jM=15.0_rate=110.0_rep=000.hdf5")

def save_ax(ax, name, **kwargs):
    kwargs.setdefault("dpi", 300)
    kwargs.setdefault("transparent", True)

    fig = ax.get_figure()
    fig.savefig(f"./fig/paper/{name}.pdf", **kwargs)


ax = ph.plot_raster(h5f_50_70, markersize=1.0)
ax.set_xlim(120, 210)
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(10))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(20))
ax.xaxis.set_major_formatter(cc.get_shifted_formatter(shift=-120, fmt=".0f"))
ax.set_title("70 Hz", loc="Right")
ax.set_xlabel("")
ax.set_ylabel("")
save_ax(ax, "raster_50_70_90sec")

ax = ph.plot_raster(h5f_50_90, markersize=1.0)
ax.set_xlim(90, 120)
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(5))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(10))
ax.xaxis.set_major_formatter(cc.get_shifted_formatter(shift=-90, fmt=".0f"))
ax.set_title("90 Hz", loc="Right")
ax.set_xlabel("")
ax.set_ylabel("")
save_ax(ax, "raster_50_90_30sec")


ax = ph.plot_raster(h5f_50_110, markersize=1.0)
ax.set_xlim(90, 100)
ax.xaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
ax.xaxis.set_major_formatter(cc.get_shifted_formatter(shift=-90, fmt=".0f"))
ax.set_title("110 Hz", loc="Right")
save_ax(ax, "raster_50_110_10sec")
