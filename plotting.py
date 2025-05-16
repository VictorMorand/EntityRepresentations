import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np
import matplotlib.pyplot as plt

### ENTITY LENS PLOTTING ###

def create_colored_table(
            strings, 
            headers, 
            row_headers, 
            data = None, 
            colorscale='Viridis', 
            title= "Label_Lens", 
            min_val=None, 
            max_val=None):
    """
    Create a colored table with the given data, strings, headers, and row headers.
    The data is colored using the given colorscale, and the colors are normalized
    between min_val and max_val. If min_val and max_val are not provided, the min
    and max of the data are used.
    Args: 
        data: 2D array of data to color
        strings: 2D array of strings to display in the table
    """

    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)
    assert len(row_headers) == len(strings)


    # Normalize the data
    if data is None:
        norm_data = np.zeros((len(strings), len(strings[0])))
    else: 
        norm_data = (np.array(data) - min_val) / (2 * (max_val - min_val))

    # Generate colors using normalized data
    colors = [[pc.sample_colorscale(colorscale, norm_val, colortype='rgb')[0] for norm_val in row] for row in norm_data]

    #transpose the colors array
    headers = ['tokens'] + headers
    white = 'rgb(255,255,255)'
    colors = [[white] + colors_row for colors_row in colors]
    
    colors = np.transpose(colors)
    norm_data = np.transpose(norm_data)
    strings = np.transpose([[f'<b>{row_headers[i]}</b>'] + strings[i] for i in range(len(strings))])

    # Set the text colors, with a different color for the first row
    text_colors = ['black'] + ['black'] * (len(strings) - 1)  # Black for the first row, white for others
    CELL_HEIGHT = 60
    # Create the table
    fig = go.Figure(
        data=[go.Table(
        header=dict(
            values=[f'<b>{h}</b>' for h in headers],
            line_color='black', fill_color='white',
            line_width=2, 
            align='center',font=dict(color='black', size=12)
        ),
        cells=dict(
            values=strings,
            line_color='gray',  # Set the line color for the grid
            line_width=1,  # Set the line width for the grid
            fill_color=colors,
            align='center',
            # font=dict(color=text_colors, size=12),
            font=dict(family="Computer Modern", color=text_colors, size=12),  # Set font to LaTeX's default
            height=CELL_HEIGHT
        ),
        #columnwidth=[85] * len(strings[0]),
    )])
    # Update layout
    fig.update_layout(
        title_text=title,
        title_x=0.5,  # Center the title
        height=len(strings[0]) * CELL_HEIGHT + 100,  # Adjust height based on number of rows
        width=len(strings) * 100,  # Adjust width based on number of columns
        margin=dict(l=20, r=20, t=50, b=10),
    )

    return fig


def plot_hist(hist, n_smooth = 20, title=""):
    """Plot history of training"""
    #compute smoothed loss

    #relpace 'val_loss' by 'val_metric' in hist if needed
    for h in hist:
        if "val_loss" in h:
            h["val_metric"] = h.pop("val_loss")
    
    keys = ["smooth_loss", "val_metric", "loss", "lr" ]
    plots = {}
    for h in hist :
        for key in keys:
            if key in h:
                plots[key] = plots.get(key, [])
                plots[key].append( (h["samples"], h[key]) )
    
    #smooothing 'loss' and add in hist 
    plots["smooth_loss"] = [ (hist[i]["samples"], np.mean([h["loss"] for h in hist[i:i+n_smooth]])) for i in range(len(hist)-n_smooth)]

    # loss_smooth = np.convolve(loss, np.ones(n_smooth)/n_smooth, mode='same')

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:blue'
    ax1.set_xlabel('Step')
    ax1.set_ylabel('BCE Loss', color=color)
    ax1.plot(*zip(*plots["smooth_loss"]), label="smooth_loss", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale("log")

    color = 'tab:red'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation Metric', color=color)
    ax2.plot(*zip(*plots["val_metric"]), label="val_metric", color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    min_val, max_val = ax2.get_ylim()
    if max_val > 10:
        ax2.set_yscale("log")

    color = 'tab:green'
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('lr', color=color)
    ax3.plot(*zip(*plots["lr"]), label="lr", color=color, linestyle='-.')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_yscale("log")
    
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')
    fig.tight_layout()
    plt.title(title)
    plt.show()

