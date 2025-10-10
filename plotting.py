import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
import transformer_lens as tl
import torch, gc, os
import plotly.graph_objects as go
from datetime import datetime
from processResults import get_taskVec
from LabelExtractor import infer_entities
import utils

### ENTITY LENS PLOTTING ###


def EntityLens(
    model: HookedTransformer,
    results:pd.DataFrame,
    layers: list,
    prompt: str = "The City of Lights iconic landmark",
    with_context: bool = False,
    use_TV_layer: bool = None,
    verbose: bool = False,
    compute_logit_lens: bool = False,
    output: str = "fancy",
    plot_size: tuple = (600, 5000),
):
    """
    Compute the Entity Lens for a given model and prompt.
    Args:
        model (HookedTransformer): The model to use.
        results (pd.DataFrame): dataframe containing the provided results.
        layers (list): The layers to use.
        prompt (str): The prompt to use.
        with_context (bool): Whether to use context or not.
        use_TV_layer (int): what Task Vector to use.
            - default is None: use the TV from each layer,
            - if int, use the TV from that layer for all layers.
        use_filter (bool): Whether to use the linear filter or not.
        verbose (bool): Whether to print verbose output or not.
        compute_logit_lens (bool): Whether to compute the logit lens or not.
        output (str): The output format. Can be
            - "fancy": fancy table using plotly
            - "markdown": markdown table
            - "csv": csv file
    """
    if verbose:
        print(
            f"Computing {'contextual' if with_context else 'uncontextual'} Entity Lens at layers {layers} of {model.cfg.model_name}"
        )

    str_tokens = model.to_str_tokens(prompt)
    tok_ids = range(1, len(str_tokens))  # skip the first token
    str_tokens = [str_tokens[t] for t in tok_ids]
    dtype = model.W_E.dtype
    model_name = model.cfg.model_name

    # get whole cache
    with torch.no_grad():
        _, cache = model.run_with_cache(prompt)
        reprs = []
        for layer in layers:
            if layer == -1:
                hook = tl.utils.get_act_name("embed")
            else:
                hook = tl.utils.get_act_name("resid_post", layer, "")
            reprs.append(cache[hook].detach().cpu())  # 1 x n_tokens x dim
        del cache
        gc.collect()
        torch.cuda.empty_cache()

    def get_TV(layer, dtype=torch.float32):
        # This function is assumed to be defined elsewhere in your code
        fileName = get_taskVec(
            results,
            model_name,
            layer=layer,
            with_context=with_context,
            verbose=False,
        )
        if verbose:
            print(f" layer {layer}, TaskVec loaded from {fileName}")
        TaskVec = torch.load(fileName, weights_only=True).to(dtype)
        return TaskVec

    if use_TV_layer is not None:
        TaskVec = get_TV(use_TV_layer, dtype=dtype)
        if verbose:
            print(f"Using TaskVec from layer {use_TV_layer}")
    else:
        if verbose:
            print("will load TaskVec from every layer")

    if verbose:
        print(
            f"infering entities from all tokens in prompt with{'' if with_context else 'out'} context..."
        )
    
    
    data = []
    for i, l in enumerate(layers):
        if use_TV_layer is None:
            TaskVec = get_TV(l, dtype=dtype)

        data_l = [
            {"representation": reprs[i][0, t, :], "layer": l, "text": prompt}
            for t in tok_ids
        ]

        for i, d in enumerate(data_l):
            d["id"] = i

        infer_entities(
            model,
            TaskVec,
            data_l,
            with_context=with_context,
            max_tokens=8,
            b_size=5,
            verbose=False,
        )

        if compute_logit_lens:
            for row in data_l:
                # This function is assumed to be defined elsewhere in your code
                row["logitLens"] = utils.project_on_vocab(model, row["representation"], k = 1)
        
        data += data_l

    def format_cell_data(d, for_html=False):
        """Helper function to format cell data based on output type"""
        if for_html:
            # HTML formatting with styling
            cell_text = f"<b style='font-size: 13px; color: #2c3e50;'>{d['inferred']}</b>"
            if compute_logit_lens and "logitLens" in d:
                logit_lens_text = d['logitLens'].strip().replace(' ', '&nbsp;')
                cell_text += f"<br><span style='color:#7f8c8d; font-size:10px; font-style:italic;'>'{logit_lens_text}'</span>"
            return cell_text
        else:
            # Plain text formatting for markdown/console output
            text = d['inferred']
            if compute_logit_lens and "logitLens" in d:
                logit_lens_text = d['logitLens'].strip()
                text += f" (ll: '{logit_lens_text}')"
            return text
    
    strings = []
    
    # Generate string data based on output format
    for l in layers:
        layer_data = [d for d in data if d["layer"] == l]
        if output == "html":
            # Use HTML formatting
            row_strings = [format_cell_data(d, for_html=True) for d in layer_data]
        else:
            # Use plain text formatting
            row_strings = [format_cell_data(d, for_html=False) for d in layer_data]
        strings.append(row_strings)

    row_headers = [f"{l+1}" for l in layers]
    if layers and layers[0] == -1:
        row_headers[0] = "Emb"

    if output == "markdown":
        print()
        print(f"Entity Lens for {model.cfg.model_name}")
        width = 30
        sep = ";"
        print()
        print("-" * (width + 1) * (len(str_tokens) + 1))
        print(sep.join([tok.center(width) for tok in ["token"] + str_tokens]))
        print("-" * (width + 1) * (len(str_tokens) + 1))
        for i, row in enumerate(strings):
            # Clean markdown output without HTML tags
            print(sep.join([it.center(width) for it in row_headers[i : i + 1] + row]))
        print("-" * (width + 1) * (len(str_tokens) + 1))

    elif output == "fancy":
        # MODIFICATION 2: Enhance the Plotly table presentation
        header = ["Layer"] + str_tokens
        cells = [[row] + data for row, data in zip(row_headers, strings)]
        columns = list(map(list, zip(*cells)))

        size = 50
        h, w = plot_size

        # Create a more visually appealing header
        header_style = dict(
            values=header,
            fill_color="#1f77b4",  # A professional blue
            align="center",
            font=dict(size=size + 5, family="Arial, sans-serif", color="white"),
            height=int(1.6 * size),
        )

        # Style the cells with alternating row colors for readability
        cell_style = dict(
            values=columns,
            fill_color=[['#f2f2f2', 'white'] * (len(columns[0]) // 2 + 1)],
            line_color="darkgray",
            align="center",
            font=dict(size=size, family="Arial, sans-serif"),
            height=int(1.7 * size), # Increased height to accommodate two lines
        )

        fig = go.Figure(
            data=[
                go.Table(
                    columnwidth=[10] + [50] * len(str_tokens),
                    header=header_style,
                    cells=cell_style,
                )
            ]
        )
        
        # Add a title and refine the layout for a polished look
        fig.update_layout(
            title_text=f"<b>Entity Lens Analysis</b><br><i>Prompt: '{prompt}'</i>",
            title_x=0.5,
            margin=dict(t=120, b=20, l=20, r=20), # Increased top margin for title
            height=h,
            width=w,
        )
        fig.show()
        
    elif output == "html":
        # MODIFICATION 3: Create a beautiful HTML table
        from IPython.display import HTML, display
        
        # Create HTML table - compact design for paper publication
        html = f"""
        <div style="font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif; margin: 5px; font-size: 14px;">
            <h3 style="text-align: center; color: #2c3e50; margin-bottom: 5px; font-size: 18px; font-weight: 600;">Entity Lens Analysis</h3>
            <p style="text-align: center; color: #555; font-style: italic; margin-bottom: 8px; font-size: 12px;">Prompt: "{prompt}"</p>
            
            <table style="border-collapse: collapse; width: auto; margin: 0 auto; border: 2px solid #2c3e50; font-size: 13px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); table-layout: auto;">
                <thead>
                    <tr style="background-color: #3498db; color: white;">
                        <th style="padding: 3px 5px; border: 1px solid #2c3e50; font-weight: 600; text-align: center; width: 60px; font-size: 13px;">Layer</th>
        """
        
        # Calculate optimal column widths based on content
        column_widths = []
        for i, token in enumerate(str_tokens):
            # Start with token header length
            max_length = len(token)
            
            # Check content in each row for this column
            for row_data in strings:
                if i < len(row_data):
                    # Remove HTML tags and calculate text length
                    clean_text = row_data[i].replace('<b>', '').replace('</b>', '').replace('<br>', ' ').replace('<span>', '').replace('</span>', '')
                    # Split by spaces and take the longest part for width calculation
                    words = clean_text.split()
                    if words:
                        max_word = max(words, key=len)
                        max_length = max(max_length, len(max_word))
            
            # Calculate width: base width + character width, with tighter limits
            width = max(50, min(150, max_length * 7 + 15))
            column_widths.append(width)
        
        # Add token headers with calculated widths
        for i, token in enumerate(str_tokens):
            html += f'<th style="padding: 3px 4px; border: 1px solid #2c3e50; font-weight: 600; text-align: center; width: {column_widths[i]}px; font-size: 13px;">{token}</th>'
        
        html += """
                    </tr>
                </thead>
                <tbody>
        """
        
        # Add data rows
        for i, (row_header, row_data) in enumerate(zip(row_headers, strings)):
            row_color = "#f8f9fa" if i % 2 == 0 else "#ffffff"
            html += f'<tr style="background-color: {row_color};">'
            html += f'<td style="padding: 3px 5px; border: 1px solid #2c3e50; text-align: center; font-weight: 600; background-color: #e8f4fd; color: #2c3e50; font-size: 13px;">{row_header}</td>'
            
            for cell_data in row_data:
                html += f'<td style="padding: 2px 4px; border: 1px solid #2c3e50; text-align: center; vertical-align: middle; line-height: 1.2; color: #2c3e50; font-size: 11px;">{cell_data}</td>'
            
            html += '</tr>'
        
        html += """
                </tbody>
            </table>
        </div>
        """
        
        display(HTML(html))
        
        # Save HTML to file for printing
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"entity_lens_{model.cfg.model_name.replace('/', '_')}_{timestamp}.html"
        
        # Create a complete HTML document for standalone viewing/printing
        complete_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Entity Lens Analysis</title>
    <style>
        @media print {{
            body {{ margin: 0; }}
            .no-print {{ display: none; }}
        }}
        body {{
            font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            margin: 20px;
            background: white;
        }}
    </style>
</head>
<body>
    {html.replace('<div style="font-family:', '<div class="entity-table" style="font-family:')}
    <div class="no-print" style="margin-top: 20px; text-align: center;">
        <p style="color: #666; font-size: 12px; margin-top: 10px;">
            Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Model: {model.cfg.model_name}
        </p>
    </div>
</body>
</html>"""
        
        # Save to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(complete_html)
        
        print(f"✅ HTML table saved to: {filename}")
        print(f"📄 You can open this file in a browser and print it directly.")
        
    else :
        print("Invalid output format. Choose from 'fancy', 'markdown', or 'html'. Ignoring output.")
    
    return data

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

